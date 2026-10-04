"""Multi-provider LLM client (Gemini or Anthropic): retries with backoff, rate-limit throttle, response cache,
JSON parsing with one-shot repair, and token-usage tracking. The rest of the project only calls complete()/complete_json()."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

from .config import Prompts, resolve_path

log = logging.getLogger(__name__)

KEY_ENV_FALLBACKS = {"gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "anthropic": ["ANTHROPIC_API_KEY"]}
PROVIDERS = tuple(KEY_ENV_FALLBACKS)


class LLMError(RuntimeError):
    pass


class EmptyResponse(LLMError):
    """Model answered with no text (e.g. finish_reason=MAX_TOKENS because thinking ate the budget)."""


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


_DAILY_QUOTA = re.compile(r"PerDay|per day|daily", re.I)


def quota_reset_seconds(message: str) -> float | None:
    """Seconds until a quota resets, from 'Please retry in 17h40m45.4s' or \"'retryDelay': '63645s'\"."""
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
    if m and any(m.groups()):
        return int(m[1] or 0) * 3600 + int(m[2] or 0) * 60 + float(m[3] or 0)
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", message)
    return float(m[1]) if m else None


def _fmt_wait(seconds: float | None) -> str:
    if not seconds:
        return "unknown time"
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"~{h}h {m}m" if h else f"~{m}m"


class LLMClient:
    def __init__(self, cfg: dict, prompts: Prompts):
        self.cfg = cfg["llm"]
        self.prompts = prompts
        self.provider = str(self.cfg.get("provider", "gemini")).lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown llm.provider '{self.provider}'. Use one of: {', '.join(PROVIDERS)}")
        self.usage: Counter = Counter()
        self.last_model: str = self.cfg["model"]  # the model that actually answered (differs after a fallback)
        self._client = None
        self._last_call = 0.0
        self._exhausted: dict[str, float] = {}  # model -> monotonic time its daily quota resets
        self._cache_dir: Path | None = None
        if self.cfg.get("cache_enabled"):
            self._cache_dir = resolve_path(cfg, self.cfg.get("cache_dir", ".cache"))
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self.cfg["model"]

    # ------------------------------------------------------------ setup
    def _api_key(self) -> str:
        names = [n for n in [self.cfg.get("api_key_env"), *KEY_ENV_FALLBACKS[self.provider]] if n]
        for name in names:
            if os.getenv(name):
                return os.environ[name]
        raise LLMError(f"API key not found. Copy .env.example to .env and set {KEY_ENV_FALLBACKS[self.provider][0]}.")

    def _get_client(self):
        if self._client is not None:
            return self._client
        key, timeout = self._api_key(), float(self.cfg.get("timeout_seconds", 60))
        if self.provider == "gemini":
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise LLMError("The 'google-genai' package is not installed (pip install -r requirements.txt)") from e
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        else:
            try:
                import anthropic
            except ImportError as e:
                raise LLMError("The 'anthropic' package is not installed (pip install anthropic)") from e
            self._client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=0)
        return self._client

    # ------------------------------------------------------------ helpers
    def _throttle(self):
        """Stay under the provider's requests-per-minute limit (free tiers are small)."""
        rpm = self.cfg.get("requests_per_minute")
        if rpm:
            wait = 60.0 / float(rpm) - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

    def _cache_path(self, system: str | None, user: str) -> Path | None:
        if not self._cache_dir:
            return None
        blob = json.dumps([self.provider, self.model, self.cfg["temperature"], self.cfg["max_tokens"],
                           self.cfg.get("json_mode"), system, user])
        return self._cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    def _classify(self, exc: Exception) -> str | None:
        """'retry' (transient), 'fatal' (will not fix itself) or None (not an API error - re-raise as is)."""
        if self.provider == "gemini":
            from google.genai import errors

            if isinstance(exc, errors.APIError):
                code = int(getattr(exc, "code", 0) or 0)
                if code == 429 and (_DAILY_QUOTA.search(str(exc)) or (quota_reset_seconds(str(exc)) or 0) > 300):
                    return "quota"  # retrying for the next 17 hours will not help; try another model instead
                return "retry" if code == 429 or code >= 500 else "fatal"
            try:
                import httpx

                if isinstance(exc, httpx.TransportError):
                    return "retry"
            except ImportError:  # pragma: no cover
                pass
            return "retry" if isinstance(exc, (TimeoutError, ConnectionError)) else None
        import anthropic

        if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
            return "retry"
        if isinstance(exc, anthropic.APIStatusError):
            return "fatal" if exc.status_code < 500 else "retry"
        return None

    def _fatal_message(self, exc: Exception) -> str:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        msg = str(getattr(exc, "message", None) or exc)[:300]
        hint = ""
        if code == 404:
            hint = " -> model not found: run `python main.py models` and set llm.model in config.yaml."
        elif code in (400, 401, 403):
            hint = " -> check your API key (.env) and that the model name in config.yaml is valid."
        return f"API error {code}: {msg}{hint}"

    # ------------------------------------------------------------ one request per provider
    def _once(self, client, user: str, system: str | None, model: str) -> tuple[str, int, int]:
        if self.provider == "gemini":
            from google.genai import types

            kw = dict(temperature=self.cfg["temperature"], max_output_tokens=self.cfg["max_tokens"],
                      automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            if system:
                kw["system_instruction"] = system
            if self.cfg.get("json_mode", True):
                kw["response_mime_type"] = "application/json"
            # Thinking is configured per model FAMILY (the fallback model may be a different family):
            #   gemini-3.x -> thinking_level (low/medium/high; 3.8 Flash has no 'minimal')
            #   gemini-2.5 -> thinking_budget (0 switches thinking off)
            if model.startswith("gemini-3") and self.cfg.get("thinking_level"):
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=str(self.cfg["thinking_level"]).lower())
            elif model.startswith("gemini-2.5") and self.cfg.get("thinking_budget") is not None:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=int(self.cfg["thinking_budget"]))
            resp = client.models.generate_content(model=model, contents=user,
                                                  config=types.GenerateContentConfig(**kw))
            text = getattr(resp, "text", None)
            if not text:
                cands = getattr(resp, "candidates", None) or []
                reason = getattr(cands[0], "finish_reason", None) if cands else "no candidates (blocked?)"
                raise EmptyResponse(f"Gemini returned no text from '{model}' (finish_reason={reason}). Thinking tokens probably used up "
                                    "llm.max_tokens: set llm.thinking_level: low (gemini-3.x) or llm.thinking_budget: 0 (gemini-2.5).")
            u = getattr(resp, "usage_metadata", None)
            tin = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
            tout = ((getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)) if u else 0
            return text, tin, tout
        kwargs = dict(model=model, max_tokens=self.cfg["max_tokens"], temperature=self.cfg["temperature"],
                      messages=[{"role": "user", "content": user}])
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _models_to_try(self) -> list[str]:
        fallbacks = self.cfg.get("fallback_models") or []
        if isinstance(fallbacks, str):
            fallbacks = [fallbacks]
        return [self.model] + [m for m in fallbacks if m and m != self.model]

    def _call_api(self, user: str, system: str | None) -> str:
        """Try the main model with retries/backoff; if it stays overloaded (429/5xx) or is not found, try the fallbacks."""
        client = self._get_client()
        retries = int(self.cfg.get("max_retries", 3))
        delay = float(self.cfg.get("retry_base_delay", 2.0))
        models = self._models_to_try()
        last: Exception | None = None
        quota_hits: dict[str, float | None] = {}
        for index, model in enumerate(models):
            if self._exhausted and self._exhausted.get(model, 0) > time.monotonic():  # known to be out of quota this session
                quota_hits[model] = self._exhausted[model] - time.monotonic()
                continue
            for attempt in range(1, retries + 1):
                self._throttle()
                try:
                    text, tin, tout = self._once(client, user, system, model)
                except EmptyResponse as e:  # deterministic for this model: don't retry it, go to the fallback
                    last = e
                    log.warning("Model '%s' returned no text: %s", model, e)
                    break
                except LLMError:
                    raise
                except Exception as e:  # noqa: BLE001 - classified below, unknown errors are re-raised untouched
                    kind = self._classify(e)
                    if kind is None:
                        raise
                    if kind == "quota":
                        wait = quota_reset_seconds(str(e))
                        self._exhausted[model] = time.monotonic() + (wait or 3600)
                        quota_hits[model] = wait
                        last = e
                        log.warning("Daily quota exhausted for '%s' (resets in %s) - not retrying it", model, _fmt_wait(wait))
                        break
                    if kind == "fatal":
                        if getattr(e, "code", None) == 404 and index < len(models) - 1:
                            log.warning("Model '%s' not found - trying fallback '%s'", model, models[index + 1])
                            last = e
                            break
                        raise LLMError(self._fatal_message(e)) from e
                    last = e
                    log.warning("LLM call failed (model %s, attempt %d/%d): %s", model, attempt, retries, e)
                    if attempt < retries:
                        time.sleep(delay * 2 ** (attempt - 1))
                    continue
                self.usage["calls"] += 1
                self.usage["input_tokens"] += tin
                self.usage["output_tokens"] += tout
                self.last_model = model
                if index > 0:
                    self.usage["fallback_calls"] += 1
                    log.warning("Answered by fallback model '%s' (main model '%s' unavailable)", model, self.model)
                return text
            if index < len(models) - 1:
                log.warning("Giving up on '%s' after %d attempts - switching to '%s'", model, retries, models[index + 1])
        if quota_hits and len(quota_hits) == len(models):
            detail = ", ".join(f"{m} (resets in {_fmt_wait(w)})" for m, w in quota_hits.items())
            raise LLMError(f"Free-tier daily quota exhausted for: {detail}. Wait for the reset, add a model with remaining "
                           "quota to llm.fallback_models, enable billing on the API key, or use mode 'rules'.")
        raise LLMError(f"LLM call failed after {retries} attempts: {last} (models tried: {', '.join(models)})")

    # ------------------------------------------------------------ public API
    def complete(self, user: str, system: str | None = None, use_cache: bool = True) -> str:
        path = self._cache_path(system, user) if use_cache else None
        if path and path.exists():
            self.usage["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self._call_api(user, system)
        if path:
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text

    def complete_json(self, user: str, system: str | None = None) -> dict:
        text = self.complete(user, system)
        try:
            return parse_json_object(text)
        except ValueError as first_error:
            log.info("Malformed JSON (%s) - asking the model to repair it", first_error)
            repaired = self.complete(self.prompts.render("repair", bad_output=text[:6000]), system)
            try:
                return parse_json_object(repaired)
            except ValueError as e:
                raise LLMError(f"Model did not return valid JSON: {e}") from e

    def ping(self) -> str:
        """Tiny uncached request to verify key + model (used by `python main.py check`)."""
        return self.complete("Reply with the single word: ok", use_cache=False).strip().strip('"').strip()

    def list_models(self) -> list[str]:
        client = self._get_client()
        if self.provider == "gemini":
            return sorted(m.name.removeprefix("models/") for m in client.models.list()
                          if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]))
        return sorted(m.id for m in client.models.list())