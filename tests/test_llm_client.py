"""LLM client behaviour with mocked SDKs - no network, no API key. Covers Gemini (default) and Anthropic."""
from types import SimpleNamespace

import pytest

from resume_parser import llm_client
from resume_parser.config import load_config, load_prompts
from resume_parser.llm_client import LLMClient, LLMError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """load_config() calls load_dotenv(), which would re-inject the real keys from .env after a test deleted them."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)


def build(provider, script, tmp_path, cache=False, **llm_overrides):
    cfg = load_config()
    settings = dict(provider=provider, cache_enabled=cache, cache_dir=str(tmp_path), requests_per_minute=None,
                    retry_base_delay=0, fallback_models=[])
    settings.update(llm_overrides)  # tests may override the defaults above
    cfg["llm"].update(settings)
    client = LLMClient(cfg, load_prompts(cfg))
    return client


# =============================================================== Gemini
google_errors = pytest.importorskip("google.genai.errors")


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text, usage = item if isinstance(item, tuple) else (item, (10, 5, 0))
        return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")],
                               usage_metadata=SimpleNamespace(prompt_token_count=usage[0], candidates_token_count=usage[1],
                                                              thoughts_token_count=usage[2]))


def gemini(script, tmp_path, **kw):
    c = build("gemini", script, tmp_path, **kw)
    c._client = SimpleNamespace(models=FakeModels(script))
    return c


def api_error(code):
    cls = google_errors.ClientError if code < 500 else google_errors.ServerError
    return cls(code, {"error": {"message": "boom", "status": "X"}})


def test_gemini_success_sends_config_and_counts_tokens(tmp_path):
    c = gemini([('{"ok": true}', (100, 20, 30))], tmp_path)
    assert c.complete_json("hello", system="be precise") == {"ok": True}
    call = c._client.models.calls[0]
    assert call["model"] == c.model and call["contents"] == "hello"
    assert call["config"].system_instruction == "be precise"
    assert call["config"].response_mime_type == "application/json" and call["config"].temperature == 0
    assert c.usage["input_tokens"] == 100 and c.usage["output_tokens"] == 50  # answer + thinking tokens


def _thinking(c):
    return c._client.models.calls[0]["config"].thinking_config


def test_gemini_3_uses_thinking_level_and_ignores_budget(tmp_path):
    c = gemini(['{"a": 1}'], tmp_path, model="gemini-3.8-flash", thinking_level="low", thinking_budget=0)
    c.complete("x")
    assert str(_thinking(c).thinking_level).endswith("LOW") and _thinking(c).thinking_budget is None


def test_gemini_25_uses_thinking_budget_and_ignores_level(tmp_path):
    c = gemini(['{"a": 1}'], tmp_path, model="gemini-2.5-flash", thinking_level="low", thinking_budget=0)
    c.complete("x")
    assert _thinking(c).thinking_budget == 0 and _thinking(c).thinking_level is None


def test_gemini_thinking_config_is_optional(tmp_path):
    c = gemini(['{"a": 1}'], tmp_path, model="gemini-3.8-flash", thinking_level=None, thinking_budget=None)
    c.complete("x")
    assert _thinking(c) is None


def test_empty_response_moves_on_to_fallback_model(tmp_path):
    empty = SimpleNamespace(text=None, candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")], usage_metadata=None)
    c = gemini([], tmp_path, model="gemini-3.8-flash", fallback_models=["gemini-2.5-flash"])
    answers = [empty, SimpleNamespace(text='{"ok": 1}', candidates=[], usage_metadata=None)]
    c._client.models.generate_content = lambda **kw: (c._client.models.calls.append(kw), answers.pop(0))[1]
    assert c.complete_json("x") == {"ok": 1}
    assert [call["model"] for call in c._client.models.calls] == ["gemini-3.8-flash", "gemini-2.5-flash"]



def test_gemini_retries_429_and_5xx_then_succeeds(tmp_path, no_sleep):
    c = gemini([api_error(429), api_error(503), '{"ok": 1}'], tmp_path, retry_base_delay=2.0)
    assert c.complete_json("hi") == {"ok": 1}
    assert no_sleep == [2.0, 4.0]  # exponential backoff


def test_gemini_gives_up_after_max_retries(tmp_path):
    c = gemini([api_error(503)] * 3, tmp_path, max_retries=3)
    with pytest.raises(LLMError, match="after 3 attempts"):
        c.complete("hi")


@pytest.mark.parametrize("code,hint", [(404, "python main.py models"), (400, "API key"), (403, "API key")])
def test_gemini_client_errors_are_not_retried_and_give_hints(tmp_path, code, hint):
    c = gemini([api_error(code), '{"x": 1}'], tmp_path)
    with pytest.raises(LLMError, match=hint):
        c.complete("hi")
    assert len(c._client.models.calls) == 1


def test_gemini_empty_response_has_clear_error(tmp_path):
    c = gemini([""], tmp_path)
    with pytest.raises(LLMError, match="no text"):
        c.complete("hi")


def test_gemini_malformed_json_is_repaired_once(tmp_path):
    c = gemini(['{"a": ', '{"a": 1}'], tmp_path)
    assert c.complete_json("hi") == {"a": 1}
    assert "<broken>" in c._client.models.calls[1]["contents"]


def test_unrepairable_json_raises(tmp_path):
    c = gemini(["garbage", "still garbage"], tmp_path)
    with pytest.raises(LLMError, match="valid JSON"):
        c.complete_json("hi")


def test_cache_avoids_second_call_and_ping_bypasses_it(tmp_path):
    c = gemini(['{"a": 1}', "ok"], tmp_path, cache=True)
    assert c.complete_json("same") == c.complete_json("same") == {"a": 1}
    assert c.usage["calls"] == 1 and c.usage["cache_hits"] == 1
    assert c.ping() == "ok" and c.usage["calls"] == 2


def test_throttle_spaces_requests(tmp_path, monkeypatch, no_sleep):
    c = gemini(['{"a":1}', '{"a":2}'], tmp_path, requests_per_minute=30)
    clock = iter([100.0, 100.0, 100.5, 100.5, 102.0])  # monotonic readings
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: next(clock))
    c.complete("one", use_cache=False)
    c.complete("two", use_cache=False)
    assert no_sleep and abs(no_sleep[0] - 1.5) < 1e-6  # 60/30 = 2s interval, 0.5s already elapsed


def test_missing_key_and_unknown_provider(tmp_path, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        build("gemini", [], tmp_path).complete("hi")
    with pytest.raises(LLMError, match="Unknown llm.provider"):
        build("openai", [], tmp_path)


def test_google_api_key_is_accepted_as_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "abc")
    assert build("gemini", [], tmp_path)._api_key() == "abc"


# =============================================================== Anthropic (optional provider)
anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=item)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def claude(script, tmp_path):
    c = build("anthropic", script, tmp_path)
    c._client = SimpleNamespace(messages=FakeMessages(script))
    return c


def test_anthropic_provider_retries_then_succeeds(tmp_path):
    req = httpx.Request("POST", "http://x")
    c = claude([anthropic.APIConnectionError(request=req), '{"ok": true}'], tmp_path)
    assert c.complete_json("hi", system="sys") == {"ok": True}
    assert c._client.messages.calls[0]["system"] == "sys" and c.usage["input_tokens"] == 10


def test_anthropic_client_errors_not_retried(tmp_path):
    resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    c = claude([anthropic.APIStatusError("nope", response=resp, body=None), '{"a":1}'], tmp_path)
    with pytest.raises(LLMError, match="401"):
        c.complete("hi")
    assert len(c._client.messages.calls) == 1


# =============================================================== fallback models
def test_overloaded_main_model_falls_back_and_reports_it(tmp_path):
    c = gemini([api_error(503), api_error(503), '{"from": "fallback"}'], tmp_path, max_retries=2,
               fallback_models=["backup-model"])
    assert c.complete_json("hi") == {"from": "fallback"}
    assert [call["model"] for call in c._client.models.calls] == [c.model, c.model, "backup-model"]
    assert c.last_model == "backup-model" and c.usage["fallback_calls"] == 1


def test_main_model_not_found_goes_straight_to_fallback(tmp_path):
    c = gemini([api_error(404), '{"ok": 1}'], tmp_path, fallback_models=["backup-model"])
    assert c.complete_json("hi") == {"ok": 1}
    assert len(c._client.models.calls) == 2  # no retries wasted on a model that does not exist


def test_bad_key_does_not_trigger_fallback(tmp_path):
    c = gemini([api_error(403), '{"ok": 1}'], tmp_path, fallback_models=["backup-model"])
    with pytest.raises(LLMError, match="API key"):
        c.complete("hi")
    assert len(c._client.models.calls) == 1


def test_all_models_down_raises_clear_error(tmp_path):
    c = gemini([api_error(503)] * 4, tmp_path, max_retries=2, fallback_models=["backup-model"])
    with pytest.raises(LLMError, match="models tried: .*backup-model"):
        c.complete("hi")


def test_main_model_healthy_never_touches_fallback(tmp_path):
    c = gemini(['{"a": 1}'], tmp_path, fallback_models=["backup-model"])
    c.complete("hi")
    assert c.last_model == c.model and c.usage["fallback_calls"] == 0


# =============================================================== REAL Gemini SDK <-> local mock HTTP server
def _serve(script):
    """Tiny HTTP server that imitates the Gemini REST API. script: {model: [(status, json_body), ...]}"""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = _json.loads(self.rfile.read(length) or b"{}")
            model = self.path.split("/models/")[1].split(":")[0]
            status, payload = script[model].pop(0)
            seen.append((model, body))
            raw = _json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def _ok(text, pin=12, pout=5):
    return 200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": pin, "candidatesTokenCount": pout, "totalTokenCount": pin + pout}}


def _overloaded():
    return 503, {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}


def real_sdk_client(server, tmp_path, **overrides):
    from google import genai
    from google.genai import types

    c = build("gemini", [], tmp_path, **overrides)
    c._client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
        base_url=f"http://127.0.0.1:{server.server_port}", timeout=5000))
    return c


def test_real_sdk_request_and_response_parsing(tmp_path):
    server, seen = _serve({"main-model": [_ok('{"name": "Riya"}', 120, 30)]})
    c = real_sdk_client(server, tmp_path)
    c.cfg["model"] = "main-model"
    assert c.complete_json("extract this", system="be precise") == {"name": "Riya"}
    model, body = seen[0]
    assert body["contents"][0]["parts"][0]["text"] == "extract this"
    assert body["systemInstruction"]["parts"][0]["text"] == "be precise"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["temperature"] == 0
    assert c.usage["input_tokens"] == 120 and c.usage["output_tokens"] == 30
    server.shutdown()


def test_real_sdk_503_then_fallback_end_to_end(tmp_path, caplog):
    script = {"main-model": [_overloaded(), _overloaded()], "backup-model": [_ok('{"ok": true}')]}
    server, seen = _serve(script)
    c = real_sdk_client(server, tmp_path, max_retries=2, fallback_models=["backup-model"])
    c.cfg["model"] = "main-model"
    assert c.complete_json("hi") == {"ok": True}
    assert [m for m, _ in seen] == ["main-model", "main-model", "backup-model"]
    assert c.last_model == "backup-model"
    server.shutdown()


def test_real_sdk_does_not_emit_function_calling_warning(tmp_path, caplog):
    import logging
    server, _ = _serve({"main-model": [_ok('{"a": 1}')]})
    c = real_sdk_client(server, tmp_path)
    c.cfg["model"] = "main-model"
    with caplog.at_level(logging.WARNING):
        c.complete("hi")
    assert "automatic function calling" not in caplog.text.lower()
    server.shutdown()


# ---- daily free-tier quota (the real-world failure: 429 'GenerateRequestsPerDayPerProjectPerModel-FreeTier')
DAILY_429 = ("You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, limit: 20, "
             "model: gemini-3.8-flash Please retry in 17h40m45.44s. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier "
             "retryDelay': '63645s'")


def daily_quota_error():
    return google_errors.ClientError(429, {"error": {"message": DAILY_429, "status": "RESOURCE_EXHAUSTED"}})


def test_quota_reset_parsing():
    from resume_parser.llm_client import quota_reset_seconds
    assert round(quota_reset_seconds(DAILY_429)) == 17 * 3600 + 40 * 60 + 45
    assert quota_reset_seconds("retryDelay': '63645s'") == 63645
    assert quota_reset_seconds("nothing here") is None


def test_daily_quota_is_not_retried_and_falls_back_to_next_model(tmp_path):
    c = gemini([daily_quota_error(), '{"ok": 1}'], tmp_path, model="gemini-3.8-flash", fallback_models=["gemini-3.5-flash"])
    assert c.complete_json("x") == {"ok": 1}
    assert [call["model"] for call in c._client.models.calls] == ["gemini-3.8-flash", "gemini-3.5-flash"]  # 1 try, not 4


def test_exhausted_model_is_skipped_on_later_calls(tmp_path):
    c = gemini([daily_quota_error(), '{"a": 1}', '{"b": 2}'], tmp_path, model="gemini-3.8-flash", fallback_models=["gemini-3.5-flash"])
    c.complete("one", use_cache=False)
    c.complete("two", use_cache=False)
    assert [call["model"] for call in c._client.models.calls] == ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash"]


def test_all_models_out_of_quota_gives_a_clear_message(tmp_path):
    c = gemini([daily_quota_error()], tmp_path, model="gemini-3.8-flash", fallback_models=[])
    with pytest.raises(LLMError, match="daily quota exhausted.*17h 40m"):
        c.complete("x")
    assert len(c._client.models.calls) == 1


def test_per_minute_429_is_still_retried(tmp_path):
    transient = google_errors.ClientError(429, {"error": {"message": "slow down, retry in 5s", "status": "RESOURCE_EXHAUSTED"}})
    c = gemini([transient, '{"ok": 1}'], tmp_path, model="gemini-3.8-flash", fallback_models=[])
    assert c.complete_json("x") == {"ok": 1} and len(c._client.models.calls) == 2