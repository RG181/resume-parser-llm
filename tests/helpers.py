import json
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


class FakeLLM:
    """Stands in for LLMClient: records prompts, returns canned JSON. No network needed."""

    def __init__(self, payload: dict | None = None, fail: Exception | None = None):
        self.payload, self.fail, self.prompts_seen = payload or {}, fail, []
        self.usage, self.model = {}, "fake-model"

    def complete_json(self, user, system=None):
        self.prompts_seen.append(user)
        if self.fail:
            raise self.fail
        return dict(self.payload)
