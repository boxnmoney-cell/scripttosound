import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from studio.models import Analysis, Event, Review, Script, Segment
from studio.pipeline import translate, write_script
from studio.profiles import PROFILES
from studio.providers import APIError, Providers
from studio.storage import invalidate, project_for_upload, read_json, save_json


@pytest.fixture
def analysis():
    return Analysis(events=[Event(id="E001", start=0, end=2, description="Кирка разбивает камень",
                                 evidence="0–2: виден удар и разрушение блока", confidence="confirmed"),
                            Event(id="E002", start=3, end=4, description="Возможно виден алмаз",
                                  evidence="Размыто", confidence="uncertain")],
                    outcome="unknown", outcome_evidence="Нет результата", notes=[])


@pytest.fixture
def script():
    return Script(language="pt-BR", segments=[Segment(id="S001", start=0, end=2,
                  text="A picareta quebrou a pedra.", event_ids=["E001"])])


def test_no_uncertain_events_or_overlapping_speech(analysis, script):
    script.check(analysis, 5)
    script.segments[0].event_ids = ["E002"]
    with pytest.raises(ValueError, match="неподтверждённые"):
        script.check(analysis, 5)
    script.segments[0].event_ids = ["E001"]
    script.segments.append(Segment(id="S002", start=1, end=3, text="Outra frase", event_ids=["E001"]))
    with pytest.raises(ValueError, match="пересекаются"):
        script.check(analysis, 5)


def test_nonfinite_timing_rejected():
    with pytest.raises(ValidationError):
        Segment(id="S", start=float("nan"), end=2, text="x", event_ids=["E"])


def test_analysis_time_and_duplicates(analysis):
    analysis.events.append(analysis.events[0])
    with pytest.raises(ValueError, match="ID"):
        analysis.check_duration(5)


def test_stage_invalidation_preserves_analysis():
    state = {k: k for k in ["analysis", "script", "voice_script", "translation", "audio", "export"]}
    invalidate(state, "script")
    assert state == {"analysis": "analysis", "script": "script"}


def test_storage_deduplicates_uploads(tmp_path):
    a, first = project_for_upload(tmp_path, b"same media", ".mp4", "viper")
    b, second = project_for_upload(tmp_path, b"same media", ".mp4", "viper")
    assert a == b and first == second
    save_json(a / "project.json", {"language": "pt-BR", "text": "ação"})
    assert read_json(a / "project.json")["text"] == "ação"
    with pytest.raises(ValueError):
        project_for_upload(tmp_path, b"x", ".mp4", "../../escape")


def test_provider_cache_and_schema(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"issues": []}'}}],
                                        "usage": {"total_tokens": 22}})
    api = Providers("test-secret", transport=httpx.MockTransport(handler))
    for _ in range(2):
        assert api.structured("model-a", "rules", "facts", Review, tmp_path).issues == []
    api.structured("model-b", "rules", "facts", Review, tmp_path)
    assert len(calls) == 2
    assert all("test-secret" not in f.read_text() for f in tmp_path.glob("*.json"))
    api.close()


def test_api_errors_do_not_echo_secrets():
    api = Providers("secret", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="secret")))
    with pytest.raises(APIError) as error:
        api.structured("m", "s", "c", Review)
    assert "secret" not in str(error.value)
    api.close()


def test_paid_speech_post_is_not_retried(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("lost response", request=request)
    api = Providers(eleven_key="test", transport=httpx.MockTransport(handler))
    with pytest.raises(APIError, match="списать кредиты"):
        api.speech("hello", "Voice123", "eleven_v4", tmp_path)
    assert len(calls) == 1
    api.close()


class FakeAPI:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []
    def structured(self, *args):
        self.calls.append(args)
        return next(self.results)


def test_translation_cannot_change_facts_or_timestamps(analysis, script, tmp_path):
    translated = script.model_copy(deep=True)
    translated.language = "ru-RU"
    translated.segments[0].start = 0.5
    with pytest.raises(ValueError, match="время"):
        translate(FakeAPI([translated]), script, "ru-RU", analysis, PROFILES[0], "model", tmp_path)


def test_language_change_preserves_event_binding(analysis, script, tmp_path):
    translated = script.model_copy(deep=True)
    translated.language = "ru-RU"
    translated.segments[0].text = "Кирка разбила камень."
    api = FakeAPI([translated, Review(issues=[])])
    result = translate(api, script, "ru-RU", analysis, PROFILES[0], "model", tmp_path)
    assert result.segments[0].event_ids == ["E001"]
    assert result.segments[0].start == 0
    assert script.language == "pt-BR"


def test_failed_factual_review_is_not_published(analysis, script, tmp_path):
    api = FakeAPI([script, Review(issues=["unsupported cashout"]), script, Review(issues=["unsupported cashout"])])
    with pytest.raises(ValueError, match="unsupported cashout"):
        write_script(api, analysis, 5, PROFILES[0], "pt-BR", "natural", "Наблюдатель", "m", tmp_path)
    assert len(api.calls) == 4
