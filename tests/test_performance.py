import json

import httpx
import pytest
from pydantic import ValidationError

from studio.media import srt
from studio.models import Analysis, Delivery, DirectionItem, DirectionPlan, Event, PerformanceSettings, Review, Script, Segment
from studio.performance import automatic_audio_tags, tts_text, voice_fingerprint
from studio.pipeline import direct_script, translate
from studio.profiles import PROFILES
from studio.providers import Providers
from studio.storage import digest


def scenario():
    analysis = Analysis(events=[Event(id="E001", start=0, end=5, description="Кирка задела блок и отскочила",
                                      evidence="На кадрах видно отскок", confidence="confirmed")],
                        outcome="unknown", outcome_evidence="Финала нет", notes=[])
    script = Script(language="pt-BR", segments=[Segment(id="S001", start=0, end=5,
                    text="Olha esse salto!", event_ids=["E001"])])
    return analysis, script


class FakeAPI:
    def __init__(self, responses):
        self.responses = iter(responses)
    def structured(self, *args):
        return next(self.responses)


def test_existing_project_keeps_neutral_audio_fingerprint():
    raw = {"language": "pt-BR", "perspective": "observer", "segments": [
        {"id": "S001", "start": 0.0, "end": 5.0, "text": "Olá!", "event_ids": ["E001"]}]}
    old_key = digest({"script": raw, "voice": "Voice123", "model": "eleven_v3"})
    loaded = Script.model_validate(raw)
    assert voice_fingerprint(loaded, "Voice123", "eleven_v3", True) == old_key


def test_acting_plan_cannot_rewrite_words_or_timing(tmp_path):
    analysis, script = scenario()
    original = script.model_dump()
    delivery = Delivery(emotion="amused", laugh=True, reason="Короткий отскок кирки", pause_before_ms=200)
    api = FakeAPI([DirectionPlan(segments=[DirectionItem(id="S001", delivery=delivery)]), Review(issues=[])])
    result = direct_script(api, script, analysis, 5, PROFILES[0], PerformanceSettings(), "model", tmp_path)
    assert result.segments[0].text == script.segments[0].text
    assert (result.segments[0].start, result.segments[0].end, result.segments[0].event_ids) == (0, 5, ["E001"])
    assert result.segments[0].delivery.laugh
    assert script.model_dump() == original


def test_acting_plan_rejects_missing_or_new_segment(tmp_path):
    analysis, script = scenario()
    api = FakeAPI([DirectionPlan(segments=[DirectionItem(id="new", delivery=Delivery())])])
    with pytest.raises(ValueError, match="состав"):
        direct_script(api, script, analysis, 5, PROFILES[0], PerformanceSettings(), "model", tmp_path)


def test_words_subtitles_and_transmission_are_separate():
    _, script = scenario()
    segment = script.segments[0]
    segment.delivery = Delivery(emotion="amused", laugh=True, reason="Видимый отскок")
    assert tts_text(segment, True) == "[playfully] Olha esse salto! [laughs]"
    assert tts_text(segment, False) == "Olha esse salto!"
    assert "[" not in script.plain_text()
    assert "[" not in srt(script)
    segment.delivery.laugh_position = "before"
    assert tts_text(segment, True) == "[playfully] [laughs] Olha esse salto!"
    assert not automatic_audio_tags("eleven_multilingual_v2")
    assert not automatic_audio_tags("eleven_v4")  # Unknown capability is never inferred.
    assert automatic_audio_tags("eleven_v3")


def test_spoken_text_rejects_embedded_stage_directions():
    with pytest.raises(ValidationError, match="произносимый"):
        Segment(id="S", start=0, end=5, text="[laughs] Hello", event_ids=["E001"])


def test_laughter_requires_justification_time_and_permission():
    with pytest.raises(ValidationError, match="объясняет"):
        Delivery(laugh=True)
    with pytest.raises(ValidationError, match="Разделите"):
        Delivery(emotion="tense", laugh=True, reason="Danger")
    with pytest.raises(ValidationError, match="Пауза"):
        Segment(id="S", start=0, end=.8, text="Hi", event_ids=["E001"],
                delivery=Delivery(pause_before_ms=800, reason="Пауза"))
    analysis, script = scenario()
    script.segments[0].delivery = Delivery(laugh=True, reason="Отскок")
    script.performance.allow_laughter = False
    with pytest.raises(ValueError, match="отключён"):
        script.check(analysis, 5)
    script.performance.allow_laughter = True
    script.segments.append(Segment(id="S002", start=20, end=25, text="De novo!", event_ids=["E001"],
                                   delivery=Delivery(laugh=True, reason="Отскок")))
    with pytest.raises(ValueError, match="частый смех"):
        script.check(analysis, 30)


def test_translation_preserves_delivery(tmp_path):
    analysis, script = scenario()
    script.segments[0].delivery = Delivery(emotion="surprised", reason="Отскок")
    changed = script.model_copy(deep=True)
    changed.language = "ru-RU"
    changed.segments[0].delivery = Delivery()
    with pytest.raises(ValueError, match="эмоции"):
        translate(FakeAPI([changed]), script, "ru-RU", analysis, PROFILES[0], "m", tmp_path)
    changed.segments[0].delivery = script.segments[0].delivery.model_copy()
    changed.segments[0].text = "Вот это прыжок!"
    result = translate(FakeAPI([changed, Review(issues=[])]), script, "ru-RU", analysis, PROFILES[0], "m", tmp_path)
    assert result.segments[0].delivery == script.segments[0].delivery


def test_emotion_toggle_and_pause_invalidate_audio_but_reason_does_not():
    _, script = scenario()
    segment = script.segments[0]
    before = voice_fingerprint(script, "V", "eleven_v3", True)
    segment.delivery = Delivery(emotion="surprised", reason="Отскок")
    directed = voice_fingerprint(script, "V", "eleven_v3", True)
    assert before != directed
    assert voice_fingerprint(script, "V", "eleven_v3", False) == before
    segment.delivery.reason = "Уточнённое объяснение того же события"
    assert voice_fingerprint(script, "V", "eleven_v3", True) == directed
    segment.delivery.pause_before_ms = 300
    assert voice_fingerprint(script, "V", "eleven_v3", False) != before


def test_preview_uses_the_same_speech_cache_as_full_generation(tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b"test audio", headers={"content-type": "audio/mpeg"})
    api = Providers(eleven_key="test", transport=httpx.MockTransport(handler))
    _, script = scenario()
    segment = script.segments[0]
    segment.delivery = Delivery(emotion="amused", laugh=True, reason="Отскок")
    text = tts_text(segment, True)
    preview = api.speech(text, "Voice123", "eleven_v3", tmp_path)
    full = api.speech(text, "Voice123", "eleven_v3", tmp_path)
    assert preview == full and len(calls) == 1
    assert calls[0]["text"].endswith("[laughs]")
    api.speech(tts_text(segment, False), "Voice123", "eleven_v3", tmp_path)
    assert len(calls) == 2
    api.close()
