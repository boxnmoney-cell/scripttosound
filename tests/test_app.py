from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from studio import media
from studio.models import Analysis, Delivery, DirectionItem, DirectionPlan, Event, Review, Script, Segment
from studio.profiles import PROFILES
from studio.performance import voice_fingerprint
from studio.storage import digest, save_json


def test_app_opens_without_api_keys_and_languages_are_independent(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIPT_TO_SOUND_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "🎙️ Script to Sound"
    script = next(s for s in app.selectbox if s.label == "Язык сценария")
    voice = next(s for s in app.selectbox if s.label == "Язык озвучки")
    script.select("English (US)")
    voice.select("Português (Brasil)")
    app.run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label == "Язык сценария").value == "English (US)"
    assert next(s for s in app.selectbox if s.label == "Язык озвучки").value == "Português (Brasil)"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_existing_project_can_export_and_stale_voice_cannot(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIPT_TO_SOUND_DATA_DIR", str(tmp_path))
    directory = tmp_path / "projects" / "fixture"
    directory.mkdir(parents=True)
    video = directory / "source.mp4"
    audio = directory / "voice.wav"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x240:r=25:d=2",
               "-c:v", "libx264", str(video)])
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=2", str(audio)])
    analysis = Analysis(events=[Event(id="E001", start=0, end=2, description="Тестовое событие",
                                      evidence="Тестовые данные", confidence="confirmed")],
                        outcome="unknown", outcome_evidence="Тест", notes=[])
    script = Script(language="pt-BR", segments=[Segment(id="S001", start=0, end=2, text="Teste.", event_ids=["E001"])])
    fingerprint = voice_fingerprint(script, "Voice123", "test_model", False)
    save_json(tmp_path / "preferences.json", {"voices": {"pt-BR": {"id": "Voice123", "model": "test_model"}}})
    save_json(directory / "project.json", {"name": "UI fixture", "video": video.name,
              "profile_id": PROFILES[0].id, "info": media.video_info(video).model_dump(),
              "analysis": analysis.model_dump(), "analysis_profile": digest(PROFILES[0].model_dump()),
              "script": script.model_dump(), "audio": {"path": audio.name,
              "fingerprint": fingerprint, "timing": [{"id": "S001", "start": 0, "end": 2}]}})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.session_state["active_project"] = str(directory)
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label == "Собрать MP4").click()
    app.run()
    assert not app.exception
    assert list(directory.glob("export-*.mp4"))
    next(t for t in app.text_input if t.label == "Voice ID").set_value("DifferentVoice")
    app.run()
    assert not app.exception
    assert not any(b.label == "Собрать MP4" for b in app.button)
    assert any("актуальная озвучка" in w.value for w in app.warning)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_ui_directs_previews_and_reuses_audio_without_real_api(monkeypatch, tmp_path):
    from studio.providers import Providers
    from studio.storage import read_json

    monkeypatch.setenv("SCRIPT_TO_SOUND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-only")
    requests = []

    def structured(self, model, system, content, schema, cache=None):
        if schema is DirectionPlan:
            return DirectionPlan(segments=[DirectionItem(id="S001", delivery=Delivery(
                emotion="amused", laugh=True, reason="Подтверждённый отскок"))])
        assert schema is Review
        return Review(issues=[])

    def speech(self, text, voice, model, cache):
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (digest({"text": text, "voice": voice, "model": model}) + ".wav")
        if not path.exists():
            requests.append(text)
            media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=0.5", str(path)])
        return path

    monkeypatch.setattr(Providers, "structured", structured)
    monkeypatch.setattr(Providers, "validate_speech", lambda *args: None)
    monkeypatch.setattr(Providers, "speech", speech)
    directory = tmp_path / "projects" / "directed"
    directory.mkdir(parents=True)
    video = directory / "source.mp4"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x240:r=25:d=3",
               "-c:v", "libx264", str(video)])
    analysis = Analysis(events=[Event(id="E001", start=0, end=3, description="Кирка отскочила",
                                      evidence="Тестовые данные", confidence="confirmed")],
                        outcome="unknown", outcome_evidence="Тест", notes=[])
    script = Script(language="pt-BR", segments=[Segment(id="S001", start=0, end=3,
                    text="Olha esse salto!", event_ids=["E001"])])
    save_json(tmp_path / "preferences.json", {"voices": {"pt-BR": {"id": "Voice123", "model": "eleven_v3"}}})
    save_json(directory / "project.json", {"name": "Direction fixture", "video": video.name,
              "profile_id": PROFILES[0].id, "info": media.video_info(video).model_dump(),
              "analysis": analysis.model_dump(), "analysis_profile": digest(PROFILES[0].model_dump()),
              "script": script.model_dump()})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.session_state["active_project"] = str(directory)
    app.run()
    next(b for b in app.button if b.label == "Расставить эмоции, сохранив текст").click()
    app.run()
    assert not app.exception
    state = read_json(directory / "project.json")
    assert state["script"]["segments"][0]["text"] == "Olha esse salto!"
    assert state["script"]["segments"][0]["delivery"]["laugh"]
    next(b for b in app.button if b.label == "Прослушать одну реплику").click()
    app.run()
    assert not app.exception
    assert len(requests) == 1 and "[laughs]" in requests[0]
    next(b for b in app.button if b.label == "Создать озвучку").click()
    app.run()
    assert not app.exception
    assert len(requests) == 1
    assert "audio" in read_json(directory / "project.json")
    next(t for t in app.toggle if t.label == "Передавать эмоции и смех в ElevenLabs").set_value(False)
    app.run()
    assert not app.exception
    assert not any(b.label == "Собрать MP4" for b in app.button)
