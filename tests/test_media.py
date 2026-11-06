import shutil

import pytest

from studio import media
from studio.models import Delivery, Script, Segment


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def test_real_media_pipeline_preserves_last_frame_and_silence(tmp_path):
    video = tmp_path / "source.mp4"
    speech = tmp_path / "speech.wav"
    voice = tmp_path / "voice.wav"
    output = tmp_path / "output.mp4"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x240:rate=25:duration=4",
               "-f", "lavfi", "-i", "sine=frequency=220:duration=4", "-c:v", "libx264", "-c:a", "aac", str(video)])
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(speech)])
    info = media.video_info(video)
    sampled = media.frames(video, tmp_path / "frames", info.duration, fps=1)
    assert len(sampled) >= 4 and sampled[-1]["time"] > 3.8
    script = Script(language="pt-BR", segments=[Segment(id="S001", start=1, end=2.5, text="Teste", event_ids=["E001"])])
    timing = media.assemble_voice([speech], script, voice, info.duration)
    assert abs(media.duration(voice) - 4) < 0.1
    assert timing[0]["start"] == 1 and timing[0]["end"] == pytest.approx(2, abs=.05)
    # A second of silence precedes the actual synthetic voice (not just a duration check).
    raw = media.run(["ffmpeg", "-v", "error", "-i", str(voice), "-t", "0.8", "-f", "s16le", "-"])
    assert raw and not any(raw)
    media.export_video(video, voice, output, info, original_volume=0.1)
    assert abs(media.duration(output) - 4) < 0.15
    assert "00:00:01,000 --> 00:00:02,000" in media.srt(script, timing)


def test_overlong_voice_is_not_cut_off(tmp_path):
    speech = tmp_path / "long.wav"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=2", str(speech)])
    script = Script(language="pt-BR", segments=[Segment(id="S001", start=0, end=1, text="Long", event_ids=["E001"])])
    with pytest.raises(ValueError, match="Сократите"):
        media.assemble_voice([speech], script, tmp_path / "output.wav", 3)


def test_multiple_fragments_keep_their_positions(tmp_path):
    import array
    speech = tmp_path / "tone.wav"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", str(speech)])
    script = Script(language="ru-RU", segments=[
        Segment(id="S001", start=0, end=1, text="Один", event_ids=["E001"]),
        Segment(id="S002", start=2, end=3, text="Два", event_ids=["E002"])])
    output = tmp_path / "mixed.wav"
    media.assemble_voice([speech, speech], script, output, 4)
    samples = array.array("h", media.run(["ffmpeg", "-v", "error", "-i", str(output), "-ac", "1", "-f", "s16le", "-"]))
    rate = 44100
    assert any(samples[:rate // 2])
    assert not any(samples[rate:rate * 2])
    assert any(samples[rate * 2:rate * 2 + rate // 2])
    assert not any(samples[rate * 3:])


def test_directed_pause_is_real_silence_and_leaves_video_length_intact(tmp_path):
    import array
    speech = tmp_path / "tone.wav"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=0.5", str(speech)])
    segment = Segment(id="S001", start=0, end=2, text="Olha!", event_ids=["E001"],
                      delivery=Delivery(pause_before_ms=300, reason="Пауза перед видимым поворотом"))
    script = Script(language="pt-BR", segments=[segment])
    output = tmp_path / "pause.wav"
    timing = media.assemble_voice([speech], script, output, 2)
    samples = array.array("h", media.run(["ffmpeg", "-v", "error", "-i", str(output), "-ac", "1", "-f", "s16le", "-"]))
    assert not any(samples[:int(44100 * .29)])
    assert any(samples[int(44100 * .31):int(44100 * .8)])
    assert media.duration(output) == pytest.approx(2, abs=.01)
    assert timing[0]["start"] == .3
    assert "00:00:00,300 --> 00:00:00,800" in media.srt(script, timing)
