import math
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
    sampled = media.frames(video, tmp_path / "frames", info.duration)
    assert len(sampled) == 20 and sampled[-1]["time"] == 3.8
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


@pytest.mark.parametrize("seconds,source_fps", [(.04, 25), (.24, 25), (1, 25), (1.04, 25), (2.12, 25), (2, 2)])
def test_five_fps_including_short_tail_and_low_source_fps(tmp_path, seconds, source_fps):
    from PIL import Image
    video = tmp_path / "source.mp4"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=64x96:rate={source_fps}:duration={seconds}",
               "-c:v", "libx264", str(video)])
    info = media.video_info(video)
    sampled = media.frames(video, tmp_path / "frames", info.duration)
    assert [f["time"] for f in sampled] == media.sample_times(info.duration)
    assert len(sampled) == math.ceil(info.duration * 5 - 1e-9)
    assert all(f["time"] < info.duration for f in sampled)
    with Image.open(sampled[0]["path"]) as image:
        assert image.size == (64, 96)  # Small originals are not needlessly upscaled.


def test_no_240_frame_cap_and_cache_requires_complete_sequence(tmp_path, monkeypatch):
    video = tmp_path / "long.mp4"
    media.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=32x48:r=5:d=51",
               "-c:v", "libx264", str(video)])
    info = media.video_info(video)
    actual_run = media.run
    calls = []
    def capture(*args, **kwargs):
        calls.append(args[0])
        return actual_run(*args, **kwargs)
    monkeypatch.setattr(media, "run", capture)
    sampled = media.frames(video, tmp_path / "frames", info.duration)
    assert len(sampled) == 255
    assert sampled[-1]["time"] == 50.8
    assert len(calls) == 1  # One decode pass, not one FFmpeg process per image.
    again = media.frames(video, tmp_path / "frames", info.duration)
    assert sampled == again and len(calls) == 1
    sampled[100]["path"].unlink()
    repaired = media.frames(video, tmp_path / "frames", info.duration)
    assert repaired[100]["path"].is_file() and len(calls) == 2


def test_incomplete_extraction_never_creates_a_valid_manifest(tmp_path, monkeypatch):
    from pathlib import Path
    video = tmp_path / "source.mp4"
    video.write_bytes(b"mock input")
    def incomplete(args, **kwargs):
        Path(args[-1].replace('%06d', '000001')).write_bytes(b"one frame")
        return b""
    monkeypatch.setattr(media, "run", incomplete)
    with pytest.raises(ValueError, match="1 кадров вместо 10"):
        media.frames(video, tmp_path / "frames", 2)
    assert not list((tmp_path / "frames").rglob("manifest.json"))


def test_frame_images_match_their_sampling_times(tmp_path):
    from PIL import Image, ImageStat
    for i in range(25):
        Image.new("RGB", (32, 32), (i * 8,) * 3).save(tmp_path / f"in_{i:02d}.png")
    video = tmp_path / "clock.mkv"
    media.run(["ffmpeg", "-y", "-framerate", "25", "-i", str(tmp_path / "in_%02d.png"),
               "-c:v", "ffv1", str(video)])
    sampled = media.frames(video, tmp_path / "frames", media.video_info(video).duration)
    assert [f["time"] for f in sampled] == [0, .2, .4, .6, .8]
    for frame, expected in zip(sampled, [0, 40, 80, 120, 160]):
        with Image.open(frame["path"]) as image:
            assert ImageStat.Stat(image.convert("L")).mean[0] == pytest.approx(expected, abs=3)
