import base64
import json
import math
import subprocess
import tempfile
from pathlib import Path

from .config import FRAME_EXTRACTION_VERSION, FRAME_RATE
from .models import Script, VideoInfo
from .performance import speech_window
from .storage import read_json, save_json


def run(args, timeout=180):
    if Path(args[0]).name in {"ffmpeg", "ffmpeg.exe"} and "-nostdin" not in args:
        args = [args[0], "-nostdin", *args[1:]]
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("Установите FFmpeg и добавьте ffmpeg/ffprobe в PATH.") from exc
    if result.returncode:
        message = result.stderr.decode(errors="replace")[-1200:]
        raise RuntimeError(f"Ошибка обработки медиа: {message}")
    return result.stdout


def probe(path: Path):
    raw = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    return json.loads(raw)


def video_info(path: Path):
    info = probe(path)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not video:
        raise ValueError("В файле нет видеодорожки")
    return VideoInfo(duration=float(video.get("duration", info["format"].get("duration", 0))),
                     width=video["width"], height=video["height"],
                     has_audio=any(s["codec_type"] == "audio" for s in info["streams"]))


def duration(path: Path):
    value = float(probe(path)["format"]["duration"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Некорректная длительность медиа")
    return value


def sample_times(video_duration: float):
    """One sample every 200 ms, including a partial last second; never thin the grid."""
    if not math.isfinite(video_duration) or not 0 < video_duration <= 600:
        raise ValueError("Длительность видео должна быть от 0 до 600 секунд")
    count = math.ceil(video_duration * FRAME_RATE - 1e-9)
    return [round(i / FRAME_RATE, 3) for i in range(max(1, count))]


def frames(path: Path, output: Path, video_duration: float):
    """Extract the complete 5 fps sequence in one FFmpeg pass before any API calls."""
    times = sample_times(video_duration)
    source = path.stat()
    signature = {"version": FRAME_EXTRACTION_VERSION, "fps": FRAME_RATE,
                 "duration": video_duration, "count": len(times),
                 "source_size": source.st_size, "source_mtime_ns": source.st_mtime_ns}
    # Isolate the new sequence from sparse frames produced by older releases.
    target = output / FRAME_EXTRACTION_VERSION
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "manifest.json"
    names = [f"frame_{i + 1:06d}.jpg" for i in range(len(times))]
    try:
        valid = read_json(manifest) == signature
    except (ValueError, OSError):
        valid = False
    valid = valid and all((target / name).is_file() and (target / name).stat().st_size > 0 for name in names)
    if not valid:
        # A failed extraction must never be mistaken for a completed sequence.
        manifest.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=target) as tmp:
            staging = Path(tmp)
            filters = (f"setpts=PTS-STARTPTS,fps=fps={FRAME_RATE}:start_time=0:round=up:eof_action=pass,"
                       "scale=w='min(iw,960)':h='min(ih,960)':force_original_aspect_ratio=decrease")
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                 "-map", "0:v:0", "-an", "-sn", "-vf", filters, "-fps_mode", "passthrough",
                 "-frames:v", str(len(times)), "-q:v", "3", "-start_number", "1",
                 str(staging / "frame_%06d.jpg")], timeout=600)
            extracted = list(staging.glob("frame_*.jpg"))
            if len(extracted) != len(times) or not all((staging / name).is_file() and
                                                        (staging / name).stat().st_size > 0 for name in names):
                raise ValueError(f"Извлечено {len(extracted)} кадров вместо {len(times)}. "
                                 "Анализ не запущен: последовательность 5 кадров/с должна быть полной.")
            for name in names:
                (staging / name).replace(target / name)
            save_json(manifest, signature)
    return [{"time": time, "path": target / name} for time, name in zip(times, names)]


def image_url(path: Path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def assemble_voice(parts, script: Script, output: Path, total_duration: float):
    """Fit small overruns only, align each segment, preserve pauses and the ending."""
    if len(parts) != len(script.segments):
        raise ValueError("Не все реплики озвучены")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters, labels, report = [], [], []
    for i, (path, segment) in enumerate(zip(parts, script.segments)):
        actual = duration(path)
        window = speech_window(segment)
        ratio = actual / window
        if ratio > 1.12:
            raise ValueError(f"Реплика {segment.id}: {actual:.1f} с при окне {window:.1f} с. "
                             "Сократите эту реплику или расширьте её интервал.")
        args += ["-i", str(path)]
        tempo = max(1.0, ratio)
        speech_start = segment.start + segment.delivery.pause_before_ms / 1000
        delay = round(speech_start * 1000)
        filters.append(f"[{i}:a]atempo={tempo:.8f},adelay={delay}:all=1[a{i}]")
        labels.append(f"[a{i}]")
        report.append({"id": segment.id, "original_seconds": actual, "tempo": tempo,
                       "start": speech_start, "end": min(segment.end, speech_start + actual / tempo)})
    mix = f"amix=inputs={len(parts)}:normalize=0," if len(parts) > 1 else ""
    filters.append("".join(labels) + mix + f"apad=whole_dur={total_duration}[out]")
    args += ["-filter_complex", ";".join(filters), "-map", "[out]", "-t", str(total_duration), "-ar", "44100",
             "-c:a", "pcm_s16le", str(output)]
    run(args)
    if abs(duration(output) - total_duration) > 0.15:
        raise ValueError("Длительность собранного звука не совпала с роликом")
    return report


def export_video(video: Path, audio: Path, output: Path, info: VideoInfo, original_volume=0.0):
    if not 0 <= original_volume <= 1:
        raise ValueError("Некорректная громкость")
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(audio)]
    if original_volume and info.has_audio:
        args += ["-filter_complex", f"[0:a]volume={original_volume}[bg];"
                 "[bg][1:a]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95:level=false[a]",
                 "-map", "0:v:0", "-map", "[a]"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    args += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-t", str(info.duration), "-movflags", "+faststart", str(output)]
    run(args, timeout=600)
    if abs(duration(output) - info.duration) > 0.2:
        raise ValueError("Экспорт изменил длительность ролика")


def srt(script: Script, timings=None):
    def stamp(seconds):
        total = round(seconds * 1000)
        hours, rem = divmod(total, 3600000)
        minutes, rem = divmod(rem, 60000)
        sec, ms = divmod(rem, 1000)
        return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"
    result = []
    by_id = {t["id"]: t for t in timings or []}
    for i, segment in enumerate(script.segments, 1):
        t = by_id.get(segment.id, {"start": segment.start, "end": segment.end})
        result.append(f"{i}\n{stamp(t['start'])} --> {stamp(t['end'])}\n{segment.text.strip()}\n")
    return "\n".join(result)
