import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def project_for_upload(root: Path, content: bytes, suffix: str, profile_id: str):
    if suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise ValueError("Неподдерживаемый формат видео")
    if not re.fullmatch(r"[a-z0-9_-]+", profile_id):
        raise ValueError("Некорректный профиль")
    key = hashlib.sha256(content).hexdigest()[:24] + "-" + profile_id
    directory = root / "projects" / key
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / ("source" + suffix.lower())
    if not video.exists():
        video.write_bytes(content)
    return directory, video


def list_projects(root: Path):
    projects = []
    for path in sorted((root / "projects").glob("*/project.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            state = read_json(path)
            projects.append((path.parent, state.get("name", path.parent.name)))
        except (ValueError, OSError):
            continue
    return projects


def invalidate(state, stage):
    stages = {
        "analysis": ["script", "voice_script", "translation", "audio", "export"],
        "script": ["voice_script", "translation", "audio", "export"],
        "voice_script": ["audio", "export"],
        "audio": ["export"],
    }
    for key in stages[stage]:
        state.pop(key, None)
