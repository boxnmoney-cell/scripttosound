import json
import re

import httpx
from PIL import Image

from studio.config import ANALYSIS_BATCH_SIZE, OPENAI_MODEL
from studio.media import sample_times
from studio.pipeline import analyze
from studio.profiles import PROFILES
from studio.providers import Providers


def test_all_frames_are_sent_to_luna_as_images_not_video(tmp_path):
    # A static scene can have identical images; every 200ms sample still needs
    # its own timestamp and must not disappear due to image deduplication.
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), "blue").save(image)
    sampled = [{"time": t, "path": image} for t in sample_times(60)]
    seen, counts, models = [], [], []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        models.append(body["model"])
        content = body["messages"][1]["content"]
        assert isinstance(content, list)
        assert all(part["type"] in {"text", "image_url"} for part in content)
        pictures = [part for part in content if part["type"] == "image_url"]
        assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in pictures)
        assert all(p["image_url"]["detail"] == "high" for p in pictures)
        if "outcome_evidence" in body["messages"][0]["content"]:
            assert len(pictures) == 12
            result = {"events": [], "outcome": "unknown", "outcome_evidence": "No verified result", "notes": []}
        else:
            counts.append(len(pictures))
            for part in content:
                if part["type"] == "text":
                    match = re.fullmatch(r"Actual frame time: ([\d.]+)s", part["text"])
                    if match:
                        seen.append(float(match[1]))
            result = {"events": []}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]})

    api = Providers("test", transport=httpx.MockTransport(handler))
    analyze(api, PROFILES[0], sampled, 60, OPENAI_MODEL, tmp_path / "cache")
    assert set(models) == {"gpt-6-luna"}
    assert sorted(set(seen)) == sample_times(60)
    assert len(sampled) == 300
    assert counts[0] == ANALYSIS_BATCH_SIZE
    assert counts[1] == ANALYSIS_BATCH_SIZE + 2
    assert len(counts) == 38 and len(models) == 39
    api.close()
