"""The requested model and sampling policy are fixed for this application."""

OPENAI_MODEL = "gpt-6-luna"
FRAME_RATE = 5
FRAME_EXTRACTION_VERSION = "uniform-5fps-v1"
ANALYSIS_BATCH_SIZE = 8
ANALYSIS_OVERLAP = 2


def analysis_settings():
    return {"model": OPENAI_MODEL, "fps": FRAME_RATE,
            "frames_version": FRAME_EXTRACTION_VERSION}
