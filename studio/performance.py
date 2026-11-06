"""Keep acting directions separate from spoken words and subtitle text."""
from .models import Script, Segment
from .storage import digest

EMOTIONS = {
    "neutral": "Спокойно", "tense": "Напряжённо", "surprised": "Удивление",
    "excited": "Радость", "amused": "С улыбкой", "relieved": "Облегчение",
    "disappointed": "Разочарование",
}
INTENSITIES = {"restrained": "Сдержанно", "natural": "Естественно", "expressive": "Выразительно"}
POSITIONS = {"before": "Перед репликой", "after": "После реплики"}

# These are performance hints interpreted by the TTS model, not literal words.
# Delivery and exact laughter length remain voice/model-dependent.
EMOTION_TAGS = {
    "neutral": "", "tense": "[tense]", "surprised": "[surprised]",
    "excited": "[excited]", "amused": "[playfully]", "relieved": "[relieved]",
    "disappointed": "[sad]",
}

DIRECTION_RULES = """ACTING DIRECTION AND NATURAL STORYTELLING:
Build the story from concrete observed actions. The opening should quickly establish
the real situation; no generic 'today we embark on an incredible adventure'. Keep
the strongest reaction for an actual turning point. End once, according to the real
outcome. A clip with little happening may have a quiet delivery and a shorter script.
Vary sentence length and rhythm. Let routine movement breathe; do not fill every gap.
Avoid lists of every object, forced jokes, repeated 'wow', shouting throughout,
financial hype and formulaic conclusions. Use natural language for the requested locale.

Set delivery for each segment separately from its text. The text contains ONLY spoken
words: no [tags], stage directions, 'laughs', 'haha', 'kkkk', SSML or explanations.
Emotion is the narrator's performance, not a factual assertion about the player's mind.
For every non-neutral direction, provide a short Russian reason tied to that segment's
confirmed event_ids. Do not invent danger, intention, failure, rescue or payoff to justify it.
neutral: routine actions and connective narration. Use neutral where there is no strong beat.
tense: a genuinely visible unresolved risk or near miss; never after it already resolved.
surprised: a clearly observed unexpected change, without inventing an impossible event.
excited: a significant confirmed gameplay event, without framing gambling as easy income.
amused: a genuinely light or funny visible moment; not ridicule or cheering someone's losses.
relieved: only after a confirmed resolution of previously established tension.
disappointed: a confirmed setback; never invent loss because the video ended.

Laughter is OPTIONAL, brief and rare. Zero laughs is correct for serious/uneventful footage.
No compulsory quota: maximum one laugh for clips <=60s, at most two in longer clips,
at least 12s apart. Do not laugh through unresolved danger or a loss. Set laugh_position
to before/after depending on the actual beat; leave at least 2s for laughter plus speech.
For natural intensity prefer an amused tone over a full laugh. If allow_laughter=false,
all segments must have laugh=false. Do not narrate that the actual player laughed.

pause_before_ms is real silence inserted before speaking, not a spoken direction.
Use 0 for most segments; an occasional 150–350ms pause can help a turning point.
Keep the phrase aligned to its event and reserve time for pause AND laughter.
Restrained intensity: mostly neutral, rare mild reaction, almost no laughter.
Natural intensity: conversational contrast, a few justified accents, no theatrical arc.
Expressive intensity: stronger contrast at real highlights, still no constant shouting
or mandatory laughter. A small number of meaningful beats is better than many tags.
"""


def automatic_audio_tags(model_id):
    # v3 has documented expressive delivery. Do not infer support for an unknown
    # model ID or a version number from a voice name. Other models need a preview.
    return model_id == "eleven_v3"


def tts_text(segment: Segment, use_tags: bool):
    if not use_tags:
        return segment.text
    direction = segment.delivery
    parts = [EMOTION_TAGS[direction.emotion]]
    if direction.laugh and direction.laugh_position == "before":
        parts.append("[laughs]")
    parts.append(segment.text)
    if direction.laugh and direction.laugh_position == "after":
        parts.append("[laughs]")
    return " ".join(p for p in parts if p)


def speech_window(segment):
    return segment.end - segment.start - segment.delivery.pause_before_ms / 1000


def voice_fingerprint(script: Script, voice_id, model_id, use_tags):
    # Old projects with neutral delivery keep their already-generated audio.
    # Only audible directions/timing change synthesis identity; notes do not.
    payload = script.model_dump(exclude={"performance"})
    audible_direction = False
    for raw, segment in zip(payload["segments"], script.segments):
        raw.pop("delivery")
        if segment.delivery.pause_before_ms:
            raw["pause_before_ms"] = segment.delivery.pause_before_ms
            audible_direction = True
        rendered = tts_text(segment, use_tags)
        if rendered != segment.text:
            raw["tts_text"] = rendered
            audible_direction = True
    if audible_direction:
        payload["delivery_format"] = 1
    return digest({"script": payload, "voice": voice_id, "model": model_id})
