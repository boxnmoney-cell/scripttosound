from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VideoInfo(Model):
    duration: float = Field(gt=0, le=600)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_audio: bool


class Event(Model):
    id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    description: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    confidence: Literal["confirmed", "uncertain"]
    source: Literal["video", "user"] = "video"

    @model_validator(mode="after")
    def interval(self):
        if self.end < self.start:
            raise ValueError("Конец события раньше начала")
        return self


class Observation(Model):
    events: list[Event]


class Analysis(Model):
    events: list[Event]
    outcome: Literal["cashout", "loss", "finished", "unknown"]
    outcome_evidence: str
    notes: list[str]

    def check_duration(self, duration: float):
        ids = [e.id for e in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("Повторяющиеся ID событий")
        if any(e.end > duration for e in self.events):
            raise ValueError("Событие выходит за пределы видео")
        if any(b.start < a.start for a, b in zip(self.events, self.events[1:])):
            raise ValueError("События должны идти по времени")
        return self


class Segment(Model):
    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    event_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def interval(self):
        if self.end <= self.start:
            raise ValueError("Реплика должна иметь положительную длительность")
        return self


class Script(Model):
    language: str
    perspective: Literal["observer", "first_person"] = "observer"
    segments: list[Segment] = Field(min_length=1)

    def check(self, analysis: Analysis, duration: float):
        confirmed = {e.id for e in analysis.events if e.confidence == "confirmed"}
        used_ids = set()
        previous_end = 0.0
        for segment in self.segments:
            if segment.id in used_ids:
                raise ValueError("Повторяющиеся ID реплик")
            used_ids.add(segment.id)
            if segment.start < previous_end - 0.001 or segment.end > duration + 0.001:
                raise ValueError("Реплики пересекаются или выходят за пределы ролика")
            if not set(segment.event_ids).issubset(confirmed):
                raise ValueError("Реплика ссылается на неподтверждённые события")
            if not segment.text.strip():
                raise ValueError("Пустая реплика")
            previous_end = segment.end
        return self

    def plain_text(self):
        return "\n\n".join(s.text for s in self.segments)


class Review(Model):
    issues: list[str]


class Profile(Model):
    id: str
    name: str
    vision_rules: str
    writing_rules: str


LANGUAGES = {
    "Português (Brasil)": "pt-BR", "Русский": "ru-RU",
    "English (US)": "en-US", "Español (México)": "es-MX",
    "Español (Latinoamérica)": "es-419", "Tiếng Việt": "vi-VN",
    "Português (Portugal)": "pt-PT", "Deutsch": "de-DE", "Français": "fr-FR",
}
