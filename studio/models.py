import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class Delivery(Model):
    emotion: Literal["neutral", "tense", "surprised", "excited", "amused", "relieved", "disappointed"] = "neutral"
    laugh: bool = False
    laugh_position: Literal["before", "after"] = "after"
    pause_before_ms: int = Field(default=0, ge=0, le=800)
    reason: str = ""

    @model_validator(mode="after")
    def grounded(self):
        if (self.emotion != "neutral" or self.laugh or self.pause_before_ms) and not self.reason.strip():
            raise ValueError("Укажите событие, которое объясняет эмоцию, смех или паузу")
        if self.laugh and self.emotion in {"tense", "disappointed"}:
            raise ValueError("Разделите смех и напряжение/разочарование на разные реплики")
        return self


class PerformanceSettings(Model):
    intensity: Literal["restrained", "natural", "expressive"] = "natural"
    allow_laughter: bool = True


class DirectionItem(Model):
    id: str
    delivery: Delivery


class DirectionPlan(Model):
    segments: list[DirectionItem]


class Segment(Model):
    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)
    event_ids: list[str] = Field(min_length=1)
    delivery: Delivery = Field(default_factory=Delivery)

    @field_validator("text")
    @classmethod
    def spoken_words_only(cls, value):
        value = value.strip()
        if not value or re.search(r"\[[^\]\n]*\]|<[^>\n]*>", value):
            raise ValueError("В реплике нужен только произносимый текст. Эмоции задайте в отдельных полях.")
        return value

    @model_validator(mode="after")
    def interval(self):
        if self.end <= self.start:
            raise ValueError("Реплика должна иметь положительную длительность")
        if self.end - self.start - self.delivery.pause_before_ms / 1000 < 0.3:
            raise ValueError("Пауза не оставляет достаточно времени для реплики")
        return self


class Script(Model):
    language: str
    perspective: Literal["observer", "first_person"] = "observer"
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    segments: list[Segment] = Field(min_length=1)

    def check(self, analysis: Analysis, duration: float):
        confirmed = {e.id for e in analysis.events if e.confidence == "confirmed"}
        used_ids = set()
        previous_end = 0.0
        laughs = []
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
            if segment.delivery.laugh:
                if not self.performance.allow_laughter:
                    raise ValueError("Смех отключён в настройках сценария")
                if segment.end - segment.start - segment.delivery.pause_before_ms / 1000 < 2:
                    raise ValueError("Для смеха с репликой оставьте хотя бы две секунды")
                laughs.append(segment.start)
        if len(laughs) > min(2, max(1, math.ceil(duration / 60))):
            raise ValueError("Слишком частый смех: максимум один раз в минутном ролике, два — в более длинном")
        if any(b - a < 12 for a, b in zip(laughs, laughs[1:])):
            raise ValueError("Между эпизодами смеха оставьте не менее 12 секунд")
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
