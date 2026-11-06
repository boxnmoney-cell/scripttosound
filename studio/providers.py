import json
import time
from pathlib import Path

import httpx

from .storage import digest, read_json, save_json


class APIError(RuntimeError):
    pass


class Providers:
    """Credentials stay in memory. Cached results never contain request headers."""

    def __init__(self, openai_key="", eleven_key="", transport=None):
        self.openai_key, self.eleven_key = openai_key, eleven_key
        self.client = httpx.Client(timeout=httpx.Timeout(180, connect=20), transport=transport)
        self.usage = []

    def close(self):
        self.client.close()

    def request(self, method, url, *, headers, **kwargs):
        for attempt in range(3):
            try:
                response = self.client.request(method, url, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise APIError("Не удалось связаться с API. Проверьте подключение и доступ к сервису.") from exc
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if response.is_error:
                hints = {401: "Проверьте API-ключ.", 403: "Нет доступа к операции или модели.",
                         404: "Модель или голос не найдены.", 422: "Параметры не поддерживаются.",
                         429: "Лимит запросов или недостаточно кредитов."}
                # Do not echo remote error bodies: they can include submitted content or credentials.
                raise APIError(f"API вернул HTTP {response.status_code}. " + hints.get(response.status_code, "Повторите позже."))
            return response
        raise APIError("Запрос не завершён")

    def structured(self, model, system, content, schema, cache: Path | None = None):
        if not self.openai_key:
            raise APIError("Укажите OPENAI_API_KEY в .env или в настройках приложения.")
        system += "\nReturn one JSON object matching this schema exactly:\n" + json.dumps(schema.model_json_schema())
        body = {"model": model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        key = digest(body)
        target = cache / (key + ".json") if cache else None
        if target and target.exists():
            return schema.model_validate(read_json(target))
        data = self.request("POST", "https://api.openai.com/v1/chat/completions",
                            headers={"Authorization": "Bearer " + self.openai_key}, json=body).json()
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Неполный ответ")
            result = schema.model_validate_json(choice["message"]["content"])
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise APIError("Модель вернула неполный или некорректный ответ. Результат не сохранён.") from exc
        self.usage.append({"model": model, "usage": data.get("usage", {})})
        if target:
            save_json(target, result.model_dump())
        return result

    def eleven_get(self, path, **params):
        if not self.eleven_key:
            raise APIError("Укажите ELEVENLABS_API_KEY в настройках или .env.")
        return self.request("GET", "https://api.elevenlabs.io" + path,
                            headers={"xi-api-key": self.eleven_key}, params=params).json()

    def voices(self):
        return self.eleven_get("/v1/voices").get("voices", [])

    def models(self):
        return [m for m in self.eleven_get("/v1/models") if m.get("can_do_text_to_speech")]

    def validate_speech(self, voice_id, model_id, language):
        if not voice_id.isalnum():
            raise ValueError("Некорректный Voice ID")
        selected = next((m for m in self.models() if m["model_id"] == model_id), None)
        if selected is None:
            raise ValueError("Модель не найдена среди доступных моделей синтеза. Обновите список.")
        supported = {item["language_id"] for item in selected.get("languages", [])}
        if supported and language.split("-")[0] not in supported and language not in supported:
            raise ValueError("Модель не заявляет поддержку языка озвучки. Выберите другую модель.")
        self.eleven_get("/v1/voices/" + voice_id)

    def speech(self, text, voice_id, model_id, cache: Path):
        if not self.eleven_key:
            raise APIError("Не указан ключ ElevenLabs")
        # Reject path separators in IDs, including URL query or traversal characters.
        if not voice_id.isalnum() or not all(c.isalnum() or c in "_-" for c in model_id):
            raise ValueError("Некорректный Voice ID или ID модели")
        key = digest({"text": text, "voice": voice_id, "model": model_id, "format": "mp3_44100_128"})
        output = cache / (key + ".mp3")
        if output.exists() and output.stat().st_size:
            return output
        # Do not automatically retry a paid synthesis POST: an interrupted response
        # may already have consumed credits. The user can retry the missing segment.
        try:
            response = self.client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": self.eleven_key}, params={"output_format": "mp3_44100_128"},
                json={"text": text, "model_id": model_id})
        except httpx.TransportError as exc:
            raise APIError("Ответ ElevenLabs не получен. Проверьте историю генераций перед повтором: запрос мог списать кредиты.") from exc
        if response.is_error:
            raise APIError(f"ElevenLabs: HTTP {response.status_code}. Проверьте голос, модель, ключ и кредиты.")
        if not response.content or "audio" not in response.headers.get("content-type", ""):
            raise APIError("ElevenLabs не вернул аудио")
        cache.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".part")
        temp.write_bytes(response.content)
        temp.replace(output)
        return output
