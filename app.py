"""Run with: python -m streamlit run app.py"""
import json
import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from studio import media
from studio.models import Analysis, LANGUAGES, PerformanceSettings, Profile, Script
from studio.pipeline import analyze, direct_script, review, translate, write_script
from studio.performance import EMOTIONS, INTENSITIES, POSITIONS, automatic_audio_tags, speech_window, tts_text, voice_fingerprint
from studio.profiles import COMMON_RULES, PROFILES
from studio.providers import Providers
from studio.storage import digest, invalidate, list_projects, project_for_upload, read_json, save_json

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("SCRIPT_TO_SOUND_DATA_DIR", str(ROOT / "data")))
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Script to Sound", page_icon="🎙️", layout="wide")


def save_state(directory, state):
    save_json(directory / "project.json", state)


def script_rows(script):
    return [{"id": s.id, "start": s.start, "end": s.end,
             "text": s.text, "emotion": EMOTIONS[s.delivery.emotion],
             "laugh": s.delivery.laugh, "laugh_position": POSITIONS[s.delivery.laugh_position],
             "pause_before_ms": s.delivery.pause_before_ms, "reason": s.delivery.reason,
             "event_ids": ", ".join(s.event_ids)} for s in script.segments]


def rows_script(rows, language, perspective="observer", performance=None):
    emotions = {v: k for k, v in EMOTIONS.items()}
    positions = {v: k for k, v in POSITIONS.items()}
    segments = []
    for row in rows:
        segments.append({"id": row["id"], "start": row["start"], "end": row["end"], "text": row["text"],
                         "event_ids": [v.strip() for v in row["event_ids"].split(",") if v.strip()],
                         "delivery": {"emotion": emotions[row["emotion"]], "laugh": row["laugh"],
                                      "laugh_position": positions[row["laugh_position"]],
                                      "pause_before_ms": row["pause_before_ms"], "reason": row["reason"]}})
    return Script(language=language, perspective=perspective,
                  performance=performance or PerformanceSettings(), segments=segments)


def script_columns():
    return {"text": st.column_config.TextColumn("Реплика", width="large"),
            "emotion": st.column_config.SelectboxColumn("Эмоция", options=list(EMOTIONS.values()), required=True),
            "laugh": st.column_config.CheckboxColumn("Короткий смех"),
            "laugh_position": st.column_config.SelectboxColumn("Когда смеяться", options=list(POSITIONS.values()), required=True),
            "pause_before_ms": st.column_config.NumberColumn("Пауза перед, мс", min_value=0, max_value=800, step=50, required=True),
            "reason": st.column_config.TextColumn("Почему здесь", width="medium")}


def main():
    st.title("🎙️ Script to Sound")
    st.caption("Видео → проверяемые события → сценарий на выбранном языке → озвучка")
    prefs = read_json(DATA / "preferences.json", {})
    names = list(LANGUAGES)
    custom_profiles = read_json(DATA / "profiles.json", {})
    profiles = [Profile.model_validate(custom_profiles.get(p.id, p.model_dump())) for p in PROFILES]
    with st.sidebar:
        st.header("Настройки")
        script_label = st.selectbox("Язык сценария", names,
                                   index=names.index(prefs.get("script_label", names[0])))
        voice_options = ["Как сценарий"] + names
        voice_label = st.selectbox("Язык озвучки", voice_options,
                                  index=voice_options.index(prefs.get("voice_label", "Как сценарий")))
        script_language = LANGUAGES[script_label]
        voice_language = script_language if voice_label == "Как сценарий" else LANGUAGES[voice_label]
        st.caption("Если языки различаются, перед синтезом появится отдельный текст озвучки.")
        with st.expander("Ключи и модели"):
            openai_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
            eleven_key = st.text_input("ElevenLabs API key", value=os.getenv("ELEVENLABS_API_KEY", ""), type="password")
            vision_model = st.text_input("Модель анализа", value=os.getenv("OPENAI_VISION_MODEL", "gpt-5.5"))
            text_model = st.text_input("Модель текста", value=os.getenv("OPENAI_TEXT_MODEL", "gpt-5.5"))
            st.caption("Ключи из полей используются только в текущей сессии. Для постоянной настройки заполните локальный .env.")
        api = Providers(openai_key, eleven_key)
        voice_prefs = prefs.get("voices", {}).get(voice_language, {})
        st.subheader("Голос")
        if st.button("Загрузить мои голоса и модели", disabled=not eleven_key):
            try:
                st.session_state["catalog_voices"] = api.voices()
                st.session_state["catalog_models"] = api.models()
            except Exception as exc:
                st.error(str(exc))
        catalog_voices = st.session_state.get("catalog_voices", [])
        if catalog_voices:
            options = ["Ввести Voice ID"] + [v["voice_id"] for v in catalog_voices]
            labels = {v["voice_id"]: v.get("name", v["voice_id"]) + " · " + v["voice_id"] for v in catalog_voices}
            saved_voice = voice_prefs.get("id")
            chosen = st.selectbox("Голос из аккаунта", options, index=options.index(saved_voice) if saved_voice in options else 0,
                                  format_func=lambda v: labels.get(v, v), key="catalog_voice_" + voice_language)
            if chosen != "Ввести Voice ID":
                voice_id = chosen
            else:
                voice_id = st.text_input("Voice ID", value=voice_prefs.get("id", ""), key="voice_" + voice_language)
        else:
            voice_id = st.text_input("Voice ID", value=voice_prefs.get("id", ""), key="voice_" + voice_language)
        st.caption("Ваш ориентир: André — Natural excitement. Точный ID берётся из библиотеки ElevenLabs.")
        catalog_models = st.session_state.get("catalog_models", [])
        if catalog_models:
            ids = [m["model_id"] for m in catalog_models]
            saved_model = voice_prefs.get("model")
            tts_model = st.selectbox("Модель ElevenLabs", ids, index=ids.index(saved_model) if saved_model in ids else 0)
        else:
            tts_model = st.text_input("ID модели ElevenLabs", value=voice_prefs.get("model", ""),
                                      placeholder="Загрузите доступные модели", key="tts_model_" + voice_language)
        if st.button("Сохранить языки и голос"):
            prefs.update(script_label=script_label, voice_label=voice_label)
            prefs.setdefault("voices", {})[voice_language] = {"id": voice_id.strip(), "model": tts_model.strip()}
            save_json(DATA / "preferences.json", prefs)
            st.success("Сохранено без API-ключей")
        with st.expander("История проектов"):
            projects = list_projects(DATA)
            if projects:
                selected = st.selectbox("Проект", range(len(projects)), format_func=lambda i: projects[i][1])
                if st.button("Открыть проект"):
                    st.session_state["active_project"] = str(projects[selected][0])
                    st.rerun()
            else:
                st.caption("Проекты появятся после загрузки видео.")

    try:
        run_project(api, profiles, custom_profiles, script_language, voice_language,
                    vision_model, text_model, voice_id.strip(), tts_model.strip())
    finally:
        if api.usage:
            st.caption("API за текущую операцию: " + str(len(api.usage)) + " запросов; " +
                       str(sum(item["usage"].get("total_tokens", 0) for item in api.usage)) + " токенов. Кэшированные ответы не учитываются.")
        api.close()


def run_project(api, profiles, custom_profiles, script_language, voice_language,
                vision_model, text_model, voice_id, tts_model):
    with st.expander("Новый проект", expanded="active_project" not in st.session_state):
        profile_choice = st.selectbox("Игра", profiles, format_func=lambda p: p.name)
        upload = st.file_uploader("Видео", type=["mp4", "mov", "mkv", "avi", "webm"])
        if st.button("Создать / открыть проект этого видео", disabled=upload is None):
            try:
                directory, video = project_for_upload(DATA, upload.getvalue(), Path(upload.name).suffix, profile_choice.id)
                info = media.video_info(video)
                if not (directory / "project.json").exists():
                    save_state(directory, {"name": upload.name, "video": video.name,
                                          "profile_id": profile_choice.id, "info": info.model_dump()})
                st.session_state["active_project"] = str(directory)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if "active_project" not in st.session_state:
        st.info("Выберите игру и загрузите видео. Ключи нужны при обращении к API; редактор и настройки доступны сразу.")
        st.markdown("Сценарий и озвучка могут быть на одном языке или на разных. Анализ сохраняется для новых языковых версий.")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            st.warning("Для обработки видео установите FFmpeg. Инструкция находится в README.md.")
        return
    directory = Path(st.session_state["active_project"])
    state = read_json(directory / "project.json")
    profile = next(p for p in profiles if p.id == state["profile_id"])
    video = directory / state["video"]
    from studio.models import VideoInfo
    info = VideoInfo.model_validate(state["info"])
    cache = directory / "cache"
    st.subheader(state["name"])
    st.caption(f"{profile.name} · {info.width}×{info.height} · {info.duration:.1f} с")
    with st.expander("Исходное видео"):
        st.video(str(video))
        st.caption("Старая озвучка и встроенные субтитры не считаются подтверждением событий.")
    with st.expander("Правила этой игры"):
        st.caption("Профили предварительные: точность механик требует проверки на реальном видео.")
        vision_rules = st.text_area("Что распознавать", profile.vision_rules, height=200, key="vision_" + profile.id)
        writing_rules = st.text_area("Правила сценария", profile.writing_rules, height=160, key="writer_" + profile.id)
        if st.button("Сохранить профиль"):
            updated = profile.model_copy(update={"vision_rules": vision_rules, "writing_rules": writing_rules})
            custom_profiles[profile.id] = updated.model_dump()
            save_json(DATA / "profiles.json", custom_profiles)
            st.rerun()
        st.text(COMMON_RULES)

    st.subheader("1. События видео")
    quality = st.selectbox("Подробность", ["Обычный анализ", "Подробный анализ"])
    st.caption("До 240 основных кадров и дополнительные кадры начала и финала. Ключевые события между кадрами могут остаться нераспознанными.")
    if st.button("Проанализировать видео", disabled=not api.openai_key, type="primary"):
        status = st.empty()
        try:
            with st.spinner("Извлекаю и анализирую кадры…"):
                sampled = media.frames(video, directory / "frames", info.duration, fps=1 if quality == "Обычный анализ" else 2)
                sampled += media.frames(video, directory / "frames", info.duration, fps=3, end=min(3, info.duration))
                sampled += media.frames(video, directory / "frames", info.duration, fps=4, start=max(0, info.duration - 5))
                sampled = sorted({f["time"]: f for f in sampled}.values(), key=lambda f: f["time"])
                result = analyze(api, profile, sampled, info.duration, vision_model, cache, status.info)
            state["analysis"] = result.model_dump()
            state["analysis_profile"] = digest(profile.model_dump())
            invalidate(state, "analysis")
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            status.empty()
            st.error(str(exc))
    if "analysis" not in state:
        return
    analysis = Analysis.model_validate(state["analysis"]).check_duration(info.duration)
    profile_current = state.get("analysis_profile") == digest(profile.model_dump())
    if not profile_current:
        st.warning("Профиль изменён. Повторите анализ перед созданием сценария; прежние результаты сохранены.")
    with st.expander("Проверить / исправить события", expanded=not bool(state.get("script"))):
        for note in analysis.notes:
            st.info(note)
        st.caption("confirmed — подтверждено; uncertain — сомнительно, в сценарий не попадёт. Можно добавлять и удалять события. Правки помечаются как внесённые вами.")
        event_rows = st.data_editor([e.model_dump() for e in analysis.events],
                                   disabled=["id", "source"], num_rows="dynamic", width="stretch",
                                   key="events_" + digest(state["analysis"]))
        outcome = st.selectbox("Результат", ["unknown", "finished", "cashout", "loss"],
                               index=["unknown", "finished", "cashout", "loss"].index(analysis.outcome))
        outcome_evidence = st.text_input("Основание для результата", analysis.outcome_evidence)
        if st.button("Сохранить исправления событий"):
            try:
                old = {e.id: e.model_dump() for e in analysis.events}
                for index, row in enumerate(event_rows):
                    if not row.get("id"):
                        row["id"] = "U" + digest({"row": row, "index": index})[:10]
                    if row != old.get(row["id"]):
                        row["source"] = "user"
                    if not row.get("confidence"):
                        row["confidence"] = "uncertain"
                edited = Analysis(events=event_rows, outcome=outcome, outcome_evidence=outcome_evidence, notes=analysis.notes)
                edited.check_duration(info.duration)
                if edited.model_dump() != state["analysis"]:
                    state["analysis"] = edited.model_dump()
                    invalidate(state, "analysis")
                    save_state(directory, state)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if analysis.events:
            event = st.selectbox("Показать момент", analysis.events, format_func=lambda e: f"{e.start:.1f} с — {e.description}")
            st.video(str(video), start_time=int(event.start))

    st.subheader("2. Сценарий")
    style = st.selectbox("Подача", ["Естественный комментарий", "Спокойный рассказ", "Эмоциональный комментарий"])
    saved_performance = PerformanceSettings.model_validate(state.get("script", {}).get("performance", {}))
    intensity = st.selectbox("Сила эмоций", list(INTENSITIES), index=list(INTENSITIES).index(saved_performance.intensity),
                             format_func=INTENSITIES.get, key="intensity_" + directory.name)
    allow_laughter = st.checkbox("Разрешить уместный короткий смех", value=saved_performance.allow_laughter,
                                 key="laughter_" + directory.name)
    performance = PerformanceSettings(intensity=intensity, allow_laughter=allow_laughter)
    st.caption("Смех не обязателен. GPT привязывает эмоции к подтверждённым событиям, оставляет спокойные участки и убирает пустые фразы.")
    perspective = st.selectbox("От чьего лица", ["Наблюдатель", "Мой собственный раунд — от первого лица"])
    revision = st.text_input("Пожелание к тексту", placeholder="Например: короче начало, меньше восклицаний")
    col1, col2 = st.columns(2)
    with col1:
        generate = st.button("Создать новый вариант", disabled=not api.openai_key or not profile_current)
    with col2:
        translate_existing = st.button("Перевести существующий сценарий", disabled="script" not in state or not api.openai_key or not profile_current)
    if generate or translate_existing:
        try:
            with st.spinner("Готовлю текст и проверяю соответствие событиям…"):
                old_script = Script.model_validate(state["script"]) if state.get("script") else None
                if translate_existing:
                    result = translate(api, old_script, script_language, analysis, profile, text_model, cache)
                else:
                    result = write_script(api, analysis, info.duration, profile, script_language,
                                          style, perspective, text_model, cache, revision, old_script,
                                          state.get("variation", 0), performance=performance)
                result.check(analysis, info.duration)
            invalidate(state, "script")
            state["script"] = result.model_dump()
            state["variation"] = state.get("variation", 0) + 1
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if "script" not in state:
        return
    script = Script.model_validate(state["script"])
    language_current = script.language == script_language
    performance_current = script.performance == performance
    if not performance_current:
        st.info("Настройки эмоций изменены. Создайте новый вариант или нажмите «Расставить эмоции, сохранив текст».")
    st.caption("Язык сохранённого сценария: " + script.language)
    if not language_current:
        st.info("Выбран другой язык. Переведите существующий сценарий или создайте новый вариант.")
    edited_rows = st.data_editor(script_rows(script), disabled=["id"], num_rows="fixed", width="stretch",
                                key="script_" + digest(state["script"]), column_config=script_columns())
    st.caption("У каждой реплики можно изменить эмоцию, смех и паузу. В «Почему здесь» укажите событие, объясняющее акцент. В самом тексте пометки не нужны.")
    dirty = edited_rows != script_rows(script)
    if dirty:
        st.info("Есть несохранённые правки. Сохраните их перед озвучкой.")
    if st.button("Сохранить текст и время реплик"):
        try:
            updated = rows_script(edited_rows, script.language, script.perspective, script.performance).check(analysis, info.duration)
            if updated.model_dump() != state["script"]:
                invalidate(state, "script")
                state["script"] = updated.model_dump()
                save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Расставить эмоции, сохранив текст", disabled=not api.openai_key or dirty or not profile_current or not language_current):
        try:
            with st.spinner("Подбираю эмоции и проверяю их уместность…"):
                updated = direct_script(api, script, analysis, info.duration, profile, performance, text_model, cache)
            if updated.model_dump() != state["script"]:
                invalidate(state, "script")
                state["script"] = updated.model_dump()
                save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.download_button("Скачать сценарий TXT", script.plain_text(), "script.txt")
    st.download_button("Скачать сценарий JSON", script.model_dump_json(indent=2), "script.json")
    if script.language != "ru-RU" and st.button("Показать перевод на русский", disabled=not api.openai_key or dirty):
        try:
            state["translation"] = translate(api, script, "ru-RU", analysis, profile, text_model, cache).model_dump()
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if state.get("translation"):
        with st.expander("Русский перевод для проверки", expanded=True):
            st.write(Script.model_validate(state["translation"]).plain_text())

    st.subheader("3. Озвучка")
    if voice_language == script.language:
        voice_script = script
    else:
        st.info(f"Для озвучки выбран {voice_language}. Подготовьте и проверьте перевод.")
        if st.button("Подготовить текст озвучки", disabled=not api.openai_key or dirty or not language_current):
            try:
                result = translate(api, script, voice_language, analysis, profile, text_model, cache)
                state["voice_script"] = result.model_dump()
                invalidate(state, "voice_script")
                save_state(directory, state)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        raw_voice = state.get("voice_script")
        if not raw_voice or raw_voice["language"] != voice_language:
            return
        voice_script = Script.model_validate(raw_voice)
        voice_rows = st.data_editor(script_rows(voice_script), disabled=["id", "start", "end", "event_ids"],
                                    width="stretch", key="voicetext_" + digest(raw_voice), column_config=script_columns())
        dirty_voice = voice_rows != script_rows(voice_script)
        if st.button("Сохранить текст озвучки"):
            try:
                updated = rows_script(voice_rows, voice_language, voice_script.perspective, voice_script.performance).check(analysis, info.duration)
                if updated.model_dump() != state["voice_script"]:
                    state["voice_script"] = updated.model_dump()
                    invalidate(state, "voice_script")
                    save_state(directory, state)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        dirty = dirty or dirty_voice
    use_tags = st.toggle("Передавать эмоции и смех в ElevenLabs", value=automatic_audio_tags(tts_model), key="audio_tags_" + tts_model)
    if use_tags and not automatic_audio_tags(tts_model):
        st.warning("Поддержка аудиотегов этой моделью не подтверждена. Сначала прослушайте пробу. Если пометки читаются вслух, отключите их или выберите eleven_v3.")
    elif not use_tags and any(s.delivery.laugh or s.delivery.emotion != "neutral" for s in voice_script.segments):
        st.info("Аудиотеги выключены: текст и паузы сохранятся, но явные команды эмоций и смеха не передадутся. Автоматически теги включаются для eleven_v3.")
    fingerprint = voice_fingerprint(voice_script, voice_id, tts_model, use_tags)
    rendered = {s.id: tts_text(s, use_tags) for s in voice_script.segments}
    with st.expander("Текст, который получит ElevenLabs"):
        for segment in voice_script.segments:
            st.text(f"{segment.id} · {segment.start:.2f}–{segment.end:.2f} с\n{rendered[segment.id]}")
        st.caption("Пометки синтеза не попадают в обычный TXT и субтитры. Реакция конкретного голоса на теги требует прослушивания.")
        st.download_button("Скачать текст для ElevenLabs", "\n\n".join(rendered.values()), "elevenlabs_script.txt")
    st.caption(f"Синтез: {voice_language} · {len(voice_script.segments)} реплик. Оплачиваются запросы ElevenLabs; неизменённые фрагменты берутся из кэша.")
    can_synthesize = not dirty and language_current and performance_current and profile_current and bool(api.eleven_key and api.openai_key and voice_id and tts_model)

    def validate_synthesis():
        voice_script.check(analysis, info.duration)
        issues = review(api, voice_script, analysis, profile, text_model, cache).issues
        if issues:
            raise ValueError("Исправьте текст или подачу: " + "; ".join(issues))
        api.validate_speech(voice_id, tts_model, voice_script.language)

    preview_segment = st.selectbox("Пробная реплика", voice_script.segments,
                                    format_func=lambda s: f"{s.id} · {EMOTIONS[s.delivery.emotion]} · {s.text[:80]}")
    preview_key = digest({"fingerprint": fingerprint, "id": preview_segment.id})
    if st.button("Прослушать одну реплику", disabled=not can_synthesize):
        try:
            with st.spinner("Готовлю пробную озвучку…"):
                validate_synthesis()
                part = api.speech(rendered[preview_segment.id], voice_id, tts_model, directory / "tts_cache")
                seconds = media.duration(part)
            state["preview"] = {"key": preview_key, "path": str(part.relative_to(directory)), "duration": seconds}
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if state.get("preview", {}).get("key") == preview_key and not dirty and performance_current:
        st.audio(str(directory / state["preview"]["path"]))
        st.caption("Проба — одна реплика без тишины перед ней. Указанная пауза добавится при сборке полного ролика.")
        if state["preview"]["duration"] > speech_window(preview_segment) * 1.12:
            st.warning("Проба длиннее доступного интервала. Сократите текст, уберите смех или расширьте время реплики.")
    if st.button("Создать озвучку", disabled=not can_synthesize, type="primary"):
        try:
            with st.spinner("Проверяю текст, голос и модель…"):
                validate_synthesis()
            parts = []
            status = st.empty()
            for i, segment in enumerate(voice_script.segments, 1):
                status.info(f"Озвучка {i}/{len(voice_script.segments)}: {segment.id}")
                part = api.speech(rendered[segment.id], voice_id, tts_model, directory / "tts_cache")
                # Stop at the first overlong fragment, before paying for later ones.
                actual = media.duration(part)
                if actual > speech_window(segment) * 1.12:
                    raise ValueError(f"Реплика {segment.id} длится {actual:.1f} с. Сократите её или расширьте интервал. "
                                     "Уже созданные фрагменты сохранены в кэше.")
                parts.append(part)
            audio_name = "voice-" + fingerprint[:16] + ".wav"
            timing = media.assemble_voice(parts, voice_script, directory / audio_name, info.duration)
            state["audio"] = {"path": audio_name, "fingerprint": fingerprint, "timing": timing}
            invalidate(state, "audio")
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    audio = state.get("audio")
    current_audio = audio and audio["fingerprint"] == fingerprint and not dirty and language_current and profile_current and performance_current
    if audio and not current_audio:
        st.warning("Текст, язык, голос или модель изменились. Для экспорта нужна актуальная озвучка.")
    if not current_audio:
        return
    audio_path = directory / audio["path"]
    st.audio(str(audio_path))
    st.download_button("Скачать озвучку WAV", audio_path.read_bytes(), "voiceover.wav", mime="audio/wav")
    st.download_button("Скачать субтитры SRT", media.srt(voice_script, audio["timing"]), "subtitles.srt")
    st.caption("Субтитры привязаны к репликам; пословной подсветки нет.")

    st.subheader("4. Экспорт")
    original_volume = st.slider("Громкость исходного звука", 0.0, 1.0, 0.0, 0.05)
    if original_volume:
        st.info("Если в исходнике есть голос, он тоже сохранится. Для ролика со старой озвучкой оставьте громкость 0.")
    export_key = digest({"audio": fingerprint, "original_volume": original_volume})
    if st.button("Собрать MP4"):
        try:
            output_name = "export-" + export_key[:16] + ".mp4"
            with st.spinner("Собираю видео…"):
                media.export_video(video, audio_path, directory / output_name, info, original_volume)
            state["export"] = {"path": output_name, "key": export_key}
            save_state(directory, state)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if state.get("export", {}).get("key") == export_key:
        output = directory / state["export"]["path"]
        st.video(str(output))
        st.download_button("Скачать готовый ролик", output.read_bytes(), "narrated_video.mp4", mime="video/mp4")
    st.caption("Проверка текста не гарантирует допуск ролика или ссылок правилами платформ. Перед публикацией проверьте результат.")


if __name__ == "__main__":
    main()
