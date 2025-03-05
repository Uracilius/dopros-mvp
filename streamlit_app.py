import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch  # For GPU detection
import openai
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Streamlit Page Config ==========
st.set_page_config(
    page_title="Анализ лиц в видео",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Создание необходимых директорий ==========
def create_directories():
    """
    Создаёт все нужные папки, если их нет.
    """
    try:
        Path("storage/videos").mkdir(parents=True, exist_ok=True)
        Path("storage/results").mkdir(parents=True, exist_ok=True)
        Path("storage/audio").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Ошибка при создании директорий: {str(e)}")
        raise

def analyze_csv(csv_path):
    """
    Считывает CSV и подготавливает статистику:
    - Количество каждого выражения
    - Средняя уверенность
    Возвращает сам DataFrame и текстовый отчёт.
    """
    try:
        df = pd.read_csv(csv_path)
        expression_counts = df["Expression"].value_counts()
        avg_confidence = df.groupby("Expression")["Confidence"].mean()

        summary = f"""
        📊 **Статистика по выражениям лиц**:
        - Количество выражений:
        {expression_counts.to_string()}
        - Средняя уверенность:
        {avg_confidence.to_string()}
        """
        return df, summary
    except Exception as e:
        return None, f"Ошибка при анализе CSV: {str(e)}"

def detect_emotion_shifts(df):
    """
    Identifies extreme changes in emotions within the DataFrame.
    """
    emotion_changes = []
    prev_emotion = None
    prev_prev_emotion = None

    for index, row in df.iterrows():
        current_emotion = row["Expression"]
        
        if prev_emotion and prev_prev_emotion:
            # Detect a rapid shift (A → B → C)
            if current_emotion != prev_emotion and prev_emotion != prev_prev_emotion and current_emotion != prev_prev_emotion:
                emotion_changes.append(
                    f"⚠️ Резкий переход эмоций: {prev_prev_emotion} → {prev_emotion} → {current_emotion} "
                    f"(время: {row['Timestamp (s)']} сек), (уверенность предикции: {row['Confidence']})"
                )

            # Detect alternation (A → B → A → B)
            if current_emotion == prev_prev_emotion and prev_emotion != current_emotion:
                emotion_changes.append(
                    f"⚠️ Частая смена эмоций: {prev_prev_emotion} → {prev_emotion} → {current_emotion} "
                    f"(время: {row['Timestamp (s)']} сек), (уверенность предикции: {row['Confidence']})"
                )

        prev_prev_emotion = prev_emotion
        prev_emotion = current_emotion

    return emotion_changes

import openai

def get_openai_insights(df):
    """
    Функция анализирует DataFrame, ищет признаки резких эмоциональных изменений,
    указывающих на возможное психологическое давление или попытку уклониться от ответа,
    и формирует запрос к OpenAI для получения краткого аналитического отчёта на русском языке.
    """
    try:
        # Подставьте ваш реальный API ключ
        api_key = st.secrets["openai_api_key"]
        if not api_key:
            return "Ошибка: API ключ отсутствует"

        openai.api_key = api_key

        # Формирование базовых статистических данных по DataFrame
        shape_info = f"Размер таблицы (строки, столбцы): {df.shape}"
        stats_info = df.describe().to_string()
        sample_rows = df.head(10).to_string(index=False)

        # Обнаружение резких изменений эмоций
        emotion_alerts = detect_emotion_shifts(df)
        if emotion_alerts:
            emotion_alerts_summary = "\n".join(emotion_alerts)
        else:
            emotion_alerts_summary = "Нет резких изменений эмоций, требующих безотлагательного внимания."

        summary_prompt = f"""
        Ниже приведены собранные данные и результаты предварительного эмоционального анализа, полученные в ходе опроса/допроса свидетеля или подозреваемого по делу:

        1. Общая информация о данных:
        {shape_info}

        2. Базовая статистика:
        {stats_info}

        3. Пример первых 10 строк:
        {sample_rows}

        4. Резкие изменения эмоций:
        {emotion_alerts_summary}

        На основе этих данных выяви эпизоды, которые могут указывать на психологическое давление, 
        попытку уклониться от ответа или иное нетипичное поведение. 
        Если в ходе анализа обнаружены критические и быстрые скачки эмоций, укажи их временные метки 
        (превращая их из количества секунд в мин:сек). Учитывай все данные, которые я дал. 
        Не прыгай к выводам что человек резко меняет эмоции, так как есть confidence factor, 
        который должен тоже быть высоким. Делай выводы насчет индивидуальных случаев ТОЛЬКО ТОГДА, 
        когда уверенность в предсказании высокая.

        Вот секции которые ОБЯЗАНЫ ПРИСТУТСВОВАТЬ в твоем ответе:
        КРИТИЧЕСКИЕ ЭПИЗОДЫ ЭМОЦИОНАЛЬНОЙ ДЕСТАБИЛИЗАЦИИ:
        СЕРИЯ ВЫСОКОЧАСТОТНЫХ ЭМОЦИОНАЛЬНЫХ ФЛУКТУАЦИЙ:
        ДОПОЛНИТЕЛЬНЫЕ ЭПИЗОДЫ С ВЫСОКИМ УРОВНЕМ ДОСТОВЕРНОСТИ:
        ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ:
        """

        response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Ты выступаешь в роли опытного аналитика, специализирующегося на эмоциональных и поведенческих аспектах допросов."},
                {"role": "user", "content": summary_prompt}
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Ошибка при запросе к OpenAI API: {str(e)}"

# ========== Извлечение аудио из видео ==========
def extract_audio(video_path, audio_path):
    """
    Извлекает аудио из mp4-файла с помощью ffmpeg.
    Требуется, чтобы ffmpeg был установлен в системе.
    """
    try:
        # -y перезаписывает выходной файл, если он уже есть
        cmd = f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le -y "{audio_path}"'
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        raise RuntimeError(f"Не удалось извлечь аудио: {str(e)}")

def transcribe_audio(audio_path: str, whisper_model="turbo") -> str:
    """
    Транскрибирует аудио при помощи OpenAI Whisper API.
    Требуется валидный API-ключ OpenAI (например, через переменную окружения OPENAI_API_KEY).
    """
    try:
        openai.api_key = st.secrets["openai_api_key"]
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru"
            )
        return transcript.text
    except Exception as e:
        raise RuntimeError(f"Ошибка при использовании OpenAI Whisper API: {str(e)}")

# ========== Улучшить/очистить транскрипт с помощью ChatGPT ==========
def enhance_transcript(raw_transcript):
    """
    Отправляет исходный транскрипт в ChatGPT,
    чтобы немного улучшить/очистить текст (исправить орфографию/пунктуацию и т.д.).
    """
    try:
        api_key = st.secrets["openai_api_key"]
        if not api_key:
            return "Ошибка: API ключ отсутствует"

        openai.api_key = api_key

        prompt = f"""
Вот текст транскрипции:
\"\"\"{raw_transcript}\"\"\"

Пожалуйста, сделай жесткую корректировку. Это видео относится к открытым данным по допросу свидетеля или подозреваемого, 
и будет рассматриваться сотрудником правоохранительных органов. и в основном должен начинаться с представления говорящего, 
где они живут и т.д. Твой ответ должен содержать только улучшенный тобой текст, ничего более. разбей на два лица - допрашиваемого и следователя
        """

        response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Ты помощник, который корректирует транскрипции. Ты должен возвращать ВЕСЬ улучшенный текст, который тебе предоставлен"},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при запросе к OpenAI API (enhance_transcript): {str(e)}"

# ========== Сформировать краткое резюме (summary) транскрипта ==========
def summarize_transcript(enhanced_transcript):
    """
    резюмируй.
    """
    try:
        api_key = st.secrets["openai_api_key"]
        if not api_key:
            return "Ошибка: API ключ отсутствует"

        openai.api_key = api_key

        prompt = f"""
Текст для резюме:
\"\"\"{enhanced_transcript}\"\"\" 

Сформируй краткое резюме/суть транскрипта: ТОЛЬКО важные детали.
Это видео относится к открытым данным по допросу свидетеля или подозреваемого, 
и будет рассматриваться сотрудником правоохранительных органов.
        """

        response = openai.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Ты помощник, который делает резюме транскриптов."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при запросе к OpenAI API (summarize_transcript): {str(e)}"

# ========== Запуск анализа лиц ==========
def run_face_analysis(video_path):
    """
    Запускает FaceAnalysisPipeline с GPU при наличии,
    предполагая, что ваш код внутри FaceAnalysisPipeline
    поддерживает параметр device или использует torch.cuda.is_available().
    """
    try:
        st.info("🔍 Запуск Визуального Психоанализа...")

        import sys
        sys.path.append(".")
        from face_analysis_main import FaceAnalysisPipeline  # to avoid re-import issues

        # Пример вызова, если FaceAnalysisPipeline внутри сам решает, использовать ли CUDA
        pipeline = FaceAnalysisPipeline(model_path="./yolov8n-face.pt", video_path=video_path)
        pipeline.run_analysis()

        
        csv_path = pipeline.get_csv_output_path()
        if os.path.exists(csv_path):
            return csv_path
        else:
            return None
    except Exception as e:
        st.error(f"Ошибка при запуске анализа: {str(e)}")
        return None

# ========== Основная часть Streamlit-приложения ==========
def main():
    # Initialize any needed session state variables if they don't already exist
    if "demo_file_used" not in st.session_state:
        st.session_state["demo_file_used"] = False
    if "demo_video_path" not in st.session_state:
        st.session_state["demo_video_path"] = None
    if "uploaded_video_path" not in st.session_state:
        st.session_state["uploaded_video_path"] = None

    st.title("Demo File vs Uploaded File")

    # Button to select the demo file
    if st.button("Использовать демо-файл видеозаписи допроса"):
        st.session_state["demo_file_used"] = True
        st.session_state["demo_video_path"] = "input_video.mp4"
        st.session_state["uploaded_video_path"] = None
        st.success("Демо-видео готово к анализу!")
    
    # File uploader in case user does not want the demo
    if not st.session_state["use_demo_file"]:
        uploaded_file = st.file_uploader("Upload your video file", type=["mp4"])
    if uploaded_file is not None:
        # If user manually uploads, we flip off demo mode
        st.session_state["demo_file_used"] = False
        st.session_state["demo_video_path"] = None

        # Save the uploaded file to disk, store that path in session
        video_save_path = Path("storage/videos") / uploaded_file.name
        with open(video_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_video_path"] = str(video_save_path)
        st.success("Видео успешно загружено!")
    
    # Figure out which video path we should use 
    # (demo vs uploaded), if any
    video_path = None
    if st.session_state["demo_file_used"] and st.session_state["demo_video_path"]:
        video_path = str(Path("storage/videos") / Path(st.session_state["demo_video_path"]).name)
        # Copy your demo file to storage/videos if needed...
        if not os.path.exists(video_path):
            with open(st.session_state["demo_video_path"], "rb") as src:
                data = src.read()
            with open(video_path, "wb") as dst:
                dst.write(data)
            
        st.info("Просмотр демо-видео:")
        st.video(video_path)
    elif st.session_state["uploaded_video_path"]:
        video_path = st.session_state["uploaded_video_path"]

    # Main analysis button
    if video_path is not None and st.button("ЗАПУСТИТЬ КОМПЛЕКСНЫЙ АНАЛИЗ ВЕРБАЛЬНЫХ И НЕВЕРБАЛЬНЫХ РЕАКЦИЙ"):
        # Создадим прогресс-бар для иллюзии поэтапной обработки
        progress_bar = st.progress(0)

        with st.spinner("Обработка видео..."):
            # ========== Шаг A: Анализ лиц ==========
            progress_bar.progress(10)
            csv_path = run_face_analysis(video_path)
            
            if csv_path:
                hash_value = Path(csv_path).stem

                st.success("Файл сохранён под конфиденциальным хэшем:")
                st.code(hash_value, language="text")

                progress_bar.progress(40)
                df, summary = analyze_csv(csv_path)

                # ========== Шаг B: Аналитические инсайты (Completions) ==========
                st.subheader("🧠 АНАЛИЗ ЭМОЦИОНАЛЬНОГО СОСТОЯНИЯ")
                insights = get_openai_insights(df)
                st.write(insights)
                st.download_button(
                    label="💾 ЭКСПОРТ АНАЛИТИЧЕСКИХ ДАННЫХ В CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{Path(video_path).stem}.csv",
                    mime="text/csv"
                )
            else:
                st.error("CSV файл с результатами анализа не найден.")
                progress_bar.progress(40)

            # ========== Шаг C: Извлечение и транскрипция аудио ==========
            try:
                progress_bar.progress(60)
                st.subheader("СТЕНОГРАММА ВЕРБАЛЬНЫХ ПОКАЗАНИЙ")
                audio_path = f"storage/audio/{Path(video_path).stem}.wav"

                st.info("Извлечение аудиодорожки...")
                extract_audio(video_path, audio_path)
                progress_bar.progress(70)

                st.info("Запуск расширенной речевой аналитики")
                transcript_text = transcribe_audio(audio_path, whisper_model="small")
                st.success("Транскрипция успешно завершена!")
                progress_bar.progress(85)

                # ========== Шаг D: Прогон через ChatGPT для улучшения ==========
                st.info("Форматирование транскрипции...")
                enhanced_text = enhance_transcript(transcript_text)
                st.write("**ВЕРИФИЦИРОВАННАЯ СТЕНОГРАММА:**")
                st.write(enhanced_text)
                progress_bar.progress(100)

                # ========== Сформировать краткое резюме (summary) транскрипта ==========
                st.info("Формирование аналитического резюме показаний...")
                summary_text = summarize_transcript(enhanced_text)
                st.subheader("**АНАЛИТИЧЕСКОЕ РЕЗЮМЕ ПОКАЗАНИЙ**")
                st.write(summary_text)

                st.download_button(
                    label="ЭКСПОРТ НЕОБРАБОТАННОЙ СТЕНОГРАММЫ (TXT)",
                    data=transcript_text,
                    file_name="raw_transcript.txt",
                    mime="text/plain"
                )

                st.download_button(
                    label="ЭКСПОРТ ВЕРИФИЦИРОВАННОЙ СТЕНОГРАММЫ (TXT)",
                    data=enhanced_text,
                    file_name="enhanced_transcript.txt",
                    mime="text/plain"
                )

                st.download_button(
                    label="ЭКСПОРТ АНАЛИТИЧЕСКОГО РЕЗЮМЕ (TXT)",
                    data=summary_text,
                    file_name="summary.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Ошибка при транскрипции аудио: {str(e)}")

    # Нижний колонтитул
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            © 2025 СИСТЕМА КОМПЛЕКСНОГО АНАЛИЗА СУДЕБНО-СЛЕДСТВЕННЫХ МАТЕРИАЛОВ | Версия 1.0
        </div>
        """,
        unsafe_allow_html=True 
    )

if __name__ == "__main__":
    main()
