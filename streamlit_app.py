import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
import face_recognition

# Настройка страницы
st.set_page_config(
    page_title="Анализ лиц в видео",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Создание необходимых директорий
def create_directories():
    try:
        Path("storage/videos").mkdir(parents=True, exist_ok=True)
        Path("storage/results").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Ошибка при создании директорий: {str(e)}")
        raise

# Анализ видео и распознавание лиц
def analyze_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detected_faces = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 != 0:  # Анализируем каждый 10-й кадр
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            for face_location in face_locations:
                top, right, bottom, left = face_location
                detected_faces.append((frame_count, top, right, bottom, left))

        cap.release()
        return detected_faces
    except Exception as e:
        return f"Ошибка при анализе видео: {str(e)}"

# Основной контент
st.header("🎥 Анализ лиц в видео")

uploaded_file = st.file_uploader(
    "Выберите видеофайл (MP4)",
    type=["mp4"],
    help="Загрузите видеофайл в формате MP4"
)

if uploaded_file is not None:
    create_directories()
    video_path = f"storage/videos/{uploaded_file.name}"
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Видео успешно загружено!")
    
    if st.button("Запустить анализ лиц"):
        with st.spinner("Обработка видео..."):
            results = analyze_video(video_path)
            
            if isinstance(results, str):
                st.error(results)
            else:
                st.success(f"Обнаружено {len(results)} лиц на разных кадрах видео.")
                st.write(results)
                
                # Сохранение результатов
                results_path = f"storage/results/{uploaded_file.name}_faces.txt"
                with open(results_path, "w") as f:
                    for entry in results:
                        f.write(f"Frame: {entry[0]}, Top: {entry[1]}, Right: {entry[2]}, Bottom: {entry[3]}, Left: {entry[4]}\n")
                
                with open(results_path, "r") as f:
                    st.download_button(
                        label="💾 Скачать результаты",
                        data=f.read(),
                        file_name=f"{uploaded_file.name}_faces.txt",
                        mime="text/plain"
                    )

# Нижний колонтитул
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        © 2024 Анализ лиц в видео | Версия 1.0
    </div>
    """,
    unsafe_allow_html=True 
)
