# streamlit_app.py

import streamlit as st
import openai
from openai import OpenAI
import os
import json
from datetime import datetime
from utils.methodology_handler import MethodologyHandler
from utils.pdf_parser import extract_text_from_pdf
from pathlib import Path

# Инициализация настроек
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

if 'methodology_handler' not in st.session_state:
    st.session_state.methodology_handler = None

# Настройка страницы
st.set_page_config(
    page_title="Система планирования расследований",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка стилей
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button { width: 100%; }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# Создание необходимых директорий
def create_directories():
    try:
        Path("storage/methodologies").mkdir(parents=True, exist_ok=True)
        Path("storage/results").mkdir(parents=True, exist_ok=True)
        Path("storage/files").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Ошибка при создании директорий: {str(e)}")
        raise

# Инициализация OpenAI клиента
def init_openai(api_key):
    try:
        if not api_key:
            st.error("API ключ не может быть пустым")
            return False
        
        client = OpenAI(api_key=api_key)
        st.session_state.openai_client = client
        return True
    except Exception as e:
        st.error(f"Ошибка инициализации OpenAI: {str(e)}")
        return False

# Извлечение фактов из описания
def extract_facts(client, case_description: str) -> str:
    try:
        if not case_description.strip():
            return "Ошибка: Описание дела не может быть пустым"
        
        prompt = f"""
        Проанализируй следующий текст фабулы дела и выдели ключевые факты (даты, события, участников, места, доказательства):
        {case_description}
        Ответ должен быть в виде списка пунктов.
        Квалифицируй это деяние по виду преступления ()
        """
        messages = [
            {"role": "system", "content": "Ты опытный следователь который умеет выделять факты, события, участников, места и доказательства"},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при извлечении фактов: {str(e)}"

# Создание плана расследования
def create_investigation_plan(client, facts: str, methodology_handler=None) -> str:
    try:
        if not facts.strip():
            return "Ошибка: Факты не могут быть пустыми"
        
        # Получаем контекст из методики, если есть
        methodology_context = ""
        if methodology_handler:
            try:
                methodology_context = methodology_handler.get_recommendations_context(facts)
            except Exception as e:
                st.warning(f"Не удалось получить рекомендации из методики: {str(e)}")

        prompt = f"""
        {methodology_context}
        На основе следующих фактов составь план расследования, включающий основные направления проверки, приоритетные действия и рекомендуемые экспертизы:
        Факты: {facts}
        Ответ предоставь в виде структурированного плана.
        """
        messages = [
            {"role": "system", "content": "Ты опытный следователь, умеющий планировать расследование, использующий самые современные методики расследования"},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=900,
            temperature=0.5,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при создании плана: {str(e)}"

# Боковое меню
with st.sidebar:
    st.header("🔧 Настройки")
    
    # Используем API ключ из secrets
    api_key = st.secrets.get("openai_api_key")
    if api_key and init_openai(api_key):
        st.success("API ключ действителен")
    else:
        st.error("Неверный или отсутствующий API ключ OpenAI")
        
    st.header("📚 Навигация")
    page = st.radio(
        "Выберите раздел:",
        ["Настройка методики", "Планирование расследования"]
    )

# Основной контент
if not st.session_state.openai_client:
    st.warning("Пожалуйста, введите API ключ OpenAI в боковом меню")
else:
    if page == "Настройка методики":
        st.header("📚 Загрузка методики расследования")
        
        uploaded_file = st.file_uploader(
            "Выберите PDF файл с методикой",
            type=["pdf"],
            help="Загрузите файл методики в формате PDF"
        )
        
        if uploaded_file is not None:
            if st.button("Обработать методику"):
                with st.spinner("Обработка методики..."):
                    try:
                        # Создаем директории
                        create_directories()
                        
                        # Сохраняем загруженный файл
                        file_path = f"storage/methodologies/{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Извлекаем текст
                        methodology_text = extract_text_from_pdf(file_path)
                        
                        # Инициализируем обработчик методики
                        methodology_handler = MethodologyHandler(api_key=api_key)
                        chunks = methodology_handler.process_methodology(methodology_text)
                        
                        # Сохраняем индекс
                        methodology_handler.save_index("storage/methodologies/index")
                        
                        # Сохраняем в состояние сессии
                        st.session_state.methodology_handler = methodology_handler
                        
                        st.success(f"Методика успешно обработана! Создано {chunks} фрагментов.")
                    
                    except Exception as e:
                        st.error(f"Ошибка при обработке методики: {str(e)}")

    elif page == "Планирование расследования":
        st.header("📋 Планирование расследования")
        
        case_number = st.text_input(
            "Номер дела",
            help="Введите уникальный номер дела"
        )
        
        case_description = st.text_area(
            "Описание фабулы дела",
            height=200,
            help="Введите подробное описание обстоятельств дела"
        )
        
        if st.button("Сформировать план расследования"):
            if not case_number or not case_description:
                st.warning("Заполните все поля формы")
            else:
                with st.spinner("Анализ дела и формирование плана..."):
                    try:
                        # Извлекаем факты
                        facts = extract_facts(st.session_state.openai_client, case_description)
                        
                        # Формируем план
                        plan = create_investigation_plan(
                            st.session_state.openai_client,
                            facts,
                            st.session_state.methodology_handler
                        )
                        
                        # Показываем результаты
                        st.subheader("📝 Извлечённые факты")
                        st.markdown(facts)
                        
                        st.subheader("📌 План расследования")
                        st.markdown(plan)
                        
                        # Создаем результаты для сохранения
                        results = {
                            "case_number": case_number,
                            "facts": facts,
                            "plan": plan,
                            "generated_at": datetime.now().isoformat()
                        }
                        
                        # Сохраняем результаты
                        results_path = f"storage/results/plan_{case_number}.json"
                        with open(results_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        
                        # Кнопка для скачивания
                        with open(results_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                label="💾 Скачать результаты",
                                data=f.read(),
                                file_name=f"plan_{case_number}.json",
                                mime="application/json"
                            )
                    
                    except Exception as e:
                        st.error(f"Ошибка при формировании плана: {str(e)}")

# Нижний колонтитул
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        © 2024 Система планирования расследований | Версия 1.0
    </div>
    """,
    unsafe_allow_html=True 
)