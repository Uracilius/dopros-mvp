# utils/methodology_handler.py

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

class MethodologyHandler:
    def __init__(self, api_key: str):
        """
        Инициализация обработчика методики.
        Args:
            api_key (str): API ключ для OpenAI
        """
        os.environ["openai_api_key"] = api_key
        # Инициализируем embeddings с моделью text-embedding-3-small
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        self.vector_store = None
        
        # Создаем разделитель текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n",  # Сначала разделяем по двойным переносам строк (параграфы)
                "\n",    # Затем по одинарным переносам
                ". ",    # Затем по точкам с пробелами (предложения)
                ", ",    # По запятым
                " ",     # По пробелам (слова)
                ""       # И наконец, по символам, если ничего другого не осталось
            ]
        )

    def process_methodology(self, methodology_text: str) -> int:
        """
        Обработка текста методики: разбиение на смысловые части и создание векторного индекса.
        Args:
            methodology_text (str): полный текст методики
        Returns:
            int: количество созданных документов
        """
        try:
            # Создаём документы из текста методики
            documents = self.text_splitter.create_documents([methodology_text])
            
            # Добавляем метаданные к каждому документу
            for i, doc in enumerate(documents):
                doc.metadata = {
                    "chunk_id": i,
                    "source": "methodology",
                    "chunk_size": len(doc.page_content)
                }
            
            # Создаём векторное хранилище
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            return len(documents)
            
        except Exception as e:
            print(f"Ошибка при обработке методики: {str(e)}")
            raise

    def find_relevant_recommendations(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Поиск релевантных рекомендаций из методики.
        Args:
            query (str): текст запроса (например, описание дела или фактов)
            top_k (int): количество рекомендаций для возврата
        Returns:
            List[Document]: список релевантных документов методики
        """
        if not self.vector_store:
            raise ValueError("Методика ещё не загружена. Сначала вызовите process_methodology()")
        
        # Поиск релевантных фрагментов с MMR переранжированием для разнообразия
        return self.vector_store.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=top_k * 2  # Получаем больше кандидатов для лучшего разнообразия
        )

    def get_recommendations_context(self, case_facts: str) -> str:
        """
        Формирование контекста из релевантных рекомендаций для создания плана.
        Args:
            case_facts (str): извлечённые факты по делу
        Returns:
            str: отформатированный контекст с рекомендациями
        """
        relevant_docs = self.find_relevant_recommendations(case_facts)
        
        # Форматируем рекомендации в единый текст
        context = "На основе методических рекомендаций:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            # Добавляем информацию о части методики и её размере
            context += f"{i}. [Часть {doc.metadata['chunk_id']}] {doc.page_content}\n\n"
        
        return context

    def save_index(self, path: str):
        """
        Сохранение векторного индекса на диск.
        Args:
            path (str): путь для сохранения
        """
        if self.vector_store:
            self.vector_store.save_local(path)

    def load_index(self, path: str):
        """
        Загрузка векторного индекса с диска.
        Args:
            path (str): путь к сохранённому индексу
        """
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings)
        else:
            raise FileNotFoundError(f"Индекс не найден по пути: {path}")