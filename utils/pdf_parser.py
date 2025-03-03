# utils/pdf_parser.py

import PyPDF2
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFParsingError(Exception):
    """Исключение для ошибок парсинга PDF"""
    pass

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает текст из PDF файла.
    
    Args:
        pdf_path (str): Путь к PDF файлу
        
    Returns:
        str: Извлечённый текст
        
    Raises:
        PDFParsingError: Если возникла ошибка при обработке PDF
        FileNotFoundError: Если файл не найден
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")
        
        # Проверяем расширение файла
        if not pdf_path.lower().endswith('.pdf'):
            raise PDFParsingError(f"Файл {pdf_path} не является PDF")
        
        logger.info(f"Начало обработки файла: {pdf_path}")
        
        # Открываем и читаем PDF
        text_content = []
        with open(pdf_path, 'rb') as file:
            try:
                reader = PyPDF2.PdfReader(file)
                
                # Проверяем, зашифрован ли файл
                if reader.is_encrypted:
                    raise PDFParsingError("PDF файл зашифрован")
                
                # Извлекаем текст из каждой страницы
                total_pages = len(reader.pages)
                logger.info(f"Найдено страниц: {total_pages}")
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        logger.debug(f"Обработка страницы {page_num}/{total_pages}")
                        page_text = page.extract_text()
                        
                        if not page_text:
                            logger.warning(f"Страница {page_num} не содержит текста")
                            continue
                            
                        text_content.append(page_text)
                        
                    except Exception as e:
                        logger.error(f"Ошибка при обработке страницы {page_num}: {str(e)}")
                        continue
                
            except PyPDF2.PdfReadError as e:
                raise PDFParsingError(f"Ошибка чтения PDF: {str(e)}")
        
        # Проверяем, что удалось извлечь текст
        if not text_content:
            raise PDFParsingError("Не удалось извлечь текст из PDF файла")
        
        # Объединяем текст всех страниц
        result = "\n".join(text_content)
        logger.info(f"Успешно извлечен текст из файла: {pdf_path}")
        
        return result
        
    except FileNotFoundError:
        logger.error(f"Файл не найден: {pdf_path}")
        raise
    except PDFParsingError as e:
        logger.error(f"Ошибка парсинга PDF: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Непредвиденная ошибка при обработке PDF: {str(e)}"
        logger.error(error_msg)
        raise PDFParsingError(error_msg)

def get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Получает метаданные PDF файла.
    
    Args:
        pdf_path (str): Путь к PDF файлу
        
    Returns:
        Dict[str, Any]: Словарь с метаданными PDF
        
    Raises:
        PDFParsingError: Если возникла ошибка при чтении метаданных
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")
            
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return {
                'number_of_pages': len(reader.pages),
                'metadata': reader.metadata or {},
                'is_encrypted': reader.is_encrypted,
                'file_size': os.path.getsize(pdf_path),
                'file_name': os.path.basename(pdf_path)
            }
            
    except Exception as e:
        error_msg = f"Ошибка при получении метаданных PDF: {str(e)}"
        logger.error(error_msg)
        raise PDFParsingError(error_msg)

def is_valid_pdf(file_path: str) -> bool:
    """
    Проверяет, является ли файл корректным PDF.
    
    Args:
        file_path (str): Путь к проверяемому файлу
        
    Returns:
        bool: True если файл является корректным PDF, иначе False
    """
    try:
        if not os.path.exists(file_path):
            return False
            
        if not file_path.lower().endswith('.pdf'):
            return False
            
        with open(file_path, 'rb') as file:
            try:
                PyPDF2.PdfReader(file)
                return True
            except:
                return False
    except:
        return False
