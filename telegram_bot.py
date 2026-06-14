"""
Модуль для Telegram бота, интегрированного с RAG-ассистентом.

Бот позволяет пользователям задавать вопросы ассистенту через Telegram
и получать ответы на основе векторного поиска и LLM.
"""

import os
import time
from typing import Optional
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from rag_pipeline import RAGPipeline
from cache import RAGCache
from db_logger import DatabaseLogger


class TelegramRAGBot:
    """
    Telegram бот для RAG-ассистента.
    
    Обрабатывает команды и сообщения от пользователей,
    логирует все взаимодействия в базу данных.
    """
    
    def __init__(
        self,
        token: str,
        rag_assistant: RAGPipeline,
        cache: RAGCache,
        logger: DatabaseLogger
    ):
        """
        Инициализация Telegram бота.
        
        Args:
            token: Токен Telegram бота от @BotFather
            rag_assistant: Экземпляр RAG-ассистента
            cache: Экземпляр кеша ответов
            logger: Экземпляр логгера базы данных
        """
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.logger = logger
        
        # Создаем приложение Telegram
        self.application = Application.builder().token(token).build()
        
        # Регистрируем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        
        # Регистрируем обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """
🤖 Добро пожаловать в RAG-ассистента!

Я могу отвечать на ваши вопросы, используя базу знаний.

Доступные команды:
/help - показать справку
/stats - статистика системы
/logs - получить логи в CSV формате

Просто напишите мне вопрос, и я постараюсь на него ответить!
        """
        await update.message.reply_text(welcome_message.strip())
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📚 Справка по использованию бота:

• Просто напишите вопрос - я отвечу на основе базы знаний
• Использую RAG (Retrieval-Augmented Generation) для точных ответов
• Ответы кешируются для быстрой работы

Команды:
/start - начать работу с ботом
/help - показать эту справку
/stats - статистика системы (документы, кеш)
/logs - получить логи взаимодействий в CSV формате

Примеры вопросов:
• "Что такое Python?"
• "Расскажи про RAG"
• "Что такое векторные базы данных?"
        """
        await update.message.reply_text(help_text.strip())
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            # Получаем статистику системы
            stats = self.rag_assistant.vector_store.get_collection_stats()
            doc_count = stats.get("count", 0)
            cache_size = self.cache.get_stats()["total_entries"]
            model = self.rag_assistant.model
            
            # Получаем статистику из логов
            log_stats = self.logger.get_stats()
            
            stats_message = f"""
📊 СТАТИСТИКА СИСТЕМЫ:

📚 База знаний:
  • Документов в ChromaDB: {doc_count}
  • Модель LLM: {model}

💾 Кеш:
  • Записей в кеше: {cache_size}

📝 Логи:
  • Всего запросов: {log_stats['total_requests']}
  • Из кеша: {log_stats['cached_requests']}
  • Уникальных пользователей: {log_stats['unique_users']}
  • Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс
            """
            
            await update.message.reply_text(stats_message.strip())
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при получении статистики: {str(e)}")
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)
            
            # Экспортируем логи текущего пользователя
            csv_content = self.logger.export_to_csv(user_id=user_id)
            
            if not csv_content:
                await update.message.reply_text(
                    "📝 Логов для вашего пользователя не найдено."
                )
                return
            
            # Сохраняем во временный файл
            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            # Отправляем файл пользователю
            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="📊 Ваши логи взаимодействий с ботом"
                )
            
            # Удаляем временный файл
            os.remove(filename)
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при экспорте логов: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений от пользователей"""
        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"
        
        # Показываем, что бот печатает
        await update.message.chat.send_action(action="typing")
        
        start_time = time.time()
        
        try:
            # Выполняем RAG запрос (кеш проверяется внутри pipeline)
            result = self.rag_assistant.query(user_message, use_cache=True)
            answer = result["answer"]
            from_cache = result.get("from_cache", False)
            
            # Вычисляем время ответа
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Логируем взаимодействие
            self.logger.log_interaction(
                query=user_message,
                response=answer,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=from_cache,
                response_time_ms=response_time_ms
            )
            
            # Отправляем ответ пользователю
            # Разбиваем длинные ответы на части (Telegram имеет лимит 4096 символов)
            max_length = 4000
            if len(answer) <= max_length:
                await update.message.reply_text(answer)
            else:
                # Отправляем частями
                parts = [answer[i:i+max_length] for i in range(0, len(answer), max_length)]
                for i, part in enumerate(parts):
                    if i == 0:
                        await update.message.reply_text(part)
                    else:
                        await update.message.reply_text(part)
            
            # Добавляем индикатор, если ответ из кеша
            if from_cache:
                await update.message.reply_text("💾 (ответ из кеша)", quote=False)
        
        except Exception as e:
            error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            await update.message.reply_text(error_message)
            
            # Логируем ошибку
            self.logger.log_interaction(
                query=user_message,
                response=error_message,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def run(self):
        """Запускает бота"""
        print("🤖 Запуск Telegram бота...")
        print("Бот готов к работе! Нажмите Ctrl+C для остановки.")
        self.application.run_polling()


def main():
    """Точка входа для запуска Telegram бота."""
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен в .env")
        return

    try:
        rag = RAGPipeline()
        cache = RAGCache(db_path=os.getenv("CACHE_DB_PATH", "rag_cache.db"))
        logger = DatabaseLogger()

        bot = TelegramRAGBot(
            token=token,
            rag_assistant=rag,
            cache=cache,
            logger=logger
        )
        bot.run()
    except Exception as e:
        print(f"❌ Ошибка при запуске бота: {e}")


if __name__ == "__main__":
    main()

