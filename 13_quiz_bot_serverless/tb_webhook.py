import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import BotCommand, BotCommandScopeDefault
from handlers import (
    answer_router,
    start_router,
    quiz_router,
    result_router,
    statistics_router,
    name_router,
)
import json


# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

# С помощью библиотеки os мы можем получить доступ к параметрам из шага 13 по их имени
API_TOKEN = os.getenv("API_TOKEN")
# Создаем объект бота
bot = Bot(token=API_TOKEN)
# Создаем объект диспетчера
dp = Dispatcher()


async def set_commands():
    commands = [
        BotCommand(command="start", description="Старт"),
        BotCommand(command="quiz", description="Начать игру"),
        BotCommand(command="result", description="Мой результат"),
        BotCommand(command="statistics", description="Статистика"),
    ]
    await bot.set_my_commands(commands, BotCommandScopeDefault())


# Добавляем роутеры в наш диспетчер
# Роутеры позволяют вынести в отдельные файлу хендлеры (перехватчики событий)
dp.include_router(answer_router)
dp.include_router(start_router)
dp.include_router(quiz_router)
dp.include_router(result_router)
dp.include_router(statistics_router)
dp.include_router(name_router)


async def process_event(event):
    # Передача полученного сообщения от телеграма в бот
    # Конструкция из официальной документации aiogram для произвольного асинхронного фреймворка
    update = types.Update.model_validate(
        json.loads(event["body"]), context={"bot": bot}
    )
    await dp.feed_update(bot, update)
    await set_commands()


# Точка входа
async def webhook(event, context):
    # Проверка, что прилетел POST-запрос от Telegram
    if event["httpMethod"] == "POST":
        # Bot and dispatcher initialization
        # Объект бота

        # Вызываем коррутин изменения состояния нашего бота
        await process_event(event)
        # Возвращаем код 200 успешного выполнения
        return {"statusCode": 200, "body": "ok"}

    # Если метод не POST-запрос, то выдаем код ошибки 405
    return {"statusCode": 405}
