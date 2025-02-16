# Импорт необходимых библиотек
from aiogram import types, Router, F
from aiogram.filters.command import Command
from handlers.new_quiz import new_quiz


quiz_router = Router()


# Хэндлер на команды /quiz
@quiz_router.message(F.text == "Начать игру")
@quiz_router.message(Command("quiz"))
async def cmd_quiz(message: types.Message):
    # Отправляем новое сообщение без кнопок
    await message.answer("Давайте начнем квиз!")

    # Запускаем новый квиз
    await new_quiz(message)
