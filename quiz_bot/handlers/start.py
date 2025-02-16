# Импорт необходимых библиотек
from aiogram import types, Router
from aiogram.filters import CommandStart

start_router = Router()

# Хэндлер на команды /start
@start_router.message(CommandStart())
async def cmd_start(message: types.Message):
    # Прикрепляем кнопки к сообщению
    await message.reply("Добро пожаловать в квиз!")
    await message.answer("Пожалуйста, введите ваше имя:")
