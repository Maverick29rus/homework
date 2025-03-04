# Импорт необходимых библиотек
from aiogram import types, Router, F
from aiogram.filters.command import Command
from db_handlers.get_hendlers import get_result

result_router = Router()


# Хэндлер на команды /result
@result_router.message(F.text == "Мой результат")
@result_router.message(Command("result"))
async def show_result(message: types.Message):
    result = await get_result(message.from_user.id)
    await message.answer(f"Ваш результат: {result} правильных ответов\n")
