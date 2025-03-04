# Импорт необходимых библиотек
from aiogram import types, Router, F
from aiogram.filters.command import Command
from db_handlers.get_hendlers import get_statistics

statistics_router = Router()


# Хэндлер на команды /statistics
@statistics_router.message(F.text == "Статистика")
@statistics_router.message(Command("statistics"))
async def show_statistics(message: types.Message):
    statistics = await get_statistics()
    await message.answer(statistics)
