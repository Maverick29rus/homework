# Импорт необходимых библиотек
import asyncio
from create_bot import bot, dp
from db_handlers.create_table import create_table
from handlers.start import start_router
from handlers.quiz import quiz_router
from handlers.answers import answer_router
from handlers.name import name_router
from handlers.statistics import statistics_router
from handlers.result import result_router
from commands import set_commands


# Запуск процесса поллинга новых апдейтов
async def main():

    # Запускаем создание таблицы базы данных
    await create_table()
    # Добавляем роутеры
    dp.include_router(start_router)
    dp.include_router(quiz_router)
    dp.include_router(answer_router)
    dp.include_router(statistics_router)
    dp.include_router(result_router)
    dp.include_router(name_router)
    # Запускаем пуллинг бота
    await dp.start_polling(bot)
    await set_commands()


if __name__ == "__main__":
    asyncio.run(main())
