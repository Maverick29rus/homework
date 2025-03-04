# Импорт необходимых библиотек
import aiosqlite
from db_handlers.create_table import DB_NAME


# Функция добавления имени в базу данных
async def update_name(user_id, name):
    # Создаем соединение с базой данных
    # (если она не существует, она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Вставляем новую запись или заменяем ее,
        # если с данным user_id уже существует
        await db.execute(
            "INSERT INTO quiz_state (user_id, user_name) VALUES (?, ?) \
            ON CONFLICT(user_id) DO UPDATE SET user_name = ?",
            (user_id, name, name),
        )
        # Сохраняем изменения
        await db.commit()


# функция добавления номера текущего вопроса
async def update_quiz_index(user_id, index):
    # Создаем соединение с базой данных
    # (если она не существует, она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Вставляем новую запись или заменяем ее,
        # если с данным user_id уже существует
        await db.execute(
            "INSERT INTO quiz_state (user_id, question_index) VALUES (?, ?) \
            ON CONFLICT(user_id) DO UPDATE SET question_index = ?",
            (user_id, index, index),
        )
        # Сохраняем изменения
        await db.commit()


# функция добавления результата
async def update_result(user_id, result):
    # Создаем соединение с базой данных
    # (если она не существует, она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Вставляем новую запись или заменяем ее,
        # если с данным user_id уже существует
        await db.execute(
            "INSERT INTO quiz_state (user_id, result) VALUES (?, ?) \
            ON CONFLICT(user_id) DO UPDATE SET result = ?",
            (user_id, result, result),
        )
        # Сохраняем изменения
        await db.commit()
