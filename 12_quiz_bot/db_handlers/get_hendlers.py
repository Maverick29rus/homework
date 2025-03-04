# Импорт необходимых библиотек
import aiosqlite
from db_handlers.create_table import DB_NAME


# функция получения номера текущего вопроса
async def get_quiz_index(user_id):
    # Подключаемся к базе данных
    async with aiosqlite.connect(DB_NAME) as db:
        # Получаем запись для заданного пользователя
        async with db.execute(
            "SELECT question_index FROM quiz_state WHERE user_id = (?)", (user_id,)
        ) as cursor:
            # Возвращаем результат
            result = await cursor.fetchone()
            if result is not None:
                return result[0]
            else:
                return 0


# функция получения результата пользователя
async def get_result(user_id):
    # Подключаемся к базе данных
    async with aiosqlite.connect(DB_NAME) as db:
        # Получаем записи для всех пользователей
        async with db.execute(
            "SELECT result FROM quiz_state WHERE user_id = (?)", (user_id,)
        ) as cursor:
            # Возвращаем результат
            result = await cursor.fetchone()
            if result is not None:
                return result[0]
            else:
                return 0


# функция получения результата всех пользователей
async def get_statistics():
    # Подключаемся к базе данных
    async with aiosqlite.connect(DB_NAME) as db:
        # Получаем записи для всех пользователей
        async with db.execute("SELECT user_name, result FROM quiz_state") as cursor:
            rows = await cursor.fetchall()
            if rows:
                response = "Статистика пользователей:\n"
                for row in rows:
                    response += f"{row[0]}: {row[1]} правильных ответов\n"
            else:
                response = "Пока нет данных."
        return response
