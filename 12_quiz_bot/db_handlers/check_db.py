# Импорт необходимых библиотек
import aiosqlite
import asyncio

# Зададим имя базы данных
DB_NAME = "quiz_bot/quiz.db"


# вывод содержимого базы данных в терминал
async def check_db():
    # Подключаемся к базе данных
    async with aiosqlite.connect(DB_NAME) as db:
        # Выполняем SQL-запрос
        async with db.execute("SELECT * FROM quiz_state") as cursor:
            # Извлекаем все строки из результата запроса
            rows = await cursor.fetchall()
            # Обрабатываем и выводим данные
            for row in rows:
                print(
                    f"user_id: {row[0]}, question_index: {row[1]}, user_name: {row[2]}, correct_answers: {row[3]}"
                )


asyncio.run(check_db())
