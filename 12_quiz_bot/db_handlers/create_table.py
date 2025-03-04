# Импорт необходимых библиотек
import aiosqlite

# Зададим имя базы данных
DB_NAME = "quiz_bot/quiz.db"

# функция создания базы данных
async def create_table():
    # Создаем соединение с базой данных
    # (если она не существует, то она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Выполняем SQL-запрос к базе данных
        await db.execute(
            "CREATE TABLE IF NOT EXISTS quiz_state \
            (user_id INTEGER PRIMARY KEY, \
             question_index INTEGER, \
             user_name TEXT DEFAULT 'Аноним', \
             result INTEGER DEFAULT 0)"
        )
        # Сохраняем изменения
        await db.commit()
