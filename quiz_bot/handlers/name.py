# Импорт необходимых библиотек
from aiogram import types, Router
from keyboards.main_kb import main_kb
from db_handlers.update_handlers import update_name

name_router = Router()

# Хэндлер на отправку имени
@name_router.message(lambda message: message.text.isalpha())
async def process_name(message: types.Message):
    # Обновление имени в базе данных
    await update_name(message.from_user.id, message.text)

    # Отправка клавиатуры
    await message.answer(
        f"Спасибо, {message.text}! Ваше имя сохранено.", reply_markup=main_kb()
    )
