# Импорт необходимых библиотек
# import os
# from aiogram.types import FSInputFile
# from create_bot import images_dir
from db_handlers.get_hendlers import get_quiz_index
from data.quiz_data import quiz_data
from keyboards.quiz_kb import generate_options_keyboard


async def get_question(message, user_id):
    # Запрашиваем из базы текущий индекс для вопроса
    current_question_index = await get_quiz_index(user_id)

    # Получаем список вариантов ответа для текущего вопроса
    opts = quiz_data[current_question_index]["options"]

    # Получаем ID картинки
    photo_id = quiz_data[current_question_index]["image"]
    # альтернативный вариант получения картинки
    # photo_file = FSInputFile(
    #    path=os.path.join(images_dir, str(current_question_index) + ".webp")
    # )

    # Функция генерации кнопок для текущего вопроса квиза
    kb = generate_options_keyboard(opts)

    # Получаем текущий вопрос
    question = quiz_data[current_question_index]["question"]

    # Отправляем в чат сообщение с вопросом, прикрепляем сгенерированные кнопки и картинку
    await message.answer_photo(
        photo=photo_id,
        reply_markup=kb,
        caption=f"{question}",
    )
