# Импорт необходимых библиотек
import asyncio
from aiogram import types, Router
from db_handlers.get_hendlers import get_result, get_quiz_index
from db_handlers.update_handlers import update_quiz_index, update_result
from handlers.get_question import get_question
from data.quiz_data import quiz_data
from keyboards.main_kb import main_kb

answer_router = Router()


# Хэндлер на callback вопроса
@answer_router.callback_query()
async def right_answer(callback: types.CallbackQuery):
    # редактируем текущее сообщение с целью убрать кнопки (reply_markup=None)
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None,
    )

    # Получаем ответ пользователя
    usr_answer = callback.data

    # Получаем ID пользователя
    user_id = callback.from_user.id

    # Отправляем в чат сообщение, о выбранном ответе
    await callback.message.answer(f"Ваш ответ: {usr_answer}")

    # Получение текущего вопроса для данного пользователя
    current_question_index = await get_quiz_index(user_id)

    # Получаем индекс правильного ответа для текущего вопроса
    correct_index = quiz_data[current_question_index]["correct_option"]

    # Получаем правильный ответ
    correct_answer = quiz_data[current_question_index]["options"][correct_index]

    # Получение количества правильных ответов для данного пользователя
    result = await get_result(user_id)

    # Проверка правильности ответа
    if usr_answer == correct_answer:
        # Отправляем в чат сообщение о том что ответ верный
        await callback.message.answer("Верно!")
        # Обновление числа правильных ответов
        result += 1
        await update_result(user_id, result)
    else:
        # Отправляем в чат сообщение об ошибке с указанием верного ответа
        await callback.message.answer(
            f"Неправильно. Правильный ответ: {correct_answer}"
        )

    # Обновление номера текущего вопроса в базе данных
    current_question_index += 1
    await update_quiz_index(user_id, current_question_index)

    # Проверяем достигнут ли конец квиза
    if current_question_index < len(quiz_data):
        # Задержка 1 секунда чтобы пользователь успел прочитать правильный ответ
        await asyncio.sleep(1)
        # Следующий вопрос
        await get_question(callback.message, user_id)
    else:
        # Уведомление об окончании квиза с выводом результата
        await callback.message.answer("Это был последний вопрос. Квиз завершен!")
        await callback.message.answer(
            f"Поздравляю! Ваш результат: {result} правильных ответов",
            reply_markup=main_kb(),
        )
