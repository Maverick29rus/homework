from aiogram import types, F, Router
from aiogram.filters import Command, CommandStart
from keyboards import main_kb
from service import (
    get_question,
    new_quiz,
    update_quiz_index,
    get_quiz_state,
    update_result,
    get_statistics,
    update_user_name,
    get_quiz_data,
    get_quiz_count,
)

answer_router = Router()


# Хэндлер на callback вопроса
@answer_router.callback_query()
async def answer(callback: types.CallbackQuery):
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
    current_question_index = await get_quiz_state(user_id, 'question_index')

    # Получаем индекс правильного ответа для текущего вопроса
    correct_index = await get_quiz_data(current_question_index, "correct_option")

    # Получем варианты ответов
    options = (await get_quiz_data(current_question_index, "options")).split(",")

    # Получаем правильный ответ
    correct_answer = options[correct_index]

    # Получение количества правильных ответов для данного пользователя
    result = await get_quiz_state(user_id, 'result')
    
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

    # Получаем количество вопросов
    len_quiz_data = await get_quiz_count()

    # Проверяем достигнут ли конец квиза
    if current_question_index < len_quiz_data:
        # Следующий вопрос
        await get_question(callback.message, user_id)
    else:
        # Уведомление об окончании квиза с выводом результата
        await callback.message.answer("Это был последний вопрос. Квиз завершен!")
        await callback.message.answer(
            f"Поздравляю! Ваш результат: {result} правильных ответов",
            reply_markup=main_kb(),
        )


start_router = Router()


# Хэндлер на команды /start
@start_router.message(CommandStart())
async def cmd_start(message: types.Message):
    # Прикрепляем кнопки к сообщению
    await message.reply_photo(
        photo="https://storage.yandexcloud.net/maverick-quiz-bot-bucket/moto.jpeg",
        caption="Добро пожаловать в квиз!",
    )
    await message.answer("Пожалуйста, введите ваше имя:")


quiz_router = Router()


# Хэндлер на команды /quiz
@quiz_router.message(F.text == "Начать игру")
@quiz_router.message(Command("quiz"))
async def cmd_quiz(message: types.Message):
    # Отправляем новое сообщение без кнопок
    await message.answer("Давайте начнем квиз!")
    # Запускаем новый квиз
    await new_quiz(message)


result_router = Router()


# Хэндлер на команды /result
@result_router.message(F.text == "Мой результат")
@result_router.message(Command("result"))
async def show_result(message: types.Message):
    result = await get_quiz_state(message.from_user.id, 'result')
    await message.answer(f"Ваш результат: {result} правильных ответов\n")


statistics_router = Router()


# Хэндлер на команды /statistics
@statistics_router.message(F.text == "Статистика")
@statistics_router.message(Command("statistics"))
async def show_statistics(message: types.Message):
    statistics = await get_statistics()
    await message.answer(statistics)


name_router = Router()


# Хэндлер на отправку имени
@name_router.message(F.text)
async def process_name(message: types.Message):
    # Обновление имени в базе данных
    await update_user_name(message.from_user.id, message.text)

    # Отправка клавиатуры
    await message.answer(
        f"Спасибо, {message.text}! Ваше имя сохранено.", reply_markup=main_kb()
    )
