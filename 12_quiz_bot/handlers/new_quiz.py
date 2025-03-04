# Импорт необходимых библиотек
from db_handlers.update_handlers import update_quiz_index, update_result
from handlers.get_question import get_question


async def new_quiz(message):
    # получаем id пользователя, отправившего сообщение
    user_id = message.from_user.id

    # сбрасываем значение текущего индекса вопроса квиза в 0
    current_question_index = 0
    await update_quiz_index(user_id, current_question_index)

    # сбрасываем значение количества правильных ответов в 0
    result = 0
    await update_result(user_id, result)

    # запрашиваем новый вопрос для квиза
    await get_question(message, user_id)
