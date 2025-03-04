import asyncio
from database import pool, execute_update_query, execute_select_query
from keyboards import generate_options_keyboard


async def get_quiz_data(question_id, query):
    get_query = f"""
        DECLARE $question_id AS Uint32;

        SELECT {query}
        FROM `quiz_data`
        WHERE question_id == $question_id;
    """
    results = execute_select_query(pool, get_query, question_id=question_id)

    return results[0][query]


async def get_quiz_count():
    get_count = f"""SELECT COUNT(*) FROM `quiz_data`;"""
    results = execute_select_query(pool, get_count)

    return results[0][0]


async def get_quiz_state(user_id, query):
    get_query = f"""
        DECLARE $user_id AS Uint64;

        SELECT {query}
        FROM `quiz_state`
        WHERE user_id == $user_id;
    """
    results = execute_select_query(pool, get_query, user_id=user_id)

    if len(results) == 0:
        return 0
    if results[0][query] is None:
        return 0
    return results[0][query]


async def get_question(message, user_id):
    # Запрашиваем из базы текущий индекс для вопроса
    current_question_index = await get_quiz_state(user_id, 'question_index')
    # Получаем список вариантов ответа для текущего вопроса
    opts = (await get_quiz_data(current_question_index, "options")).split(",")
    # Получаем ID картинки
    photo_id = await get_quiz_data(current_question_index, "image")
    # Функция генерации кнопок для текущего вопроса квиза
    kb = generate_options_keyboard(opts)
    # Получаем текущий вопрос
    question = await get_quiz_data(current_question_index, "question")
    await asyncio.sleep(1)
    # Отправляем в чат сообщение с вопросом, прикрепляем сгенерированные кнопки и картинку
    await message.answer_photo(
        photo=photo_id,
        reply_markup=kb,
        caption=f"{question}",
    )


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


# функция получения результата всех пользователей
async def get_statistics():
    get_all_results = f"""
        SELECT user_name, result
        FROM `quiz_state`
    """

    results = execute_select_query(pool, get_all_results)

    if len(results) == 0:
        response = "Пока нет данных."
    response = "Статистика пользователей:\n"
    for item in results:
        response += f"{item.user_name}: {item.result} правильных ответов\n"
    return response


async def update_quiz_index(user_id, question_index):
    set_quiz_state = f"""
        DECLARE $user_id AS Uint64;
        DECLARE $question_index AS Uint64;

        UPSERT INTO `quiz_state` (`user_id`, `question_index`)
        VALUES ($user_id, $question_index);
    """

    execute_update_query(
        pool,
        set_quiz_state,
        user_id=user_id,
        question_index=question_index,
    )


async def update_result(user_id, result):
    set_result = f"""
        DECLARE $user_id AS Uint64;
        DECLARE $result AS Uint64;

        UPSERT INTO `quiz_state` (`user_id`, `result`)
        VALUES ($user_id, $result);
    """

    execute_update_query(
        pool,
        set_result,
        user_id=user_id,
        result=result,
    )


async def update_user_name(user_id, user_name):
    set_user_name = f"""
        DECLARE $user_id AS Uint64;
        DECLARE $user_name AS Utf8;

        UPSERT INTO `quiz_state` (`user_id`, `user_name`)
        VALUES ($user_id, $user_name);
    """

    execute_update_query(
        pool,
        set_user_name,
        user_id=user_id,
        user_name=user_name,
    )
