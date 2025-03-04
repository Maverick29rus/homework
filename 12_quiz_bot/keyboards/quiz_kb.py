# Импорт необходимых библиотек
from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder


def generate_options_keyboard(answer_options):
    # Создаем сборщика клавиатур типа Inline
    builder = InlineKeyboardBuilder()

    # В цикле создаем 4 Inline кнопки, а точнее Callback-кнопки
    for option in answer_options:
        builder.add(
            types.InlineKeyboardButton(
                # Текст на кнопках соответствует вариантам ответов
                text=option,
                # Присваиваем данные для колбэк запроса.
                callback_data=option,
            )
        )

    # Выводим по одной кнопке в столбик
    builder.adjust(1)
    return builder.as_markup()
