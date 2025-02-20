from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton


def generate_options_keyboard(answer_options):
    # Создаем сборщика клавиатур типа Inline
    builder = InlineKeyboardBuilder()

    # В цикле создаем 4 Inline кнопки, а точнее Callback-кнопки
    for option in answer_options:
        builder.add(
            InlineKeyboardButton(
                # Текст на кнопках соответствует вариантам ответов
                text=option,
                # Присваиваем данные для колбэк запроса.
                callback_data=option,
            )
        )

    # Выводим по одной кнопке в столбик
    builder.adjust(1)
    return builder.as_markup()


def main_kb():
    # Создание списка кнопок
    kb_list = [
        [KeyboardButton(text="Начать игру"), KeyboardButton(text="Мой результат")],
        [KeyboardButton(text="Статистика")],
    ]
    # Создаем сборщика клавиатур типа Reply
    keyboard = ReplyKeyboardMarkup(
        keyboard=kb_list,
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    return keyboard
