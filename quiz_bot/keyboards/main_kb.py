# Импорт необходимых библиотек
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


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
