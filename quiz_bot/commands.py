from aiogram.types import BotCommand, BotCommandScopeDefault
from create_bot import bot


async def set_commands():
    commands = [
        BotCommand(command="start", description="Старт"),
        BotCommand(command="quiz", description="Начать игру"),
        BotCommand(command="result", description="Мой результат"),
        BotCommand(command="statistics", description="Статистика"),
    ]
    await bot.set_my_commands(commands, BotCommandScopeDefault())
