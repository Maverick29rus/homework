# database.py
import os

# Импортируем библиотеку для работы с базой данных (БД)
import ydb

# Получаем переменные подключения к БД из окружения
YDB_ENDPOINT = os.getenv("YDB_ENDPOINT")
YDB_DATABASE = os.getenv("YDB_DATABASE")


def get_ydb_pool(ydb_endpoint, ydb_database, timeout=30):
    # Задаем конфигурацию драйвера, параметры подключения
    ydb_driver_config = ydb.DriverConfig(
        ydb_endpoint,
        ydb_database,
        credentials=ydb.credentials_from_env_variables(),
        root_certificates=ydb.load_ydb_root_certificate(),
    )

    # Инициализируем драйвер
    ydb_driver = ydb.Driver(ydb_driver_config)
    # Ожидаем, пока драйвер станет активным для запросов.
    ydb_driver.wait(fail_fast=True, timeout=timeout)

    # Возвращаем пул сессий
    return ydb.SessionPool(ydb_driver)


# Зададим настройки базы данных
pool = get_ydb_pool(YDB_ENDPOINT, YDB_DATABASE)


# Функция форматирования входных аргументов
def _format_kwargs(kwargs):
    return {"${}".format(key): value for key, value in kwargs.items()}


# Заготовки из документации
# https://ydb.tech/en/docs/reference/ydb-sdk/example/python/#param-prepared-queries
def execute_update_query(pool, query, **kwargs):
    def callee(session):
        # Наши подготовленные, параметризованные запросы
        prepared_query = session.prepare(query)
        session.transaction(ydb.SerializableReadWrite()).execute(
            prepared_query, _format_kwargs(kwargs), commit_tx=True
        )

    # Реализация стратегии повторных запросов
    return pool.retry_operation_sync(callee)


# Заготовки из документации
# https://ydb.tech/en/docs/reference/ydb-sdk/example/python/#param-prepared-queries
def execute_select_query(pool, query, **kwargs):
    def callee(session):
        # Наши подготовленные, параметризованные запросы
        prepared_query = session.prepare(query)
        result_sets = session.transaction(ydb.SerializableReadWrite()).execute(
            prepared_query, _format_kwargs(kwargs), commit_tx=True
        )
        return result_sets[0].rows

    # Реализация стратегии повторных запросов
    return pool.retry_operation_sync(callee)


# Структура квиза
quiz_data = [
    {
        "question": "Вот двухколёсный BMW серии GS: большой комфортабельный \
            эндуро. Какое прозвище он получил в народе?",
        "options": ["Джиксер", "Жесть", "Гусь"],
        "correct_option": 2,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/0.webp",
    },
    {
        "question": "Это — по-настоящему большая Honda Gold Wing для далёких \
            асфальтовых круизов. Какая кличка приклеилась к мотоциклу?",
        "options": ["Винга", "Голда", "Линкор"],
        "correct_option": 1,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/1.webp",
    },
    {
        "question": "Ещё одна Хонда — стильный турер PC 800 Pacific Coast родом из девяностых. Вспомните кличку мотоцикла?",
        "options": ["Рысь", "Восьмисотый", "Пуфик"],
        "correct_option": 2,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/2.webp",
    },
    {
        "question": "Спортивный Suzuki GSX-R1000 тоже не остался без клички. Какой?",
        "options": ["Джиксер", "Госарь", "Вдова"],
        "correct_option": 0,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/3.webp",
    },
    {
        "question": "«Спорттурист» Kawasaki ZZ-R400 остаётся отличным вариантом для начинающего байкера: он понятный, комфортный и не слишком буйный. А ещё у него есть прозвище. Какое?",
        "options": ["Зизер", "Зета", "Зюзя"],
        "correct_option": 0,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/4.webp",
    },
    {
        "question": "Ещё один Кавасаки — с заводским индексом и неофициальной кличкой. Знаете, как прозвали модель ER-6?",
        "options": ["Эрка", "Ёрш", "Йорик"],
        "correct_option": 1,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/5.webp",
    },
    {
        "question": "А это — дорожная Yamaha FZX 750, которая за пределами Японии носила имя Fazer. А какой никнейм модель получила у нас?",
        "options": ["Батя", "Пианино", "Физик"],
        "correct_option": 2,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/6.webp",
    },
    {
        "question": "Как насчёт прозвища для Ducati Multistrada 1200? У этого универсального «итальянца» оно тоже имеется",
        "options": ["Мультик", "Дрозд", "Страдалец"],
        "correct_option": 0,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/7.webp",
    },
    {
        "question": "BMW S 1000 RR с рядной «четвёркой» — лютый спортбайк: лёгкий, быстрый, технически продвинутый. Под стать характеристикам должно быть и русское прозвище, верно?",
        "options": ["Сатана", "Косой", "Сыр"],
        "correct_option": 2,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/8.webp",
    },
    {
        "question": "А вот лёгкая дорожная Honda VTR250. Уже догадались, как её кличут байкеры?",
        "options": ["Витя", "Ватрушка", "Варя"],
        "correct_option": 1,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/9.webp",
    },
    {
        "question": "Немолодой чоппер Yamaha Virago стильно выглядит и приятно едет. А ещё носит эффектную кличку. Какую?",
        "options": ["Выдра", "Ведьма", "Вираж"],
        "correct_option": 1,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/10.webp",
    },
    {
        "question": "Напоследок покажем Хонду — стильный круизер 600 VLX линейки Steed. И нам нужно верное прозвище модели!",
        "options": ["Гантеля", "Пончик", "Стыд"],
        "correct_option": 2,
        "image": "https://storage.yandexcloud.net/maverick-quiz-bot-bucket/11.webp",
    },
]
