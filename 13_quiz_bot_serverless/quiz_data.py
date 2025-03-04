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

# Переводим список словарей в список кортежей
question_id = 0
result = []
for q in quiz_data:
    result.append(
        (
            question_id,
            q["question"],
            ",".join(q["options"]),
            q["correct_option"],
            q["image"],
        )
    )
    question_id += 1

# Печатаем получившийся список вопросов
print(result)
