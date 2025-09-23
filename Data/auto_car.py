
# Инициализирует новый класс Auto_Car с заданными атрибутами
class Auto_Car:
    Name_class = 'Беспилотный автомобиль'

    def __init__(self, version, SAE_class, sensors):
        self.version = version # Версия прошивки
        self.SAE_class = SAE_class # Уровень автономности по SAE
        self.sensors = sensors # Количество датчиков


        # Метод выбора дистанции.
    def distance_selection(self):
        # Команды для выбора дистанции
        print("Выбираю дистанцию.")

        # Метод перестроения.
    def lane_change(self):
        # Команды для перестроения
        print("Перестраиваюсь.")

        # Метод контроля полосы.
    def lane_control(self):
        # Команды для контроля полосы
        print("Контролирую положение в полосе.")

        # Метод парковки.
    def parking(self):
        # Команды для парковки
        print("Паркуюсь.")

        # Метод контроля скорости.
    def speed_control(self):
        # Команды для контроля скорости
        print("Контролирую скорость.")
