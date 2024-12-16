import pyautogui
import cv2
import numpy as np
import time
import math


# Задаем параметры для фильтрации по количеству черных пикселей
min_black_pixels1 = 200   # Минимальное количество черных пикселей для препятствия
max_black_pixels1 = 1300  # Максимальное количество черных пикселей для препятствия

# Минимальная площадь контура для фильтрации
min_contour_area = 150  # Минимальная площадь контура для фильтрации

min_obstacle_distance = 10

# Координаты области экрана, где находится динозавр (определите заранее)
dino_x, dino_y = 145, 93           # Примерные координаты динозавра
jump_threshold = 170  # Пороговое расстояние для прыжка


while True:
    # Захват скриншота всего экрана с помощью PyAutoGUI
    screenshot = pyautogui.screenshot()

    # Преобразуем скриншот в формат, поддерживаемый OpenCV
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Обрезка изображения по заданным отступам (укажите нужные отступы)
    top = 400
    bottom = 400
    left = 50
    right = 1000
    img = img[top:img.shape[0] - bottom, left:img.shape[1] - right]

    # Преобразование в оттенки серого для упрощения порогового преобразования
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Пороговое преобразование для выделения черных областей (препятствий)
    _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

    # Поиск контуров на бинарном изображении
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отображение координат динозавра для отладки
    cv2.circle(img, (dino_x, dino_y), radius=10, color=(0, 0, 255), thickness=-1)

    closest_distance = float('inf')
    closest_obstacle = None

    for contour in contours:
        # Фильтрация контуров по площади
        if cv2.contourArea(contour) < min_contour_area:
            continue

        # Создаем маску для текущего контура и подсчитываем черные пиксели внутри
        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Подсчет черных пикселей внутри маски
        black_pixel_count = cv2.countNonZero(cv2.bitwise_and(binary_img, binary_img, mask=mask))

        # Фильтрация контуров по количеству черных пикселей для выделения препятствий
        if min_black_pixels1 <= black_pixel_count <= max_black_pixels1:
            # Вычисление центра препятствия
            M = cv2.moments(contour)
            if M["m00"] != 0:
                obstacle_x = int(M["m10"] / M["m00"])
                obstacle_y = int(M["m01"] / M["m00"])

                # Проверка расстояния между динозавром и найденным контуром
                distance_to_dino = math.sqrt((obstacle_x - dino_x) ** 2 + (obstacle_y - dino_y) ** 2)

                # Если препятствие слишком близко к динозавру, пропускаем его
                if distance_to_dino < min_obstacle_distance:
                    continue

                # Рисуем окружность вокруг найденного препятствия
                cv2.circle(img, (obstacle_x, obstacle_y), radius=5, color=(255, 0, 0), thickness=-1)

                # Проверка расстояния между динозавром и препятствием
                distance = math.sqrt((obstacle_x - dino_x) ** 2 + (obstacle_y - dino_y) ** 2)

                # Печать расстояния для отладки
                print(f"Distance: {distance} Jump Threshold: {jump_threshold}")

                # Сохраняем ближайшее препятствие
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obstacle = (obstacle_x, obstacle_y)

    # Если ближайшее препятствие ближе, чем порог, то имитируем прыжок
    if closest_obstacle and closest_distance < jump_threshold:
        pyautogui.press("space")
        print("Jump triggered!")


    # Условие выхода: Нажатие клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Добавляем задержку для снижения нагрузки на процессор
    time.sleep(0.2)

# Освобождаем ресурсы
cv2.destroyAllWindows()
