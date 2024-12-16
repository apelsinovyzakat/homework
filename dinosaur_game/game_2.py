import pyautogui
import cv2
import numpy as np
import time
import math

# Задаем параметры для фильтрации по количеству черных пикселей
min_black_pixels1 = 500   # Минимальное количество черных пикселей для зеленого контура
max_black_pixels1 = 2000  # Максимальное количество черных пикселей для зеленого контура

# Словарь для хранения предыдущих позиций объектов
object_positions = {}

# Переменная для времени предыдущего кадра
previous_time = time.time()

while True:
    # Захват скриншота всего экрана с помощью PyAutoGUI
    screenshot = pyautogui.screenshot()

    # Преобразуем скриншот в формат, поддерживаемый OpenCV
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Обрезка изображения по заданным отступам
    top = 400
    bottom = 400
    left = 50
    right = 1000
    img = img[top:img.shape[0] - bottom, left:img.shape[1] - right]

    # Преобразование в оттенки серого для упрощения порогового преобразования
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Пороговое преобразование для выделения черных областей
    _, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

    # Поиск контуров на бинарном изображении
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Получаем текущее время для вычисления скорости
    current_time = time.time()
    time_elapsed = current_time - previous_time

    # Обновляем время предыдущего кадра
    previous_time = current_time

    # Списки для хранения контуров разных цветов
    green_objects = []
    red_objects = []

    for i, contour in enumerate(contours):
        # Создаем маску для текущего контура и подсчитываем черные пиксели внутри
        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Подсчет черных пикселей внутри маски
        black_pixel_count = cv2.countNonZero(cv2.bitwise_and(binary_img, binary_img, mask=mask))
        
        # Классификация контуров по количеству черных пикселей
        if min_black_pixels1 <= black_pixel_count <= max_black_pixels1:
            green_objects.append(contour)
            
            # Вычисление центра текущего контура
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Вычисление скорости (расстояние, пройденное объектом)
            if i in object_positions:
                # Получаем предыдущую позицию объекта
                prevX, prevY = object_positions[i]
                
                # Вычисляем пройденное расстояние
                distance = math.sqrt((cX - prevX) ** 2 + (cY - prevY) ** 2)
                
                # Вычисляем скорость: расстояние / время
                speed = distance / time_elapsed if time_elapsed > 0 else 0
                
                # Подписываем объект его скоростью
                cv2.putText(img, f"Speed: {speed:.2f} px/s", (cX, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Обновляем позицию объекта для следующего измерения
            object_positions[i] = (cX, cY)
        else:
            red_objects.append(contour)


    # Отображение контуров: зеленые и красные
    cv2.drawContours(img, green_objects, -1, (0, 255, 0), 2)  # Зеленые контуры
    cv2.drawContours(img, red_objects, -1, (0, 0, 255), 2)  # Красные контуры

    # Показ результатов в окне
    cv2.imshow("Detected Black Objects with Speed", img)


    # Условие выхода: Нажатие клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # Добавляем задержку, чтобы снизить нагрузку на процессор (например, 0.5 секунд)
    time.sleep(5)


