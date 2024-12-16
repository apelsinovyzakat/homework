
import pyautogui
import cv2
import numpy as np

# Захватываем скриншот экрана с помощью PyAutoGUI
screenshot = pyautogui.screenshot()

# Преобразуем скриншот в формат, поддерживаемый OpenCV
img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Определяем отступы для обрезки (например, 50 пикселей с каждой стороны)
top = 250
bottom = 520
left = 980
right = 50

# Обрезаем изображение
img = img[top:img.shape[0] - bottom, left:img.shape[1] - right]
# Сохраняем изображение в файл
cv2.imwrite("screenshot.png", img)

print("Скриншот сохранен как screenshot.png")
# Преобразование в оттенки серого
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применение порогового преобразования для выделения черных объектов
# Задаем порог для темных (черных) областей
_, binary_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("binary_img.png", binary_img)
# Поиск контуров на бинарном изображении
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фильтрация контуров по количеству черных пикселей внутри них
filtered_objects = []
otrer_objects = []
min_black_pixels1 = 500   # Минимальное количество черных пикселей
max_black_pixels1 = 2000  # Максимальное количество черных пикселей

min_black_pixels2 = 500   # Минимальное количество черных пикселей
max_black_pixels2 = 2000  # Максимальное количество черных пикселей

for contour in contours:
    # Создаем маску для текущего контура
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Подсчитываем количество черных пикселей внутри контура
    black_pixel_count = cv2.countNonZero(cv2.bitwise_and(binary_img, binary_img, mask=mask))
    
    # Проверяем, попадает ли количество черных пикселей в заданный диапазон
    if min_black_pixels1 <= black_pixel_count <= max_black_pixels1:
        filtered_objects.append(contour)
        print(black_pixel_count, "green")

        # Вычисляем центр контура для координат
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Координаты объекта (зелёного): ({cX}, {cY})")
            # Подписываем координаты на изображении
            cv2.putText(img, f"({cX}, {cY})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    else:
        otrer_objects.append(contour)
        print(black_pixel_count, "red")

        # Вычисляем центр контура для координат
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(f"Координаты объекта (красного): ({cX}, {cY})")
            # Подписываем координаты на изображении
            cv2.putText(img, f"({cX}, {cY})", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Отображаем результаты, обрисовав найденные объекты на исходном изображении
cv2.drawContours(img, filtered_objects, -1, (0, 255, 0), 2)  # Рисуем зеленые контуры
cv2.drawContours(img, otrer_objects, -1, (0, 0, 255), 2)  # Рисуем зеленые контуры

# Сохраняем результат
cv2.imwrite("result_with_black_objects.png", img)
cv2.imshow("Detected Black Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
