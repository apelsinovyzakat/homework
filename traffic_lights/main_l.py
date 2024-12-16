import cv2
import numpy as np
import os

def traffic_light_detector(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([8, 90, 200])
    upper_red1 = np.array([13, 255, 255])
    lower_red2 = np.array([170, 160, 160])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([17, 165, 165])
    upper_yellow = np.array([25, 255, 255])
    lower_green = np.array([75, 90, 99])
    upper_green = np.array([90, 255, 120])

    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((10, 10), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for contour in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    for contour in contours_green:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow(f'Traffic Light Detection - {image_path}', image)
    cv2.waitKey(0)

for x in os.listdir("imgs_lights"):
    if x.endswith(".jpg"):
        traffic_light_detector(os.path.join("imgs_lights", x))

cv2.destroyAllWindows()
