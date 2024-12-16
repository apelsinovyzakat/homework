import cv2
import numpy as np
from PIL import Image, ImageEnhance

file_path = "C:\\Users\\timur\\Downloads\\55bc0bed-83df-49ab-9e5d-3123be93eafc (1).png"
image_cv = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def flip_image(image, mode):
    return cv2.flip(image, mode)

def scale_image(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

def crop_center(image):
    (h, w) = image.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return image[start_y:start_y + min_dim, start_x:start_x + min_dim]

def translate_image(image, x_shift, y_shift):
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated = cv2.warpAffine(image, M, (w, h))
    return translated

def blur_image(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_to_black_and_white(image):
    grayscale = convert_to_grayscale(image)
    _, black_and_white = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    return black_and_white

def apply_color_tint(image, channel):
    channels = list(cv2.split(image))
    for i in range(len(channels)):
        if i != channel:
            channels[i] = np.zeros_like(channels[i])
    return cv2.merge(channels)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def adjust_saturation(image, factor):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(image_pil)
    image_enhanced = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(image_enhanced), cv2.COLOR_RGB2BGR)

def invert_colors(image):
    return cv2.bitwise_not(image)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

rotated_clockwise = rotate_image(image_cv, -90)
rotated_counterclockwise = rotate_image(image_cv, 90)
horizontally_flipped = flip_image(image_cv, 1)
vertically_flipped = flip_image(image_cv, 0)
rotated_angle = rotate_image(image_cv, 75)
scaled_up = scale_image(image_cv, 2)
scaled_down = scale_image(image_cv, 0.5)
cropped = crop_center(image_cv)
translated = translate_image(image_cv, 50, 50)
blurred = blur_image(image_cv)
grayscale = convert_to_grayscale(image_cv)
black_and_white = convert_to_black_and_white(image_cv)
red_tint = apply_color_tint(image_cv, 2)
green_tint = apply_color_tint(image_cv, 1)
blue_tint = apply_color_tint(image_cv, 0)
sepia = apply_sepia(image_cv)
saturated = adjust_saturation(image_cv, 1.5)
desaturated = adjust_saturation(image_cv, 0.5)
inverted = invert_colors(image_cv)
hsv_image = convert_to_hsv(image_cv)

cv2.imwrite("images/rotated_clockwise.jpg", rotated_clockwise)
cv2.imwrite("images/rotated_counterclockwise.jpg", rotated_counterclockwise)
cv2.imwrite("images/horizontally_flipped.jpg", horizontally_flipped)
cv2.imwrite("images/vertically_flipped.jpg", vertically_flipped)
cv2.imwrite("images/rotated_angle.jpg", rotated_angle)
cv2.imwrite("images/scaled_up.jpg", scaled_up)
cv2.imwrite("images/scaled_down.jpg", scaled_down)
cv2.imwrite("images/cropped.jpg", cropped)
cv2.imwrite("images/translated.jpg", translated)
cv2.imwrite("images/blurred.jpg", blurred)
cv2.imwrite("images/grayscale.jpg", grayscale)
cv2.imwrite("images/black_and_white.jpg", black_and_white)
cv2.imwrite("images/red_tint.jpg", red_tint)
cv2.imwrite("images/green_tint.jpg", green_tint)
cv2.imwrite("images/blue_tint.jpg", blue_tint)
cv2.imwrite("images/sepia.jpg", sepia)
cv2.imwrite("images/saturated.jpg", saturated)
cv2.imwrite("images/desaturated.jpg", desaturated)
cv2.imwrite("images/inverted.jpg", inverted)
cv2.imwrite("images/hsv_image.jpg", hsv_image)
