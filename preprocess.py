from PIL import Image
import numpy as np


def scale_and_pad_image(input_image, target_width, target_height, padding_value_color):
    aspect_ratio = input_image.width / input_image.height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        scaled_width = target_width
        scaled_height = int(target_width / aspect_ratio)
    else:
        scaled_width = int(target_height * aspect_ratio)
        scaled_height = target_height

    scaled_image = input_image.resize((scaled_width, scaled_height))

    result_image = Image.new("RGB", (target_width, target_height),
                             color=(padding_value_color, padding_value_color, padding_value_color))

    start_x = (target_width - scaled_width) // 2
    start_y = (target_height - scaled_height) // 2

    result_image.paste(scaled_image, (start_x, start_y))

    return result_image


def preprocess_image(img_path, target_width, target_height, padding_value_color):
    img = Image.open(img_path)
    img = scale_and_pad_image(img, target_width, target_height, padding_value_color)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img / 255.
