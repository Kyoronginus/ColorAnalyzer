import sys
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
import subprocess
import io
import base64

from logics.create_distributions import *

def image_to_base64(image, format='PNG'):
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"

def analyze_image_colors(image_file, filename, max_size=500):
    print(f"Analyzing file: {filename}")
    
    image = Image.open(image_file)
    print(f"Opened image file successfully.")

    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)

    image = image.resize((new_width, new_height))
    img_data = np.array(image)

    hsv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    h = (h // 3) * 3
    s = (s // 8) * 8
    v = (v // 8) * 8

    simplified_hsv_image = np.stack([h, s, v], axis=-1)
    simplified_rgb_image = cv2.cvtColor(simplified_hsv_image, cv2.COLOR_HSV2RGB)

    simplified_pil = Image.fromarray(simplified_rgb_image)
    simplified_image_base64 = image_to_base64(simplified_pil)


    clean_filename = filename.rsplit('.', 1)[0]
    
    color_distribution_base64 = create_color_distribution(simplified_rgb_image, clean_filename)
    brightness_distribution_base64 = create_brightness_distribution(img_data, clean_filename)
    saturation_distribution_base64 = create_saturation_distribution(hsv_image, clean_filename)
    hue_distribution_base64 = create_hue_distribution(hsv_image, clean_filename)
    histogram_3d_base64 = create_3d_histogram(hsv_image, clean_filename)

    return simplified_image_base64, hue_distribution_base64, color_distribution_base64, brightness_distribution_base64, saturation_distribution_base64, histogram_3d_base64