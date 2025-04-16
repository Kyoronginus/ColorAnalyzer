import sys
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from psd_tools import PSDImage
import subprocess
import matplotlib.pyplot as plt

from logics.create_distributions import *

def analyze_image_colors(image_path, filename, max_size=500):
    print(f"Analyzing file: {filename}")  # Debug: Which file is being analyzed
    if filename.endswith('.psd'):
        psd = PSDImage.open(image_path)
        print("Opened PSD file successfully.")  # Debug
        
        # Save the composite image as a PNG
        composite_image_path = os.path.join('static/uploads', 'composite_' + filename.replace('.psd', '.png'))
        psd.composite().save(composite_image_path)
        print(f"Saved composite image at: {composite_image_path}")  # Debug

        # Set image_path to the PNG file for further processing
        image_path = composite_image_path

    # Open the image
    image = Image.open(image_path)
    print(f"Opened image file {image_path} successfully.")  # Debug

    # Convert image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image for analysis
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)

    image = image.resize((new_width, new_height))
    img_data = np.array(image)

    # Simplify the colors
    hsv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    h = (h // 3) * 3
    s = (s // 8) * 8
    v = (v // 8) * 8

    simplified_hsv_image = np.stack([h, s, v], axis=-1)
    simplified_rgb_image = cv2.cvtColor(simplified_hsv_image, cv2.COLOR_HSV2RGB)

    # Save the simplified image (use a supported format, such as PNG)
    simplified_image_path = os.path.join('static/results', 'simplified_' + filename.replace('.psd', '.png'))
    Image.fromarray(simplified_rgb_image).save(simplified_image_path)
    print(f"Saved simplified image: {simplified_image_path}")  # Debug

    # Call distribution functions
    color_distribution_path = create_color_distribution(simplified_rgb_image, filename.replace('.psd', ''))
    brightness_distribution_path = create_brightness_distribution(img_data, filename.replace('.psd', ''))
    saturation_distribution_path = create_saturation_distribution(hsv_image, filename.replace('.psd', ''))
    hue_distribution_path = create_hue_distribution(hsv_image, filename.replace('.psd', ''))

    print("Generated distribution paths.")  # Debug

    return simplified_image_path,hue_distribution_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path