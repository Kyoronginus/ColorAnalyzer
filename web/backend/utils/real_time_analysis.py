from flask import Flask, render_template,Response, request, redirect, url_for
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import os
import matplotlib
import time
from mss import mss
from pygetwindow import getWindowsWithTitle
matplotlib.use('Agg')  # GUIバックエンドを使用しない設定

def analyze_and_stream():
    while True:
        region = get_clip_studio_paint_window()
        if region:
            img_data = capture_screen(region)
            img_data_resized = cv2.resize(img_data, (1000, 500))

            # Analyze real-time capture and create graphs
            color_distribution_img = create_color_distribution_image(img_data)
            brightness_distribution_img = create_brightness_distribution_image(img_data)

            combined_image = overlay_analysis_on_frame(img_data, color_distribution_img, brightness_distribution_img)

            ret, jpeg = cv2.imencode('.jpg', combined_image)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            time.sleep(1)
        else:
            yield None

# Other utility functions (create_color_distribution_image, etc.)
def create_color_distribution_image(image):
    reshaped_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(reshaped_img)

    top_colors = kmeans.cluster_centers_.astype(int)
    percentages = np.unique(kmeans.labels_, return_counts=True)[1] / len(kmeans.labels_)

    # Create the color distribution plot as an image
    plt.figure(figsize=(3, 3))
    plt.bar(range(5), percentages, color=[top_colors[i] / 255 for i in range(5)])
    plt.tight_layout()

    # Convert the plot to a NumPy array (image)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the PNG buffer to a NumPy array
    color_dist_img = np.array(Image.open(buf))

    return color_dist_img

def create_brightness_distribution_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Create the brightness distribution plot as an image
    plt.figure(figsize=(3, 3))
    plt.plot(hist, color='black')
    plt.tight_layout()

    # Convert the plot to a NumPy array (image)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convert the PNG buffer to a NumPy array
    brightness_dist_img = np.array(Image.open(buf))

    return brightness_dist_img

# Helper function to overlay the analysis images on the frame
def overlay_analysis_on_frame(frame, color_distribution_img, brightness_distribution_img):
    # Maintain aspect ratio for both graphs
    color_dist_height, color_dist_width = color_distribution_img.shape[:2]
    brightness_dist_height, brightness_dist_width = brightness_distribution_img.shape[:2]

    # Rescale to maintain original aspect ratio (fit into a 200px height region)
    aspect_ratio_color = color_dist_width / color_dist_height
    aspect_ratio_brightness = brightness_dist_width / brightness_dist_height

    new_color_width = int(200 * aspect_ratio_color)
    new_brightness_width = int(200 * aspect_ratio_brightness)

    color_dist_resized = cv2.resize(color_distribution_img, (new_color_width, 200))
    brightness_dist_resized = cv2.resize(brightness_distribution_img, (new_brightness_width, 200))

    # Convert RGBA to RGB to remove alpha channel if necessary
    if color_dist_resized.shape[2] == 4:
        color_dist_resized = cv2.cvtColor(color_dist_resized, cv2.COLOR_RGBA2RGB)
    if brightness_dist_resized.shape[2] == 4:
        brightness_dist_resized = cv2.cvtColor(brightness_dist_resized, cv2.COLOR_RGBA2RGB)

    # Place the color distribution on the top-left and brightness on top-right of the frame
    frame[0:200, 0:new_color_width] = color_dist_resized
    frame[0:200, -new_brightness_width:] = brightness_dist_resized

    # Convert the frame back to RGB for proper display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame

def get_clip_studio_paint_window():
    windows = getWindowsWithTitle('Clip Studio Paint')
    if windows:
        window = windows[0]
        return {'top': window.top, 'left': window.left, 'width': window.width, 'height': window.height}
    else:
        return None
    

# Capture screen from a specific region (Clip Studio Paint window)
def capture_screen(region):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
        return np.array(img)
