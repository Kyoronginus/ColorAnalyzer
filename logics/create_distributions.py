import sys
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from watchdog.observers import Observer
import matplotlib.pyplot as plt

def create_hue_distribution(hsv_image, filename):
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    non_grayscale_mask = s > 0
    h_non_grayscale = h[non_grayscale_mask]
    
    hue_hist, bin_edges = np.histogram(h_non_grayscale, bins=180, range=(0, 180))
    bin_edges_shifted = (bin_edges) % 180

    theta = np.linspace(0, 2 * np.pi, 180)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    ax.bar(theta, hue_hist, width=2 * np.pi / 180, color=plt.cm.hsv(bin_edges_shifted[:-1] / 180.0), bottom=0.0)
    ax.set_yticks([])
    ax.set_title('Hue Distribution Wheel', va='bottom')

    hue_distribution_path = os.path.join('static/results', 'hue_wheel_' + filename + '.png')
    plt.savefig(hue_distribution_path)
    plt.close()

    return hue_distribution_path

def create_saturation_distribution(hsv_image, filename):
    s = hsv_image[:, :, 1]
    hist = cv2.calcHist([s], [0], None, [256], [0, 256])

    saturation_distribution_path = os.path.join('static/results', 'saturation_distribution_' + filename + '.png')
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='blue')
    plt.title("Saturation Distribution")
    plt.xlabel("Saturation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(saturation_distribution_path)
    plt.close()

    return saturation_distribution_path

def create_brightness_distribution(image, filename):
    brightness = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([brightness], [0], None, [256], [0, 256])

    brightness_curve_path = os.path.join('static/results', 'brightness_curve_' + filename + '.png')
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='black')
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(brightness_curve_path)
    plt.close()

    return brightness_curve_path

def create_color_distribution(image, filename):
    reshaped_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(reshaped_img)

    top_colors = kmeans.cluster_centers_.astype(int)
    percentages = np.unique(kmeans.labels_, return_counts=True)[1] / len(kmeans.labels_)

    color_distribution_path = os.path.join('static/results', 'color_distribution_' + filename + '.png')
    plt.figure(figsize=(6, 4))
    plt.bar(range(5), percentages, color=[top_colors[i] / 255 for i in range(5)])
    plt.title("Color Distribution")
    plt.xticks(range(5), [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in top_colors], rotation=45)
    plt.tight_layout()
    plt.savefig(color_distribution_path)
    plt.close()

    return color_distribution_path