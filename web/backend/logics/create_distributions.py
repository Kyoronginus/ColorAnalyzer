import sys
import os
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

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

    base64_img = fig_to_base64(fig)
    plt.close(fig)
    return base64_img

def create_saturation_distribution(hsv_image, filename):
    s = hsv_image[:, :, 1]
    hist = cv2.calcHist([s], [0], None, [256], [0, 256])

    fig = plt.figure(figsize=(8, 4))
    plt.plot(hist, color='blue')
    plt.title("Saturation Distribution")
    plt.xlabel("Saturation Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    base64_img = fig_to_base64(fig)
    plt.close(fig)
    return base64_img

def create_brightness_distribution(image, filename):
    brightness = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([brightness], [0], None, [256], [0, 256])

    fig = plt.figure(figsize=(8, 4))
    plt.plot(hist, color='black')
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    base64_img = fig_to_base64(fig)
    plt.close(fig)
    return base64_img

def create_color_distribution(image, filename):
    reshaped_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, n_init=10)
    kmeans.fit(reshaped_img)

    top_colors = kmeans.cluster_centers_.astype(int)
    percentages = np.unique(kmeans.labels_, return_counts=True)[1] / len(kmeans.labels_)

    fig = plt.figure(figsize=(6, 4))
    plt.bar(range(5), percentages, color=[top_colors[i] / 255 for i in range(5)])
    plt.title("Color Distribution")
    plt.xticks(range(5), [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in top_colors], rotation=45)
    plt.tight_layout()
    
    base64_img = fig_to_base64(fig)
    plt.close(fig)
    return base64_img

def create_3d_histogram(hsv_image, filename):
    s = hsv_image[:, :, 1].flatten()
    v = hsv_image[:, :, 2].flatten()
    
    # Create 2D histogram with fewer bins for readability and performance
    bins = 32
    hist, xedges, yedges = np.histogram2d(s, v, bins=bins, range=[[0, 256], [0, 256]])
    
    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    # Construct arrays with the dimensions for the bars.
    dx = dy = 256 / bins
    dz = hist.ravel()
    
    # Filter out zero-height bars to make the plot cleaner and faster
    mask = dz > 0
    xpos = xpos[mask]
    ypos = ypos[mask]
    dz = dz[mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by height
    cmap = plt.cm.viridis
    max_height = np.max(dz)
    if max_height > 0:
        colors = cmap(dz / max_height)
    else:
        colors = cmap(np.zeros_like(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average', shade=True)
    
    ax.set_xlabel('Saturation')
    ax.set_ylabel('Brightness')
    ax.set_zlabel('Frequency')
    ax.set_title('3D Saturation-Brightness Distribution')
    
    base64_img = fig_to_base64(fig)
    plt.close(fig)
    return base64_img