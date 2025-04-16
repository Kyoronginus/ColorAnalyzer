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
from flask import Flask, render_template, Response, request, redirect, url_for
from utils.image_analysis import analyze_image_colors
from utils.real_time_analysis import analyze_and_stream
matplotlib.use('Agg')  # GUIバックエンドを使用しない設定

app = Flask(__name__)

# Configure upload and result directories
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return redirect(url_for('home'))

    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    image_file.save(filepath)
    image = Image.open(filepath)

    # Analyze image and save results
    simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path, hue_distribution_path = analyze_image_colors(image, filename)

    # Convert paths for URL compatibility
    original_image_url = filepath.replace(os.path.sep, '/').replace('static/', '')
    simplified_image_url = simplified_image_path.replace(os.path.sep, '/').replace('static/', '')
    color_distribution_url = color_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    brightness_distribution_url = brightness_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    saturation_distribution_url = saturation_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    hue_distribution_url = hue_distribution_path.replace(os.path.sep, '/').replace('static/', '')

    return render_template('result.html', 
                           original_image=original_image_url,
                           simplified_image_path=simplified_image_url,
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url)

# Real-time video feed route
@app.route('/video_feed')
def video_feed():
    return Response(analyze_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
