from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import os
import matplotlib
import time
from utils.image_analysis import analyze_image_colors
from utils.color_3d_processor import create_3d_visualization_data
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

    # Check if user wants 3D visualization
    view_type = request.form.get('view_type', '2d')

    if view_type == '3d':
        # Generate 3D visualization data
        viz_3d_data = create_3d_visualization_data(image, max_points=8000)

        return render_template('result_3d.html',
                               original_image=original_image_url,
                               simplified_image_path=simplified_image_url,
                               color_distribution_path=color_distribution_url,
                               brightness_distribution_path=brightness_distribution_url,
                               saturation_distribution_path=saturation_distribution_url,
                               hue_distribution_path=hue_distribution_url,
                               color_data_3d=viz_3d_data['rgb_points'],
                               cluster_data=viz_3d_data['clusters'],
                               hsv_data_3d=viz_3d_data['hsv_points'],
                               color_stats=viz_3d_data['statistics'])

    return render_template('result.html',
                           original_image=original_image_url,
                           simplified_image_path=simplified_image_url,
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url)

@app.route('/upload_3d', methods=['POST'])
def upload_3d():
    """Handle 3D visualization upload"""
    if 'image' not in request.files:
        return redirect(url_for('home'))

    image_file = request.files['image']

    if image_file.filename == '':
        return redirect(url_for('home'))

    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    image_file.save(filepath)
    image = Image.open(filepath)

    # Perform traditional analysis
    simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path, hue_distribution_path = analyze_image_colors(image, filename)

    # Generate 3D visualization data
    viz_3d_data = create_3d_visualization_data(image, max_points=8000)

    # Convert paths for URL compatibility
    original_image_url = filepath.replace(os.path.sep, '/').replace('static/', '')
    simplified_image_url = simplified_image_path.replace(os.path.sep, '/').replace('static/', '')
    color_distribution_url = color_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    brightness_distribution_url = brightness_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    saturation_distribution_url = saturation_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    hue_distribution_url = hue_distribution_path.replace(os.path.sep, '/').replace('static/', '')

    return render_template('result_3d.html',
                           original_image=original_image_url,
                           simplified_image_path=simplified_image_url,
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url,
                           color_data_3d=viz_3d_data['rgb_points'],
                           cluster_data=viz_3d_data['clusters'],
                           hsv_data_3d=viz_3d_data['hsv_points'],
                           color_stats=viz_3d_data['statistics'])

@app.route('/result_3d/<filename>')
def result_3d(filename):
    """Display 3D visualization for existing file"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return redirect(url_for('home'))

    image = Image.open(filepath)

    # Perform traditional analysis
    simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path, hue_distribution_path = analyze_image_colors(image, filename)

    # Generate 3D visualization data
    viz_3d_data = create_3d_visualization_data(image, max_points=8000)

    # Convert paths for URL compatibility
    original_image_url = filepath.replace(os.path.sep, '/').replace('static/', '')
    simplified_image_url = simplified_image_path.replace(os.path.sep, '/').replace('static/', '')
    color_distribution_url = color_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    brightness_distribution_url = brightness_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    saturation_distribution_url = saturation_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    hue_distribution_url = hue_distribution_path.replace(os.path.sep, '/').replace('static/', '')

    return render_template('result_3d.html',
                           original_image=original_image_url,
                           simplified_image_path=simplified_image_url,
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url,
                           color_data_3d=viz_3d_data['rgb_points'],
                           cluster_data=viz_3d_data['clusters'],
                           hsv_data_3d=viz_3d_data['hsv_points'],
                           color_stats=viz_3d_data['statistics'])

@app.route('/api/color_data_3d/<filename>')
def api_color_data_3d(filename):
    """API endpoint to get 3D color data as JSON"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        image = Image.open(filepath)
        viz_3d_data = create_3d_visualization_data(image, max_points=10000)
        return jsonify(viz_3d_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Real-time video feed route (disabled for Cloud Run deployment)
@app.route('/video_feed')
def video_feed():
    # Note: Real-time screen capture is not available in Cloud Run environment
    return jsonify({'error': 'Real-time capture not available in cloud environment'}), 501

if __name__ == '__main__':
    # Get port from environment variable for Cloud Run
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
