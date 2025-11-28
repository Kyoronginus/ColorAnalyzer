from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import os
import matplotlib
import io
import base64
from logics.analyze_image_colors import analyze_image_colors

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# Configure upload and result directories (only if needed for temp files, but we are trying to avoid them)
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure required directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({"message": "Color Analyzer API"})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = image_file.filename
    
    # Analyze image and get base64 results
    try:
        simplified_image_b64, hue_distribution_b64, color_distribution_b64, brightness_distribution_b64, saturation_distribution_b64, histogram_3d_b64 = analyze_image_colors(image_file, filename)
        
        # Reset file pointer to read original image for base64 conversion
        image_file.seek(0)
        original_image = Image.open(image_file)
        buf = io.BytesIO()
        original_image.save(buf, format='PNG')
        buf.seek(0)
        original_img_str = base64.b64encode(buf.read()).decode('utf-8')
        original_image_b64 = f"data:image/png;base64,{original_img_str}"

        return jsonify({
            'original_image': original_image_b64,
            'simplified_image': simplified_image_b64,
            'color_distribution': color_distribution_b64,
            'brightness_distribution': brightness_distribution_b64,
            'saturation_distribution': saturation_distribution_b64,
            'hue_distribution': hue_distribution_b64,
            'histogram_3d': histogram_3d_b64
        })
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
