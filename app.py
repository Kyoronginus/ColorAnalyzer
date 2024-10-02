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

app = Flask(__name__)

# アップロードされたファイルの保存場所
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 必要なフォルダを作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ホーム画面のルート
@app.route('/')
def home():
    return render_template('home.html')

# 画像アップロードのルート
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return redirect(url_for('home'))

    # Create file path
    filename = image_file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the image file
    image_file.save(filepath)

    # Open image with Pillow
    image = Image.open(filepath)

    # Analyze image and save results
    simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path,hue_distribution_path = analyze_image_colors(image, filename)

    # Convert paths to use forward slashes
    original_image_url = filepath.replace(os.path.sep, '/').replace('static/', '')
    simplified_image_url = simplified_image_path.replace(os.path.sep, '/').replace('static/', '')
    color_distribution_url = color_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    brightness_distribution_url = brightness_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    saturation_distribution_url = saturation_distribution_path.replace(os.path.sep, '/').replace('static/', '')
    hue_distribution_url = hue_distribution_path.replace(os.path.sep, '/').replace('static/', '')

    # Render result template with image URLs
    return render_template('result.html', 
                           original_image=original_image_url,  
                           simplified_image_path=simplified_image_url,  
                           color_distribution_path=color_distribution_url,
                           brightness_distribution_path=brightness_distribution_url,
                           saturation_distribution_path=saturation_distribution_url,
                           hue_distribution_path=hue_distribution_url)

# 画像の色分析を行う関数
def analyze_image_colors(image, filename, max_size=500):
    # Convert image to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image while maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    image = image.resize((new_width, new_height))  # Resize with aspect ratio
    img_data = np.array(image)

    # Convert RGB to HSV and block color values
    hsv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    h = (h // 3) * 3  # Group hues into blocks
    s = (s // 8) * 8  # Group saturation into blocks
    v = (v // 8) * 8  # Group value (brightness) into blocks
    
    # Create simplified RGB image
    simplified_hsv_image = np.stack([h, s, v], axis=-1)
    simplified_rgb_image = cv2.cvtColor(simplified_hsv_image, cv2.COLOR_HSV2RGB)

    # Save simplified image
    simplified_image_path = os.path.join(app.config['RESULT_FOLDER'], 'simplified_' + filename)
    Image.fromarray(simplified_rgb_image).save(simplified_image_path)

    # Call functions to create color, brightness, saturation, and hue distributions
    color_distribution_path = create_color_distribution(simplified_rgb_image, filename)
    brightness_distribution_path = create_brightness_distribution(img_data, filename)
    saturation_distribution_path = create_saturation_distribution(hsv_image, filename)
    hue_distribution_path = create_hue_distribution(hsv_image, filename)

    return simplified_image_path, color_distribution_path, brightness_distribution_path, saturation_distribution_path, hue_distribution_path


# 色分布を生成して保存する関数
def create_color_distribution(image, filename):
    reshaped_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5)  # 5つのクラスターを作成
    kmeans.fit(reshaped_img)

    # クラスタ中心（頻出色）と割合を取得
    top_colors = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)

    # 色の割合を計算
    percentages = counts / counts.sum()

    # 色分布を棒グラフで表示して保存
    color_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'color_distribution_' + filename + '.png')
    plt.figure(figsize=(6, 4))
    plt.bar(range(5), percentages, color=[top_colors[i] / 255 for i in range(5)])
    plt.title("Color Distribution")
    plt.xticks(range(5), [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in top_colors], rotation=45)
    plt.tight_layout()
    plt.savefig(color_distribution_path)
    plt.close()

    return color_distribution_path

# 明度（輝度）分布を生成して保存する関数
def create_brightness_distribution(image, filename):
    brightness = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([brightness], [0], None, [256], [0, 256])

    # ヒストグラムを曲線グラフで表示して保存
    brightness_curve_path = os.path.join(app.config['RESULT_FOLDER'], 'brightness_curve_' + filename + '.png')
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

# 彩度（サチュレーション）分布を生成して保存する関数
def create_saturation_distribution(hsv_image, filename):
    s = hsv_image[:, :, 1]
    hist = cv2.calcHist([s], [0], None, [256], [0, 256])

    # ヒストグラムを曲線グラフで表示して保存
    saturation_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'saturation_distribution_' + filename + '.png')
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

def create_hue_distribution(hsv_image, filename):
    # Extract hue and saturation channels
    h = hsv_image[:, :, 0]  # Hue channel
    s = hsv_image[:, :, 1]  # Saturation channel
    
    # Mask to exclude grayscale pixels (where saturation is 0)
    non_grayscale_mask = s > 0
    
    # Only include hues for non-grayscale pixels
    h_non_grayscale = h[non_grayscale_mask]
    
    # Create histogram for hue values (for non-grayscale pixels only)
    hue_hist, bin_edges = np.histogram(h_non_grayscale, bins=180, range=(0, 180))  # Hue histogram

    # Shift hue for visual purposes, starting at yellow (adjusting for the color wheel)
    bin_edges_shifted = (bin_edges) % 180  # Shift hue -30 degrees

    # Calculate angles for polar plot
    theta = np.linspace(0, 2 * np.pi, 180)

    # Create polar plot for hue distribution
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    # Plot hue distribution as a bar graph on the polar plot
    ax.bar(theta, hue_hist, width=2 * np.pi / 180, color=plt.cm.hsv(bin_edges_shifted[:-1] / 180.0), bottom=0.0)

    # Hide labels, ticks, and polar frame for a cleaner look
    # ax.set_xticks([])
    ax.set_yticks([])
    
    # Title and save path
    ax.set_title('Hue Distribution Wheel', va='bottom')
    hue_distribution_path = os.path.join(app.config['RESULT_FOLDER'], 'hue_wheel_' + filename + '.png')
    
    # Save the plot
    plt.savefig(hue_distribution_path)
    plt.close()

    return hue_distribution_path

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

# Real-time image analysis function with color analysis
def analyze_and_stream():
    while True:
        region = get_clip_studio_paint_window()
        if region:
            img_data = capture_screen(region)
            img_data_resized = cv2.resize(img_data, (1000, 500))

            # Analyze real-time capture and create graphs
            color_distribution_img = create_color_distribution_image(img_data)
            brightness_distribution_img = create_brightness_distribution_image(img_data)

            # Overlay the graphs onto the original frame
            combined_image = overlay_analysis_on_frame(img_data, color_distribution_img, brightness_distribution_img)

            # Convert the combined image to JPEG for streaming
            ret, jpeg = cv2.imencode('.jpg', combined_image)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            time.sleep(1)  # Adjust for performance
        else:
            yield None

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



# Flask route to serve real-time video stream
@app.route('/video_feed')
def video_feed():
    return Response(analyze_and_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
