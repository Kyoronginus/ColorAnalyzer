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
from PyQt5.QtCore import Qt,QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox,QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap
from psd_tools import PSDImage
import subprocess
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

def clip_to_psd(clip_path, output_path):
    # CLIPファイルをPSDに変換するためのコマンドを実行
    command = f"python clip_to_psd.py {clip_path} -o {output_path}"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully converted {clip_path} to {output_path}")  # Debug
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {clip_path} to PSD: {e}")
        return False

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

class ClipFileHandler(FileSystemEventHandler):
    update_signal = pyqtSignal(str)

    def __init__(self, update_function):
        super().__init__()
        self.update_function = update_function

    def on_modified(self, event):
        if event.src_path.endswith(('.clip', '.psd')):
            self.update_function(f"Updated file: {event.src_path}")
            try:
                if event.src_path.endswith('.psd'):
                    color_analysis_result = analyze_image_colors(event.src_path, os.path.basename(event.src_path))
                    if color_analysis_result:
                        self.update_function(f"Color analysis completed for: {event.src_path}")
                        self.update_function(f"Results: {color_analysis_result}")
                    else:
                        self.update_function(f"Failed to analyze {event.src_path}.")
                elif event.src_path.endswith('.clip'):
                    # CLIPファイルをPSDに変換
                    output_psd_path = os.path.join('static/uploads', os.path.basename(event.src_path).replace('.clip', '.psd'))
                    if clip_to_psd(event.src_path, output_psd_path):
                        # PSD への変換に成功した場合、色分析を実行
                        color_analysis_result = analyze_image_colors(output_psd_path, os.path.basename(output_psd_path))
                        if color_analysis_result:
                            self.update_function(f"Color analysis completed for: {output_psd_path}")
                            self.update_function(f"Results: {color_analysis_result}")
                        else:
                            self.update_function(f"Failed to analyze {output_psd_path}.")
                    else:
                        self.update_function(f"Failed to convert {event.src_path} to PSD.")
            except Exception as e:
                self.update_function(f"Error processing file {event.src_path}: {e}")




# ファイル監視のためのスレッド
class MonitorThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._is_running = True

    def run(self):
        event_handler = ClipFileHandler(self.update_signal.emit)
        observer = Observer()
        directory = os.path.dirname(self.file_path)
        observer.schedule(event_handler, directory, recursive=False)
        observer.start()

        while self._is_running:
            time.sleep(1)

        observer.stop()
        observer.join()

    def stop(self):
        self._is_running = False


from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QScrollArea

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor_thread = None
        self.selected_file = None
        self.initUI()

    def initUI(self):
        # スクロール可能なウィジェットエリアを作成
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        
        # メインウィジェットをスクロールエリア内に設定
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ファイル選択関連のウィジェット
        self.label = QLabel('Select a .psd file to monitor:')
        layout.addWidget(self.label)

        self.select_button = QPushButton('Select .psd File', self)
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.monitor_button = QPushButton('Start Monitoring', self)
        self.monitor_button.setEnabled(False)
        self.monitor_button.clicked.connect(self.start_monitoring)
        layout.addWidget(self.monitor_button)

        self.stop_button = QPushButton('Stop Monitoring', self)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_monitoring)
        layout.addWidget(self.stop_button)

        # ログ出力用
        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.result_label = QLabel('Analysis Results will be shown here.')
        layout.addWidget(self.result_label)

        # メイン画像表示用の QLabel
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 400)  # サイズを指定
        layout.addWidget(self.image_label)

        # 色分析結果の画像表示用 QLabel
        self.color_distribution_label = QLabel(self)
        self.color_distribution_label.setFixedSize(400, 200)
        layout.addWidget(self.color_distribution_label)

        self.brightness_curve_label = QLabel(self)
        self.brightness_curve_label.setFixedSize(400, 200)
        layout.addWidget(self.brightness_curve_label)

        self.saturation_distribution_label = QLabel(self)
        self.saturation_distribution_label.setFixedSize(400, 200)
        layout.addWidget(self.saturation_distribution_label)

        self.hue_wheel_label = QLabel(self)
        self.hue_wheel_label.setFixedSize(400, 400)
        layout.addWidget(self.hue_wheel_label)

        # レイアウトをスクロールエリアにセット
        scroll_area.setWidget(widget)

        # 全体のレイアウトを作成してウィンドウに適用
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

        # ウィンドウのタイトルとサイズを設定
        self.setWindowTitle('psd File Monitor')
        self.setGeometry(300, 300, 500, 800)  # ウィンドウの幅と高さを指定

    def select_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Clip/Psd Files (*.psd *.clip)")
        file_path, _ = file_dialog.getOpenFileName(self, "Select .psd or .clip File", "", "Clip/Psd Files (*.psd *.clip)")
        if file_path:
            self.selected_file = file_path
            self.label.setText(f'Selected File: {os.path.basename(file_path)}')
            self.monitor_button.setEnabled(True)

    def start_monitoring(self):
        if self.selected_file:
            self.log_box.append(f"Monitoring started for: {self.selected_file}")
            self.monitor_thread = MonitorThread(self.selected_file)
            self.monitor_thread.update_signal.connect(self.update_log)
            self.monitor_thread.start()
            self.monitor_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            QMessageBox.warning(self, 'No File Selected', 'Please select a .psd file to monitor.')

    def stop_monitoring(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.log_box.append("Monitoring stopped.")
            self.stop_button.setEnabled(False)
            self.monitor_button.setEnabled(True)

    def update_log(self, message):
        self.log_box.append(message)

        if "Results:" in message:
            result_files = message.split("Results: ")[-1].strip("()").replace("'", "").split(", ")
            simplified_image_path = os.path.normpath(result_files[0].strip())
            color_distribution_path = os.path.normpath(result_files[1].strip())
            brightness_curve_path = os.path.normpath(result_files[2].strip())
            saturation_distribution_path = os.path.normpath(result_files[3].strip())
            hue_wheel_path = os.path.normpath(result_files[4].strip())

            # 各結果画像を表示
            self.display_image(simplified_image_path, self.image_label)
            self.display_image(hue_wheel_path, self.hue_wheel_label)
            self.display_image(color_distribution_path, self.color_distribution_label)
            self.display_image(brightness_curve_path, self.brightness_curve_label)
            self.display_image(saturation_distribution_path, self.saturation_distribution_label)

    def display_image(self, image_path, label):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(label.size(), aspectRatioMode=Qt.KeepAspectRatio)
                label.setPixmap(scaled_pixmap)
                self.log_box.append(f"Image displayed: {image_path}")
            else:
                self.log_box.append(f"Failed to load image: {image_path}")
        else:
            self.log_box.append(f"Image file not found: {image_path}")

    def closeEvent(self, event):
        if self.monitor_thread:
            self.monitor_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
