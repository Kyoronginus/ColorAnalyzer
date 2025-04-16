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
from logics.monitor_thread import MonitorThread
import matplotlib.pyplot as plt
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
        self.label = QLabel('Select a file to monitor:')
        layout.addWidget(self.label)

        self.select_button = QPushButton('Select a File', self)
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
        file_dialog.setNameFilter("Image Files (*.psd *.clip *.png)")
        file_path, _ = file_dialog.getOpenFileName(self, "Select .psd, .clip, or .png File", "", "Image Files (*.psd *.clip *.png)")
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
