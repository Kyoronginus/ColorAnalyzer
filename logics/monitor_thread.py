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
from logics.clip_file_handler import ClipFileHandler

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