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
import matplotlib.pyplot as plt
from gui.app_window import AppWindow
from logics.create_distributions import *

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea

os.makedirs('static/results', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)


def load_stylesheet(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # 外部スタイルシートを読み込む
    app.setStyleSheet(load_stylesheet("gui/style.qss"))

    window = AppWindow()
    window.show()
    sys.exit(app.exec_())
