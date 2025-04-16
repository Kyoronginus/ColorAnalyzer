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

from logics.create_distributions import *



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