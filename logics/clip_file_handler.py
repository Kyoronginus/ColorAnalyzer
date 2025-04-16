
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtCore import Qt,QThread, pyqtSignal
import subprocess
from logics.analyze_image_colors import analyze_image_colors
from logics.utils import clip_to_psd


class ClipFileHandler(FileSystemEventHandler):
    update_signal = pyqtSignal(str)

    def __init__(self, update_function):
        super().__init__()
        self.update_function = update_function

    def on_modified(self, event):
        if event.src_path.endswith(('.png')):
            self.update_function(f"Updated file: {event.src_path}")
            try:
                if event.src_path.endswith('.psd'):
                    color_analysis_result = analyze_image_colors(event.src_path, os.path.basename(event.src_path))
                    if color_analysis_result:
                        self.update_function(f"Color analysis completed for: {event.src_path}")
                        self.update_function(f"Results: {color_analysis_result}")
                    else:
                        self.update_function(f"Failed to analyze {event.src_path}.")

                elif event.src_path.endswith('.png'):
                    # PNGファイルをPSDに変換
                    output_psd_path = os.path.join('static/uploads', os.path.basename(event.src_path).replace('.png', '.psd'))
                    # PSD への変換に成功した場合、色分析を実行
                    color_analysis_result = analyze_image_colors(output_psd_path, os.path.basename(output_psd_path))
                    if color_analysis_result:
                        self.update_function(f"Color analysis completed for: {output_psd_path}")
                        self.update_function(f"Results: {color_analysis_result}")
                    else:
                        self.update_function(f"Failed to analyze {output_psd_path}.")
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

