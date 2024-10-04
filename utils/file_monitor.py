import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ClipFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        if event.src_path.endswith('.clip'):
            print(f"Detected change in {event.src_path}")
            self.callback(event.src_path)

def monitor_clip_files(directory, callback):
    event_handler = ClipFileHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
