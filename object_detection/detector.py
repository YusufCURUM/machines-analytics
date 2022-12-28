import contextlib
import time

from queue import Queue
from threading import Thread
import pandas as pd

import cv2
import numpy as np
import pafy
import simplejpeg
from flask import Flask

from .singleton import Singleton
print("Here")
app = Flask(__name__)
# font = cv2.FONT_HERSHEY_PLAIN
font = cv2.FONT_HERSHEY_SIMPLEX

# load Yolo model

# Load the COCO dataset labels
labels_file = "./object_detection/yolo4/classes.txt"
LABELS = [line.rstrip() for line in open(labels_file)]

# layer_names = MODEL.getLayerNames()
output_layers = None  # [layer_names[i - 1] for i in MODEL.getUnconnectedOutLayers()]
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
Conf_threshold = 0.7
NMS_threshold = 0.5
STREAM_SEC = 0.3
IGNORE_FRAME_EVERY_STREAM_SEC = True


@contextlib.contextmanager
def time_stat(func_name):
    """measures the running time for a piece of code"""
    tic = time.time()
    try:
        yield
    except:
        print(f"Error running {func_name}")
    finally:
        print(f"Time taken to run {func_name}: {time.time() - tic}")


def stream_to_queue(video_name, q, fps, cap, allowed_labels, labels_count, model, type_='live'):
    # Continuously read from the video stream and yield the result
    starting_time = time.time()
    frame_counter = 0
    while fps:
        ret, frame = cap.read()
        frame_counter += 1
        if IGNORE_FRAME_EVERY_STREAM_SEC:
            display_frame = int(frame_counter % fps * STREAM_SEC) == 0
            if not display_frame:
                continue

        if not ret:
            if type_ == 'live':
                continue
            else:
                q.put("BREAK")
                break

        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            label = LABELS[classid]
            if label in allowed_labels:
                labels_count[label] = labels_count.get(label, 0) + 1

            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (LABELS[classid], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
        df = pd.DataFrame.from_dict([labels_count])
        df.to_csv(r'%s.csv' % video_name, index=False, header=True)
        endingTime = time.time() - starting_time
        fps = frame_counter / endingTime
        cv2.putText(frame, f'FPS: {fps}', (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        frame = simplejpeg.encode_jpeg(frame, colorspace='RGB')
        q.put(frame)
        del frame


class YoloObjectDetection(metaclass=Singleton):
    """Object detector through a streaming video, YouTube live, or recorded videos, using Yolo 4
    """
    def __init__(self, video_url, type_='live', cam_name='default'):
        if "youtube" in video_url:
            video = pafy.new(video_url)
            best = video.getbest()
            video_url = best.url
        self.video_url = video_url
        self.type_ = type_
        self.init_model()
        self.cap = cv2.VideoCapture(self.video_url)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.allowed_labels = ["person", "car", "truck", "bus", "bicycle"]
        self.labels_count = {}
        self.q = Queue()
        self.th = Thread(target=stream_to_queue,
                         args=(cam_name, self.q, self.fps, self.cap, self.allowed_labels, self.labels_count, self.MODEL,))
        self.th.start()

    def init_model(self):
        self.MODEL = cv2.dnn.readNet("./object_detection/yolo4/yolov4-tiny.weights", "./object_detection/yolo4/yolov4-tiny.cfg")
        self.MODEL.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.MODEL.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.MODEL = cv2.dnn_DetectionModel(self.MODEL)
        self.MODEL.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    def get_frames(self):
        while True:
            jpeg_frame = self.q.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')


# NOTE: youtube dl package is broke for live videos
# fix it by changing the following lines in the following file
# /.venv/lib/python3.10/site-packages/pafy/backend_youtube_dl.py

# self._likes = self._ydl_info.get('like_count', 0)
# self._dislikes = self._ydl_info.get('dislike_count', 0)
