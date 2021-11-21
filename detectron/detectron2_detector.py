import torch
import detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import os, json, cv2, random

from imutils.video import FileVideoStream
from imutils.video import FPS
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

WEIGHT_PATH = './detectron/weights/model_final.pth'

# Argoverse classes for detection
CLASSES = ['person',  'bicycle',  'car',  'motorcycle',  'bus',  'truck',  'traffic_light',  'stop_sign']

class Detectron2Detector(object):
    def __init__(self):
        # Load model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = WEIGHT_PATH  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
        predictor = DefaultPredictor(cfg)
        MetadataCatalog.get("argo").set(thing_classes=CLASSES)
        metadata = MetadataCatalog.get("argo")
        self.__dict__.update(locals()) 

    def detectImage(self, path):
        img = cv2.imread(path)  # BGR
        assert img is not None, f'Image Not Found {path}'
        img_out = self.detect(img)
        cv2.imshow('result', img_out)
        cv2.waitKey(0)  # 1 millisecond
        # Save results (image with detections)
        ext = os.path.splitext(path)
        cv2.imwrite(f'{ext[0]}_result{ext[1]}', img_out)

    def detectVideo(self, path):
        fvs = FileVideoStream(path).start()                 
        time.sleep(1.0)
        # start the FPS timer
        while fvs.more():	
            frame = fvs.read()
            if frame is not None:
                img_out = self.detect(frame)
                cv2.imshow("Frame", img_out)
                cv2.waitKey(1) 

    @torch.no_grad()
    def detect(self, frame):
        outputs = self.predictor(frame)
        v = Visualizer(frame,
                    metadata=self.metadata, 
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))        
        return out.get_image()
