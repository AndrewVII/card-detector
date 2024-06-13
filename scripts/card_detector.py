import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


def init_config():
    model_weights_path = "output/model_final.pth"

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.DATASETS.TEST = ("my_dataset",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"
    return cfg


def initialize_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    return cap


def draw_predictions_on_frame(cap, predictor, cfg):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        return None

    outputs = predictor(frame)
    v = Visualizer(
        frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return out.get_image()[:, :, ::-1]


def main():
    config = init_config()
    predictor = DefaultPredictor(config)
    cap = initialize_webcam()
    cur_frame_num = 0

    while True:
        cur_frame = cap.read()[1]
        if cur_frame_num == 0:
            cur_frame = draw_predictions_on_frame(cap, predictor, config)
            if cur_frame is None:
                break
        cv2.imshow("Webcam Detection", cur_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cur_frame_num = (cur_frame_num + 1) % 5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
