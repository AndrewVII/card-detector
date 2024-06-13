import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from dotenv import load_dotenv
import os


def init_config(model):
    model_weights_path = "output/model_final.pth"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
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


def draw_predictions_on_frame(frame, predictor, cfg):
    outputs = predictor(frame)
    v = Visualizer(
        frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def main():
    load_dotenv()
    model_name = os.getenv("MODEL")
    config = init_config(model_name)
    predictor = DefaultPredictor(config)
    cap = initialize_webcam()
    cur_frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        if cur_frame_num == 0:
            result_frame = draw_predictions_on_frame(frame, predictor, config)
            if result_frame is None:
                break
        else:
            result_frame = frame

        # Calculate and display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

        cv2.imshow("Webcam Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cur_frame_num = (cur_frame_num + 1) % 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
