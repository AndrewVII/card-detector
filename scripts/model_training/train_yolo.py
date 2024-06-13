from ultralytics import YOLO
import os
from dotenv import load_dotenv


def train_yolo(yolo_path, output_model_name):
    model = YOLO("yolov8n.pt")

    model.train(
        data=f"{yolo_path}/yolo_config.yaml",
        epochs=10,
        project=f"{yolo_path}/training_runs",
    )
    model.val()

    best_model = YOLO(f"{yolo_path}/training_runs/train/weights/best.pt")
    best_model.save(f"{yolo_path}/models/{output_model_name}.pt")


if __name__ == "__main__":
    load_dotenv()
    file_path = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(file_path, "..", "..", "yolo")
    output_model_name = os.getenv("MODEL_NAME")

    os.system(f"rm -rf {yolo_path}/training_runs/*")

    train_yolo(yolo_path, output_model_name)
