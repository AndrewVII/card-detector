# scripts/card_detector.py

import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import functional as F
import cv2
import os

# Set the model path
model_path = os.path.join(os.path.dirname(__file__), "../models/mtg_card_detector.pth")

# Define the model architecture with the same setup as used during training
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
num_classes = 2  # Background + MTG Card
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# Load the trained model weights
state_dict = torch.load(model_path)

# Filter out unexpected keys
new_state_dict = {}
for k, v in state_dict.items():
    if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
        new_state_dict[k] = v
    else:
        print(f"Skipping key {k}")

model.load_state_dict(new_state_dict, strict=False)

model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Function to detect cards in an image
def detect_cards(image):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            predictions = model(image_tensor)
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"boxes": []}
    return predictions[0]  # Get the first image's predictions


# Capture video from webcam using a different backend
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect cards in the current frame
    predictions = detect_cards(frame_rgb)

    # Draw bounding boxes around detected cards
    for box in predictions["boxes"]:
        x1, y1, x2, y2 = box.int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
