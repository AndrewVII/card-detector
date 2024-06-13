import cv2
from ultralytics import YOLO


def load_model(weights="best.pt"):
    model = YOLO(weights)
    return model


def run_inference(model, frame):
    results = model(frame)
    return results


def draw_results(frame, results, model):
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].cpu().numpy()  # bounding box coordinates
            conf = box.conf[0].cpu().numpy()  # confidence score
            cls = int(box.cls[0].cpu().numpy())  # class id

            print(model.names[cls])

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{model.names[cls]} {conf:.2f}"
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            top = max(y1, label_size[1])
            cv2.rectangle(
                frame,
                (x1, top - label_size[1]),
                (x1 + label_size[0], top + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

    return frame


if __name__ == "__main__":
    model = load_model("best.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        results = run_inference(model, frame)
        result_frame = draw_results(frame, results, model)

        cv2.imshow("YOLOv8 Real-Time Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
