import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import numpy as np
import random
import cfg


def load_yolo_model(model_path):
    return YOLO(model_path)

def run_yolo_inference(model, image, conf_threshold=0.25):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return model.predict(
        source=img_rgb,
        conf=conf_threshold,
        save=False,
        save_txt=False,
        imgsz=640
    )

def run_yolo_on_image(image, model, conf_threshold=0.25):
    results = run_yolo_inference(model, image, conf_threshold)
    return results

def run_yolo_on_video(model, video_source=0, conf_threshold=0.25):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("❌ Cannot open camera or video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = run_yolo_inference(model, frame, conf_threshold)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                class_name = cfg.COCO_CLASSES[cls_id]
                color = tuple(int(c * 255) for c in cfg.CLASS_COLORS[cls_id])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
