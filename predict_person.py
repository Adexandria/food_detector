import torch
from ultralytics import YOLO
import cv2
import numpy as np

def check_interaction(box1, box2):
    """
    Returns True if two bounding boxes intersect.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x1_1 > x2_2 or x1_2 > x2_1:
        return False
    if y1_1 > y2_2 or y1_2 > y2_1:
        return False
    return True


def predict_person_with_plate(frame):

    yolo_model = YOLO('models/yolo11n.pt')

    interacting = False

    img_array = np.array(frame)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    frame_resized = cv2.resize(img_bgr, (640, 640))

    # YOLO detection
    results = yolo_model(frame_resized, conf=0.05, classes=[0, 45])  # person and bowl

    for result in results:

        person_boxes, bowl_boxes = [], []

        for box in result.boxes:
            class_id = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            if class_id == 0:
                person_boxes.append(coords)
            elif class_id == 45:
                bowl_boxes.append(coords)

            # Check for interaction
        for p_box in person_boxes:
            for b_box in bowl_boxes:
                if check_interaction(p_box, b_box):
                    interacting = True
                    break
            if interacting:
                break


    return interacting









