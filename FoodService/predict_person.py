import torch
from ultralytics import YOLO
import cv2
import numpy as np
import config

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

    yolo_model = YOLO(config.YOLO_MODEL_PATH)

    interacting = False
    
    cropped = None

    img_array = np.array(frame)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    orig_w, orig_h = frame.size

    frame_resized = cv2.resize(img_bgr, (640, 640))

    # YOLO detection
    results = yolo_model(frame_resized, conf=0.05, classes=[0, 45])  # person and bowl

    for result in results:
        annotated_frame = result.plot()
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
                    scale_x = orig_w / 640
                    scale_y = orig_h / 640
                    x1 = int(b_box[0] * scale_x)
                    y1 = int(b_box[1] * scale_y)
                    x2 = int(b_box[2] * scale_x)
                    y2 = int(b_box[3] * scale_y)

                    cropped = frame.crop((x1, y1, x2, y2))
                    break
            if interacting:
                break
    cv2.imwrite("annotated.jpg", annotated_frame)
    if cropped is not None:
        cv2.imwrite("cropped.jpg", cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR))

    return interacting, cropped if interacting else None









