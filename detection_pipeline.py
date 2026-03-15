import torch
from ultralytics import YOLO
import cv2
from predict import predict
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Function to check interaction
# -----------------------------
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

# -----------------------------
# Load YOLO and CNN models
# -----------------------------
yolo_model = YOLO('models/yolo11n.pt')

num_classes = 10  # must match your fine-tuned model
proposed_model = models.resnet18(weights=None)  # no pre-trained needed
proposed_model.fc = nn.Linear(proposed_model.fc.in_features, num_classes)

checkpoint_path = "models/food_classifier.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found. Ensure the file exists and the path is correct.")

checkpoint = torch.load(checkpoint_path, map_location=device)

proposed_model.load_state_dict(checkpoint['model_state_dict'])
proposed_model.eval()

# -----------------------------
# Preprocessing for CNN
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 640))
    annotated_frame = frame_resized.copy()
    interacting = False

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
                    break
            if interacting:
                break

    # Annotate and predict if interaction detected
    if interacting:
        cv2.putText(annotated_frame, "PERSON TOUCHING BOWL!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Preprocess frame for CNN
        frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        frame_tensor = preprocess(frame_pil).unsqueeze(0).to(device)

        # Run prediction
        predict(proposed_model, frame_tensor, device)

        # Show the final frame for 2 seconds
        cv2.imshow('YOLO Detection', annotated_frame)
        cv2.waitKey(2000)
        break

    # Show live detection
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()