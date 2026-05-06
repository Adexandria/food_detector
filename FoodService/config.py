import os
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PARENT_DIR, "data")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
CUISINE_DISHES_PATH = os.path.join(DATA_DIR, "cuisinedishes.json")
DISH_GROUP_PATH = os.path.join(DATA_DIR, "dishfoodgroup.json")
PARENT_DISHES_PATH = os.path.join(DATA_DIR, "parentdishes.json")
DISHES_PATH = os.path.join(DATA_DIR, "dishes.json")

MOBILENET_MODEL_PATH = os.path.join(PARENT_DIR, "models", "best_food_model.pth")
RESNET_MODEL_PATH = os.path.join(PARENT_DIR, "models", "resnet_food_classifier.pth")
YOLO_MODEL_PATH = os.path.join(PARENT_DIR, "models", "yolo11n.pt")

FOOD_SERVICE_KEY = {
    "PARENT": "parent",
    "DISH": "dish",
    "CUISINE": "cuisine",
    "GROUP": "group"
}