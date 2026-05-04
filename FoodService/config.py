PARENT_DIR = "FoodService"
DATA_DIR = f"{PARENT_DIR}\\data"
LABELS_PATH = f"{DATA_DIR}\\labels.json"
CUISINE_DISHES_PATH = f"{DATA_DIR}\\cuisinedishes.json"
DISH_GROUP_PATH = f"{DATA_DIR}\\dishfoodgroup.json"
PARENT_DISHES_PATH = f"{DATA_DIR}\\parentdishes.json"
DISHES_PATH = f"{DATA_DIR}\\dishes.json"

MOBILENET_MODEL_PATH = f"{PARENT_DIR}\\models\\best_food_model.pth"
RESNET_MODEL_PATH = f"{PARENT_DIR}\\models\\resnet_food_classifier.pth"
YOLO_MODEL_PATH = f"{PARENT_DIR}\\models\\yolo11n.pt"

FOOD_SERVICE_KEY = {
    "PARENT": "parent",
    "DISH": "dish",
    "CUISINE": "cuisine",
    "GROUP": "group"
}