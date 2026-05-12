import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from models.mobilenet import MobileNet
import config


def predict(model, image, device, confidence_threshold=0.25, temperature=1.5, tta_runs=5):
    """
    Predicts Dish, Cuisine, Parent Category, and Food Groups.
    Includes TTA, temperature scaling, and confidence thresholding.
    """

    with open(config.CUISINE_DISHES_PATH, 'r') as f:
        cuisine_mapping = json.load(f)
    with open(config.DISH_GROUP_PATH, 'r') as f:
        group_mapping = json.load(f)
    with open(config.PARENT_DISHES_PATH, 'r') as f:
        parent_mapping = json.load(f)
    with open(config.DISHES_PATH, "r") as f:
        dishes = json.load(f)

    mappings = {
        "dishes":   dishes,
        "cuisines": cuisine_mapping,
        "parents":  parent_mapping,
        "groups":   group_mapping
    }

    # ── Build group vocab (your original logic, untouched) ──────────────────
    unique_groups = set()
    for groups in group_mapping.values():
        if isinstance(groups, list) and len(groups) == 1 and "," in groups[0]:
            groups_list = groups[0].split(",")
        else:
            groups_list = groups
        unique_groups.update(groups_list)
    group_vocab = sorted(list(unique_groups))

    model.eval()

    all_dish, all_cuis, all_pare, all_group = [], [], [], []

    # NOTE: 'image' here is expected to be a tensor (already transformed).
    # We run the clean tensor first, then TTA noisy versions.
    for i in range(tta_runs):
        tensor = image.to(device)  # first pass: clean image as-is

        with torch.no_grad():
            outputs = model(tensor)

            # Temperature scaling + softmax for calibrated probabilities
            dish_probs  = torch.softmax(outputs[config.FOOD_SERVICE_KEY["DISH"]]    / temperature, dim=1)
            cuis_probs  = torch.softmax(outputs[config.FOOD_SERVICE_KEY["CUISINE"]] / temperature, dim=1)
            pare_probs  = torch.softmax(outputs[config.FOOD_SERVICE_KEY["PARENT"]]  / temperature, dim=1)
            group_probs = torch.sigmoid(outputs[config.FOOD_SERVICE_KEY["GROUP"]]   / temperature).squeeze()

        all_dish.append(dish_probs)
        all_cuis.append(cuis_probs)
        all_pare.append(pare_probs)
        all_group.append(group_probs)

        # After first clean pass, apply TTA noise to the raw image
        # If your caller passes a PIL image before transform, swap below
        # For tensor input, small gaussian noise simulates augmentation
        if i == 0:
            image = image + torch.randn_like(image) * 0.02  # subtle noise TTA

    # ── Average all TTA runs ─────────────────────────────────────────────────
    dish_avg  = torch.stack(all_dish).mean(dim=0)
    cuis_avg  = torch.stack(all_cuis).mean(dim=0)
    pare_avg  = torch.stack(all_pare).mean(dim=0)
    group_avg = torch.stack(all_group).mean(dim=0)

    # ── Confidence-gated predictions ─────────────────────────────────────────
    dish_conf, dish_idx = dish_avg.max(dim=1)
    cuis_conf, cuis_idx = cuis_avg.max(dim=1)
    pare_conf, pare_idx = pare_avg.max(dim=1)

    detected_indices = (group_avg > 0.5).nonzero(as_tuple=True)[0]

    detected_groups = [
    group_vocab[i.item()]
    for i in sorted(detected_indices, key=lambda i: group_avg[i.item()].item(), reverse=True)
    ]

    result = {
        "specific_dish":     mappings['dishes'][dish_idx.item()]
                             if dish_conf.item() >= confidence_threshold else "uncertain",
        "dish_confidence":   round(dish_conf.item(), 3),

        "cuisine":           list(mappings['cuisines'].keys())[cuis_idx.item()]
                             if cuis_conf.item() >= confidence_threshold else "uncertain",
        "cuisine_confidence": round(cuis_conf.item(), 3),

        "parent_category":   list(mappings['parents'].keys())[pare_idx.item()]
                             if pare_conf.item() >= confidence_threshold else "uncertain",
        "parent_confidence": round(pare_conf.item(), 3),

        "food_groups":       detected_groups,
    }

    return result