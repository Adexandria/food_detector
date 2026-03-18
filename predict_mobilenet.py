import torch
from PIL import Image
import json
from models.mobilenet import MobileNet


def predict(model, image,device):
    """
    Predicts Dish, Cuisine, Parent Category, and Food Groups.
    'mappings' should be a dict containing your lists of names.
    """

    with open("data/cuisinedishes.json", 'r') as f:
        cuisine_mapping = json.load(f)    

    with open("data/dishfoodgroup.json", 'r') as f:
        group_mapping = json.load(f)

    with open("data/parentdishes.json", 'r') as f:
        parent_mapping = json.load(f)

    with open("data/dishes.json", "r") as f:
        dishes = json.load(f)
    
    mappings = {
    "dishes": dishes,
     "cuisines": cuisine_mapping,
     "parents" : parent_mapping,
     "groups" : group_mapping
    }

    model.eval()

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        
        # 2. Get Single-Choice Indices
        _, dish_idx = torch.max(outputs["dish"], 1)
        _, cuis_idx = torch.max(outputs["cuisine"], 1)
        _, pare_idx = torch.max(outputs["parent"], 1)

        group_probs = torch.sigmoid(outputs["group"]).squeeze()
        detected_indices = (group_probs > 0.5).nonzero(as_tuple=True)[0]
        
    unique_groups = set()
    for groups in group_mapping.values():
        if isinstance(groups, list) and len(groups) == 1 and "," in groups[0]:
                groups_list = groups[0].split(",")
        else:
                groups_list = groups
        unique_groups.update(groups_list)
            
    group_vocab = sorted(list(unique_groups))

    result = {
        "specific_dish": mappings['dishes'][dish_idx.item()],
        "cuisine": list(mappings['cuisines'].keys())[cuis_idx.item()],
        "parent_category": list(mappings['parents'].keys())[pare_idx.item()],
        "food_groups": [group_vocab[i.item()] for i in detected_indices]
    }
    
    return result
