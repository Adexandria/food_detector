import torch
import torchvision.models as models
import torch.nn as nn

class MobileNet(nn.Module):
    def __init__(self, num_dishes, num_cuisines, num_parents, num_groups):
        super().__init__()
        in_features = 960

        model = models.mobilenet_v3_large(pretrained=True)

        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        def create_head(out_features):
            return nn.Sequential(
                nn.Linear(in_features, 2048),
                nn.Hardswish(),
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(2048, out_features)
            )
        self.dish_head = create_head(num_dishes)
        self.cuisine_head = create_head(num_cuisines)
        self.parent_head = create_head(num_parents)
        self.group_head = create_head(num_groups)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        return {
            "dish": self.dish_head(x),
            "cuisine": self.cuisine_head(x),
            "parent": self.parent_head(x),
            "group": self.group_head(x)
        }