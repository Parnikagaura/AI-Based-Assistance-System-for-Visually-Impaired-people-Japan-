import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ZebraMultiTaskModel(nn.Module):
    def __init__(self):
        super(ZebraMultiTaskModel, self).__init__()

        # Use pretrained ResNet backbone
        self.backbone = models.resnet18(pretrained=True)

        # Modify first layer to accept RGB (3) + Depth (1) = 4 channels
        self.backbone.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # ---- Classification Heads ----
        self.zebra_classifier = nn.Linear(num_features, 1)
        self.light_classifier = nn.Linear(num_features, 1)
        self.blink_classifier = nn.Linear(num_features, 1)

        # ---- Regression Head (x, y position) ----
        self.position_regressor = nn.Linear(num_features, 2)

    def forward(self, rgb, depth):
        # Combine RGB + Depth
        x = torch.cat([rgb, depth], dim=1)

        features = self.backbone(x)

        zebra_out = torch.sigmoid(self.zebra_classifier(features))
        light_out = torch.sigmoid(self.light_classifier(features))
        blink_out = torch.sigmoid(self.blink_classifier(features))

        position_out = self.position_regressor(features)

        return {
            "zebra": zebra_out,
            "light": light_out,
            "blink": blink_out,
            "position": position_out
        }