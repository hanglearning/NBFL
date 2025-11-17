import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_labels=10):
        super().__init__()
        # Block 1 (32 -> 16 after pool)
        self.conv1a = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1a   = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b   = nn.BatchNorm2d(32)

        # Block 2 (16 -> 8 after pool)
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)

        # Block 3 (8 -> 4 after pool)
        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a   = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b   = nn.BatchNorm2d(128)

        # Final classifier (after GAP to 1x1 -> 128 features)
        self.fc = nn.Linear(128, n_labels)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, 2)                 # 32 -> 16
        x = F.dropout(x, 0.1, training=self.training)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.max_pool2d(x, 2)                 # 16 -> 8
        x = F.dropout(x, 0.2, training=self.training)

        # Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = F.max_pool2d(x, 2)                 # 8 -> 4
        x = F.dropout(x, 0.3, training=self.training)

        # GAP + classifier
        x = F.adaptive_avg_pool2d(x, 1)        # [B,128,1,1]
        x = torch.flatten(x, 1)                 # [B,128]
        return self.fc(x)                       # logits
