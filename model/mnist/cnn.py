import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_labels=10):
        super(CNN, self).__init__()
        # modest width bump
        self.conv1 = nn.Conv2d(1, 14, kernel_size=5)      # 28->24 -> pool -> 12
        self.conv2 = nn.Conv2d(14, 28, kernel_size=5)     # 12->8
        self.conv3 = nn.Conv2d(28, 32, kernel_size=3, padding=1)  # 8->8 (new)
        self.conv_drop = nn.Dropout2d(p=0.4)              # mild reg

        # After conv1+pool: 12x12
        # After conv2: 8x8; conv3: 8x8; pool -> 4x4
        # Flatten: 32 * 4 * 4 = 512
        self.fc1 = nn.Linear(512, 80)                     # slightly larger head
        self.fc2 = nn.Linear(80, n_labels)

        # (optional) clean init; safe to omit if you prefer defaults
        # nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        # nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")
        # nn.init.zeros_(self.conv1.bias); nn.init.zeros_(self.conv2.bias); nn.init.zeros_(self.conv3.bias)
        # nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu"); nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))             # 24 -> 12
        x = F.relu(self.conv2(x))                               # 12 -> 8
        x = F.relu(self.conv3(x))                               # 8 -> 8
        x = F.max_pool2d(self.conv_drop(x), 2)                  # 8 -> 4
        x = x.view(x.size(0), -1)                               # 32*4*4 = 512
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# # CNN used in VBFL
# class CNN(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
# 		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
# 		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
# 		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
# 		self.fc1 = nn.Linear(7*7*64, 512)
# 		self.fc2 = nn.Linear(512, 10)

# 	def forward(self, inputs):
# 		tensor = inputs.view(-1, 1, 28, 28)
# 		tensor = F.relu(self.conv1(tensor))
# 		tensor = self.pool1(tensor)
# 		tensor = F.relu(self.conv2(tensor))
# 		tensor = self.pool2(tensor)
# 		tensor = tensor.view(-1, 7*7*64)
# 		tensor = F.relu(self.fc1(tensor))
# 		tensor = self.fc2(tensor)
# 		return tensor