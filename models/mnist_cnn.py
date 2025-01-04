import torch.nn as nn
import torch.nn.functional as F
from configs.model_config import ModelConfig


class ConvNet(nn.Module):
    """CNN architecture for MNIST classification."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            config.INPUT_CHANNELS, config.CONV1_CHANNELS, kernel_size=3, stride=1
        )
        self.conv2 = nn.Conv2d(
            config.CONV1_CHANNELS, config.CONV2_CHANNELS, kernel_size=3, stride=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(5 * 5 * config.CONV2_CHANNELS, config.FC1_UNITS)
        self.fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)
        self.fc3 = nn.Linear(config.FC2_UNITS, config.NUM_CLASSES)

    def forward(self, x):
        """Forward pass of the model."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
