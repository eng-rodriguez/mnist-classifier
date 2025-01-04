from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for MNIST classifier."""

    # Training parameters
    RANDOM_SEED: int = 42
    BATCH_SIZE: int = 10
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 5

    # Model architecture
    INPUT_CHANNELS: int = 1
    CONV1_CHANNELS: int = 6
    CONV2_CHANNELS: int = 16
    FC1_UNITS: int = 120
    FC2_UNITS: int = 84
    NUM_CLASSES: int = 10
