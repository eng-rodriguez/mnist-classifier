from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class MNISTDataModule:
    """Handles MNIST dataset loading and preprocessing."""

    def __init__(self, batch_size: int, data_dir: str = "data/"):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.train_data = None
        self.test_data = None

    def setup(self) -> None:
        """Initialize train and test datasets."""
        self.train_data = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_data = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and test data loaders."""
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader
