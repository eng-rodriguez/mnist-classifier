import torch
import logging
from typing import Tuple, List
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MNISTTrainer:
    """Handles training and evaluation of the ConvNet model on MNIST dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.trn_losses: List[float] = []
        self.val_losses: List[float] = []
        self.trn_corrects: List[int] = []
        self.val_corrects: List[int] = []

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, int]:
        """Train for one epoch."""
        self.model.train()
        correct = 0
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Training step
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += loss.item()

            if batch_idx % 600 == 0:
                self._log_progress(batch_idx, len(train_loader), loss, correct)

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss, correct

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, int]:
        """Evaluate the model on the test set."""
        self.model.eval()
        correct = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        return test_loss, correct

    def _log_progress(
        self, batch_idx: int, total_batches: int, loss: torch.Tensor, correct: int
    ):
        """Log training progress."""
        logger.info(
            f"Batch: [{batch_idx}/{total_batches}]\t"
            f"Loss: [{loss.item():.6f}]\t"
            f"Accuracy: [{100. * correct / ((batch_idx + 1) * self.batch_size):.2f}%]"
        )
