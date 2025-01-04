import torch
import logging
from configs.model_config import ModelConfig
from data.datamodule import MNISTDataModule
from models.mnist_cnn import ConvNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(ModelConfig.RANDOM_SEED)

    # Initialize data module
    logger.info("Initializing data module...")
    data_module = MNISTDataModule(ModelConfig.BATCH_SIZE)
    data_module.setup()
    train_loader, test_loader = data_module.get_loaders()

    # Initialize model and training components
    logger.info("Initializing model and training components...")
    model = ConvNet(ModelConfig())
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LEARNING_RATE)


if __name__ == "__main__":
    main()
