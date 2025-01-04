import torch
import logging
from configs.model_config import ModelConfig
from data.datamodule import MNISTDataModule

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


if __name__ == "__main__":
    main()
