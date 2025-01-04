import torch
import logging
from configs.model_config import ModelConfig
from data.datamodule import MNISTDataModule
from models.mnist_cnn import ConvNet
from trainers.mnist_trainer import MNISTTrainer
from utils.visualization import plot_metrics

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

    # Initialize trainer
    trainer = MNISTTrainer(
        model, criterion, optimizer, batch_size=ModelConfig.BATCH_SIZE
    )

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(ModelConfig.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{ModelConfig.EPOCHS}")
        train_loss, train_correct = trainer.train_epoch(train_loader)
        val_loss, val_correct = trainer.evaluate(test_loader)

        trainer.trn_losses.append(train_loss)
        trainer.trn_corrects.append(train_correct)
        trainer.val_losses.append(val_loss)
        trainer.val_corrects.append(val_correct)

    torch.save(model.state_dict(), "weights/mnist_cnn.pth")

    # Plot and save results
    logger.info("Plotting and saving results...")
    plot_metrics(
        trainer.trn_losses,
        trainer.trn_corrects,
        trainer.val_losses,
        trainer.val_corrects,
        save_path="results/mnist_model_metrics.png",
    )


if __name__ == "__main__":
    main()
