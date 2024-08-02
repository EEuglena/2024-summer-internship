import torch
import pytorch_lightning as pl
from model import NN
from dataset import MnistDataModule
import config
from callbacks import EarlyStopping

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    # Initialize network
    model = NN(
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )

    # DataModule
    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=config.ACCELARATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=EarlyStopping(monitor="val_loss"),
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
