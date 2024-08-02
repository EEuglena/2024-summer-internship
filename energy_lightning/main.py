import torch
import pytorch_lightning as pl
from model import NN
from dataset import MD17, MD17DataModule
import config
from pytorch_lightning.callbacks import EarlyStopping

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    # raw_dataset = MD17(
    #     root=config.ROOT,
    #     name=config.NAME,
    # )
    # train_dataset = raw_dataset[: config.TRAIN_SIZE]
    # _n = len(train_dataset[0].z)
    # _std, _mu = torch.std_mean(
    #     torch.tensor([data.energy / _n for data in train_dataset], dtype=float)
    # )
    # ref_value = {"std": _std, "mu": _mu}

    # Initialize network
    model = NN(
        embed_dim=config.EMBED_DIM,
        c_dim=config.C_DIM,
        d_dim=config.D_DIM,
        interaction_dim=config.INTERACTION_DIM,
        num_interaction=config.NUM_INTERACTION,
        readout_dim=config.READOUT_DIM,
        learning_rate=config.LEARNING_RATE,
        # ref_value=ref_value,
    )
    # DataModule
    dm = MD17DataModule(
        root=config.ROOT,
        name=config.NAME,
        batch_size=config.BATCH_SIZE,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        pred_size=config.PRED_SIZE,
        val_size=config.VAL_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=EarlyStopping(monitor="val_loss", patience=10),
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
