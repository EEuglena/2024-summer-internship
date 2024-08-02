from model import NN
from dataset import MD17DataModule
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

if __name__ == "__main__":
    ckpt_path = "./lightning_logs/version_4/checkpoints/epoch=11-step=1500.ckpt"
    hparam_path = "./lightning_logs/version_4/hparams.yaml"
    model = NN.load_from_checkpoint(ckpt_path, hparams_file=hparam_path)
    model.eval()
    dm = MD17DataModule(
        root=config.ROOT,
        name="revised ethanol",
        batch_size=config.BATCH_SIZE,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        pred_size=config.PRED_SIZE,
        val_size=config.VAL_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=EarlyStopping(monitor="val_loss", patience=10),
    )
    prediction = trainer.predict(model, dm)
    print(f"Prediction : {prediction[0].item() * 1000 : .3f} kcal/mol")
    print(
        f'Actual : {dm.test_dataloader().dataset[0]["energy"].item() * 1000 : .3f} kcal/mol'
    )
