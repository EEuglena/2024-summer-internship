import torch
import torch.nn as nn
import pytorch_lightning as pl
from embedding import EmbeddingLayer
from interaction import InteractionLayer
from readout import ReadoutLayer


class NN(pl.LightningModule):
    def __init__(
        self,
        embed_dim,
        c_dim,
        d_dim,
        interaction_dim,
        num_interaction,
        readout_dim,
        learning_rate,
        # ref_value,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.c_dim = c_dim
        self.d_dim = d_dim
        self.interaction_dim = interaction_dim
        self.readout_dim = readout_dim
        self.learning_rate = learning_rate
        # self.std = ref_value["std"]
        # self.mu = ref_value["mu"]

        self.embedding_layer = EmbeddingLayer(self.embed_dim, self.c_dim, self.d_dim)
        # self.interaction_layers = []
        # for _ in range(num_interaction):
        #     new_layer = InteractionLayer(
        #         self.c_dim, self.d_dim, self.interaction_dim
        #     ).requires_grad_(True)
        #     self.interaction_layers.append(new_layer)
        self.inter_1 = InteractionLayer(self.c_dim, self.d_dim, self.interaction_dim)
        self.inter_2 = InteractionLayer(self.c_dim, self.d_dim, self.interaction_dim)
        self.inter_3 = InteractionLayer(self.c_dim, self.d_dim, self.interaction_dim)
        self.readout_layer = ReadoutLayer(
            self.c_dim,
            self.readout_dim,
            # self.std,
            # self.mu,
        )

        self.loss_fn = self._simple_loss_function

        self.save_hyperparameters()

    def forward(self, batch):
        batch = self.embedding_layer(batch)
        # for interaction_layer in self.interaction_layers:
        #     batch = interaction_layer(batch)
        batch = self.inter_1(batch)
        batch = self.inter_2(batch)
        batch = self.inter_3(batch)
        readout = self.readout_layer(batch)
        return readout

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch["energy"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "energy": pred}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch["energy"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "energy": pred}

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch["energy"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _simple_loss_function(self, pred, y):
        return ((pred - y) ** 2).mean()
