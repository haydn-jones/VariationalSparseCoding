import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from IndexCollapseMetric import IndexCollapseMetric
from VQVAE import VQ_VAE


class VQVAETrainer(pl.LightningModule):
    def __init__(self, ):
        super().__init__()

        self.ncodebook = 512

        self.model = VQ_VAE(self.ncodebook, 32, 4, 32, 128)

        self.val_collapse = IndexCollapseMetric(self.ncodebook)

    def forward(self, x):
        recon, vq_loss, inds = self.model(x)
        return recon, vq_loss, inds
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, _ = self(x)
        loss = F.mse_loss(recon, x) + vq_loss.mean()

        self.log('train/mse', loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss.mean(), prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, vq_loss, inds = self(x)

        loss = F.mse_loss(recon, x) + vq_loss.mean()
        self.val_collapse(inds)

        self.log('val/mse', loss)
        self.log('val/vq_loss', vq_loss.mean())

        z_e = self.model.encoder(x)
        return loss, z_e
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=1e-5, T_max=4000)
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sch,
                'interval': 'epoch',
            }
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if batch_idx != 0:
            return
        
        img = batch[0][0]
        recon, _, _ = self(img.unsqueeze(0))

        self.logger.experiment.add_image('val/input', img, self.global_step)
        self.logger.experiment.add_image('val/recon', recon[0], self.global_step)
        self.logger.experiment.add_figure('val/index_freqs', self.val_collapse.plot(), self.global_step)