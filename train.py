import lightning.pytorch as pl
import torch

from CIFARDataModule import CIFARDataModule
from Trainer import VQVAETrainer
from DDICallback import DDICallback

torch.set_float32_matmul_precision('medium')

model = VQVAETrainer()
dm = CIFARDataModule(batch_size=128)

ddi_steps = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

trainer = pl.Trainer(
    accelerator='gpu',
    max_epochs=4_000,
    devices=1,
    benchmark=True,
    callbacks=[pl.callbacks.RichProgressBar(), pl.callbacks.LearningRateMonitor(), DDICallback(ddi_steps)],
    inference_mode=True,
    num_sanity_val_steps=-1, # So we can DDI over entire validation set before training
)

trainer.fit(model, dm)