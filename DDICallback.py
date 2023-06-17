import lightning.pytorch as pl
import numpy as np
import torch
import faiss


class DDICallback(pl.Callback):
    """
        Callback to perform Data Dependent Initialization on the codebook. Performs K-Means clustering on the validation set.

        epoch_schedule: List of epochs to perform DDI on (-1 for before training, 0 for after 0th epoch, etc.)

        Note: This callback assumes that the model has a codebook attribute
    """
    def __init__(self, epoch_schedule) -> None:
        super().__init__()

        self.epoch_schedule = epoch_schedule
        self.codes = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.sanity_checking and -1 in self.epoch_schedule):
            pass
        elif trainer.current_epoch not in self.epoch_schedule or len(self.codes) == 0:
            return
        
        self.codes.append(outputs[1].detach().cpu().numpy())

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.sanity_checking and -1 in self.epoch_schedule):
            pass
        elif trainer.current_epoch not in self.epoch_schedule or len(self.codes) == 0:
            return
        
        pl_module.print('DDI Callback: Re-initializing codebook') 
        # Aggressive parameters for K-Means, use faiss-gpu if available
        niter = 300
        nredo = 10

        codebook = pl_module.model.codebook

        codes = np.vstack(self.codes)

        flat = codes.transpose(0, 2, 3, 1).reshape(-1, codes.shape[1])

        kmeans = faiss.Kmeans(d=flat.shape[1], k=codebook.nEmbeds, niter=niter, nredo=nredo, gpu=1)
        kmeans.train(flat.astype(np.float32))

        codebook.embedding.weight.data.copy_(torch.from_numpy(kmeans.centroids).float().to(codebook.embedding.weight.device))
