import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics


class IndexCollapseMetric(torchmetrics.Metric):
    def __init__(self, num_elements, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("index_freqs", default=torch.zeros(num_elements))

    def update(self, encodings: torch.Tensor):
        indices, counts = encodings.unique(return_counts=True)
        self.index_freqs.scatter_add_(0, indices, counts.float())

    def compute(self):
        return self.index_freqs
    
    def plot(self):
        index_freqs = self.compute().cpu().numpy().astype(np.float32)
        # normalize
        index_freqs = index_freqs / index_freqs.sum()
        inds = index_freqs.argsort()[::-1]
        
        plt.figure()
        plt.bar(range(len(index_freqs)), index_freqs[inds])
        plt.xlabel('Rank of index (sorted by frequency)')
        plt.ylabel('Frequency')
        plt.title('Index frequency')

        # plt.ylim(0, index_freqs[inds[0]] * 1.1)

        self.reset()

        return plt.gcf()
