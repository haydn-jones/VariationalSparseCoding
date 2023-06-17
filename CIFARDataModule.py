import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.root = './data/'

    def prepare_data(self):
        # download
        CIFAR10(root=self.root, train=True, download=True)
        CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage=None):
        # transform
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # dataset
        self.train_dataset = CIFAR10(self.root, train=True,  transform=transform)
        self.val_dataset   = CIFAR10(self.root, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)