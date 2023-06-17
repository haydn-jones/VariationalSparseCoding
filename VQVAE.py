import torch
import torch.nn as nn



class VQ_VAE(nn.Module):
    def __init__(self, nEmbeds, embedDim, nResidLayers, nResidHiddens, nHiddens):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, nHiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(nHiddens, nHiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResStack(nResidLayers, nHiddens, nResidHiddens),
            nn.Conv2d(nHiddens, embedDim, kernel_size=1, stride=1, padding=0)
        )

        self.codebook = VQ_StraightThrough(nEmbeds, embedDim)

        self.decoder = nn.Sequential(
            ResStack(nResidLayers, embedDim, nResidHiddens),
            nn.ReLU(),
            nn.ConvTranspose2d(embedDim, nHiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(nHiddens, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, img):
        # encode the image
        z_e = self.encoder(img)

        # quantize the encoding, vq_loss needs to be included in objective function
        z_q, vq_loss, inds = self.codebook(z_e)

        recon = self.decoder(z_q)

        return recon, vq_loss, inds

    def decodeIndices(self, indices):
        """ Decodes a [batch, w, h] block of indices into `batch` images """
        enc = self.codebook.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(enc)

class VQ_StraightThrough(nn.Module):
    def __init__(self, nEmbeds, embedDim):
        super().__init__()

        self.embedDim = embedDim
        self.nEmbeds  = nEmbeds

        self.embedding = nn.Embedding(self.nEmbeds, self.embedDim)
        self.embedding.weight.data.uniform_(-1/self.nEmbeds, 1/self.nEmbeds)

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        input_shape = z_e.shape

        # Flatten input
        flat = z_e.view(-1, self.embedDim)
        dists = (flat-self.embedding.weight.unsqueeze(1)).norm(p=2, dim=-1).t()

        inds = torch.argmin(dists, dim=1).unsqueeze(1)
        oneHots = torch.zeros(inds.shape[0], self.nEmbeds, device=z_e.device)
        oneHots.scatter_(1, inds, 1)

        z_q = torch.matmul(oneHots, self.embedding.weight).view(input_shape)

        # VQ Loss
        vq_loss = (z_e.detach() - z_q).square().mean(dim=(1, 2, 3)) + 0.25*(z_e - z_q.detach()).square().mean(dim=(1, 2, 3))

        # Straight through
        z_q = z_e + (z_q - z_e).detach()

        return z_q.permute(0, 3, 1, 2).contiguous(), vq_loss, inds

class ResStack(nn.Module):
    """ Stack of residual blocks like in the VQVAE paper/implementation """
    def __init__(self, nResidLayers, nHiddens, nResidHiddens):
        super().__init__()

        self.stack = nn.Sequential(
            *[ResBlock(nHiddens, nResidHiddens) for _ in range(nResidLayers)],
            nn.ReLU()
        )

    def forward(self, x):
        return self.stack(x)

class ResBlock(nn.Module):
    """ Single residual block """
    def __init__(self, inMasks, residMasks):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(inMasks,    residMasks, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(residMasks, inMasks,    kernel_size=1, padding=0)
        )

    def forward(self, x):
        return x + self.block(x)
