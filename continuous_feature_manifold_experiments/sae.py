import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def forward(self, x):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, l1_coeff=1e-3):
        x_hat, z = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        l1 = z.abs().mean()
        return mse + l1_coeff * l1, {"mse": mse.item(), "l1": l1.item()}
