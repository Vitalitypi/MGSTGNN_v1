import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.window = args.in_steps
        self.horizon = args.out_steps
        self.num_nodes = args.num_nodes
        self.model = nn.Sequential(
            nn.Linear(self.num_nodes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  
        x = x.squeeze() # [B, (W+H), N]
        x_flat = x.view(-1, x.shape[2]) # [B*(W+H), N]

        validity = self.model(x_flat)

        return validity

class Discriminator_spatial(nn.Module):
    def __init__(self, args):
        super(Discriminator_spatial, self).__init__()
        self.window = args.in_steps
        self.horizon = args.out_steps
        self.num_nodes = args.num_nodes
        self.model = nn.Sequential(
            nn.Linear(self.num_nodes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  
        x = nn.Softmax(dim=-1)(torch.matmul(x.squeeze().permute(0, 2, 1), x.squeeze())) # [B, N, H] * [B, H, N] -> [B, N, N]
        x_flat = x.reshape(-1, x.shape[2]) # [B*N, N]
        validity = self.model(x_flat)

        return validity

class Discriminator_temporal(nn.Module):
    def __init__(self, args):
        super(Discriminator_temporal, self).__init__()
        self.window = args.in_steps
        self.horizon = args.out_steps
        self.num_nodes = args.num_nodes
        self.model = nn.Sequential(
            nn.Linear(self.horizon, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = nn.Softmax(dim=-1)(torch.matmul(x.squeeze(), x.squeeze().permute(0, 2, 1))) # [B, H, N] * [B, N, H] -> [B, H, H]
        x_flat = x.reshape(-1, x.shape[2]) # [B*H, H]
        validity = self.model(x_flat)

        return validity
