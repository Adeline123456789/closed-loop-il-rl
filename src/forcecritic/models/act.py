"""ACT-CVAE adapted from the ACT/LeRobot architecture (see THIRD_PARTY_NOTICES).

One RGB image, proprioception and H=4 observed wrench/motion/command frames.
Posterior targets are accepted only by bc_loss; forward is always z=0.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ACT(nn.Module):
    def __init__(self, state_dim=8, history_dim=16, width=128, latent_dim=32,
                 layers=2, backbone=None, feature_dim=512):
        super().__init__()
        if backbone is None:
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
            backbone = nn.Sequential(*list(net.children())[:-2])
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.image_proj = nn.Linear(feature_dim, width)
        self.state_proj = nn.Linear(state_dim, width)
        self.history_proj = nn.Linear(history_dim, width)
        self.z_proj = nn.Linear(latent_dim, width)
        self.action_proj = nn.Linear(4, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.obs_pos = nn.Parameter(torch.randn(1, 23, width) * .02)
        self.post_pos = nn.Parameter(torch.randn(1, 6, width) * .02)
        self.queries = nn.Parameter(torch.randn(1, 4, width) * .02)
        enc = lambda: nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0., batch_first=True)
        self.posterior = nn.TransformerEncoder(enc(), layers, enable_nested_tensor=False)
        self.encoder = nn.TransformerEncoder(enc(), layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            width, 4, width * 4, dropout=0., batch_first=True), layers)
        self.latent_head = nn.Linear(width, latent_dim * 2)
        self.action_head = nn.Linear(width, 4)
        self.latent_dim = latent_dim

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # freeze batch statistics as well as weights
        return self

    def visual(self, image):
        self.backbone.eval()
        with torch.no_grad():
            mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
            std=image.new_tensor([.229,.224,.225])[None,:,None,None]
            x = self.backbone((image-mean)/std)
            return F.adaptive_avg_pool2d(x, (4, 4)).flatten(2).transpose(1, 2)

    def decode(self, obs, z):
        if obs['history'].shape[1] != 4:
            raise ValueError('H must be 4')
        tokens = torch.cat((self.z_proj(z)[:, None], self.state_proj(obs['state'])[:, None],
                            self.history_proj(obs['history']), self.image_proj(self.visual(obs['rgb']))), 1)
        memory = self.encoder(tokens + self.obs_pos[:, :tokens.shape[1]])
        q = self.queries.expand(len(z), -1, -1)
        return self.action_head(self.decoder(q, memory)).tanh()

    def forward(self, obs):
        z = obs['state'].new_zeros((len(obs['state']), self.latent_dim))
        return self.decode(obs, z)

    def bc_loss(self, obs, actions, valid, kl_weight=10.):
        if actions.shape[1:] != (4, 4) or valid.shape != actions.shape[:2]:
            raise ValueError('BC requires Kx4 actions and a prefix mask')
        n = len(actions)
        tokens = torch.cat((self.cls.expand(n, -1, -1), self.state_proj(obs['state'])[:, None],
                            self.action_proj(actions)), 1) + self.post_pos
        pad = torch.cat((valid.new_zeros((n, 2)), ~valid), 1)
        mu, logvar = self.latent_head(self.posterior(tokens, src_key_padding_mask=pad)[:, 0]).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (.5 * logvar).exp()
        pred = self.decode(obs, z)
        reconstruction = ((pred - actions).abs() * valid[..., None]).sum() / (valid.sum().clamp_min(1) * 4)
        kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        return reconstruction + kl_weight * kl
