import torch
from torch import nn
from tqdm.auto import tqdm
from .transformer import TransformerEncoder, PositionalEncoding
from .conformer_layer import ConformerLayer
from .utils import Transpose
from .diffusion import StepEmbedding

class SMDiffusion(nn.Module):

    def __init__(
        self,
        in_channels=80,
        out_channels=80,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        num_layers=6,
        step_embed_dim_hidden=256,
        model_scheduler=None,
    ):
        super().__init__()
        in_channels = in_channels
        filter_size = filter_size
        kernel_size = kernel_size
        dropout = dropout
        depthwise = depthwise

        self.step_embedding = StepEmbedding(in_channels, step_embed_dim_hidden, filter_size)

        self.in_layer_mel = nn.Linear(in_channels, filter_size)
        self.in_layer_x = nn.Linear(in_channels, filter_size)

        self.positional_encoding = PositionalEncoding(filter_size)

        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                4,
                conv_in=filter_size,
                conv_filter_size=1024,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=num_layers,
        )

        self.out_layer = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.GELU(),
            nn.Linear(filter_size, filter_size),
            nn.GELU(),
            nn.Linear(filter_size, out_channels),
        )

        self.scheduler = model_scheduler

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, cond_mel, noisy_mel, step, z=None):
        # concatenate c and x
        c = cond_mel
        x = noisy_mel
        x = self.in_layer_x(x)
        c = self.in_layer_mel(c)
        step = self.step_embedding(step)
        cxs = c + x
        cxs = self.positional_encoding(cxs) + step
        cxs = self.layers(cxs, condition=c)
        cxs = self.out_layer(cxs)
        if z is not None:
            loss = torch.nn.functional.mse_loss(cxs, z)
            return {
                "pred_z": cxs,
                "loss": loss,
            }
        else:
            return cxs

class SMDScheduler(nn.Module):
    """
    simple conformer model for predicting the diffusion noise level
    """

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        filter_size=256,
        kernel_size=3,
        dropout=0.1,
        depthwise=True,
        num_layers=2,
    ):
        super().__init__()
        self.in_layer = nn.Linear(in_channels, filter_size)
        self.positional_encoding = PositionalEncoding(filter_size)
        self.layers = TransformerEncoder(
            ConformerLayer(
                filter_size,
                4,
                conv_in=filter_size,
                conv_filter_size=1024,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=num_layers,
        )
        self.out_layer = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.PReLU(),
            nn.Linear(filter_size, filter_size),
            nn.PReLU(),
            nn.Linear(filter_size, out_channels),
        )
        

    def forward(self, noisy_mel, beta_next, delta=None, z=None, pred_noise=None):
        bounds = torch.cat([beta_next, delta], 1)
        mu, _ = torch.min(bounds, 1)
        mu = mu[:, None]
        x = self.in_layer(noisy_mel)
        x = self.positional_encoding(x)
        x = self.layers(x)
        x = self.out_layer(x)
        # take mean over length dimension
        est_ratios = torch.sigmoid(x)
        est_ratios = x.mean((1, 2), keepdim=True)
        beta = mu * est_ratios
        b_hat = beta.view(noisy_mel.size(0), 1, 1)
        if delta is None:
            return b_hat
        e = pred_noise
        phi_loss = delta**2. / (2. * (delta**2. - b_hat))
        phi_loss = phi_loss * (z - b_hat / (delta**2.) * e).square()
        log_in = 1e-8 + delta**2. / (b_hat + 1e-8)
        log_in = torch.clamp(log_in, 1e-8, 1e8)
        phi_loss = phi_loss + torch.log(log_in) / 4.
        phi_loss = phi_loss.sum(-1) + (b_hat / delta**2 - 1) / 2. * noisy_mel.size(-1)
        if phi_loss.mean() < 0:
            print('negative phi loss')
            phi_loss = phi_loss * 0.
        return phi_loss.mean()