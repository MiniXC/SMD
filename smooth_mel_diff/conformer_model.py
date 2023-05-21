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
                2,
                conv_in=filter_size,
                conv_filter_size=filter_size,
                conv_kernel=(kernel_size, 1),
                batch_first=True,
                dropout=dropout,
                conv_depthwise=depthwise,
            ),
            num_layers=num_layers,
        )

        self.out_layer = nn.Linear(filter_size, out_channels)

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
        cx = self.in_layer_mel(c) + self.in_layer_x(x)
        cx = self.positional_encoding(cx)
        step = self.step_embedding(step)
        cx = cx + step
        cx = self.layers(cx)
        cx = self.out_layer(cx)
        if z is not None:
            loss = torch.nn.functional.mse_loss(cx, z)
            return {
                "pred_z": cx,
                "loss": loss,
            }
        else:
            return cx