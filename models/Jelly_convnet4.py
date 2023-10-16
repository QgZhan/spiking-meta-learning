import torch.nn as nn

from .models import register
from clock_driven import neuron

def conv_block(in_channels, out_channels, v_threshold=1.0, v_reset=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
        nn.MaxPool2d(2, 2)
    )


@register('Jelly_convnet4')
class ConvNet4(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, tau=2.0, T=8, v_threshold=1.0, v_reset=0.0 ):
        super().__init__()
        self.T = T
        self.static_conv = nn.Sequential(
            nn.Conv2d(x_dim, hid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hid_dim),

        )
        self.encoder = nn.Sequential(
            conv_block(hid_dim, hid_dim,  v_threshold=1.0, v_reset=0.0),
            conv_block(hid_dim, hid_dim,  v_threshold=1.0, v_reset=0.0),
            conv_block(hid_dim, z_dim,  v_threshold=1.0, v_reset=0.0),
        )
        self.out_dim = 6400


    def forward(self, spike):
        spike = self.static_conv(spike)

        out_spike = self.encoder(spike)
        # out_spike_count = out_spike.view(out_spike.shape[0], -1)
        out_spike_count = out_spike
        for t in range(1, self.T):
            out_spike = self.encoder(spike)
            # out_spike_count += out_spike.view(out_spike.shape[0], -1)
            out_spike_count += out_spike

        return out_spike_count / self.T

