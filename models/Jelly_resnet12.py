import torch.nn as nn

from .models import register
from clock_driven import neuron, layer

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, T, v_threshold=1.0, v_reset=0.0, record_firing_rate=False):
        super().__init__()
        self.T = T

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.neuron1 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.neuron2 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)
        self.neuron3 = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

        self.record_firing_rate = record_firing_rate
        if self.record_firing_rate:
            self.firing_rate = {'neuron1': 0, 'neuron2': 0, 'neuron3': 0, 'downsample': 0}

    def forward(self, x):
        # print('neuron1: ', x.shape)
        if self.record_firing_rate:
            self.firing_rate['neuron1'] += x.detach()/self.T
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.neuron1(out)

        # print('neuron2: ', out.shape)
        if self.record_firing_rate:
            self.firing_rate['neuron2'] += out.detach()/self.T
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.neuron2(out)

        # print('neuron3: ', out.shape)
        if self.record_firing_rate:
            self.firing_rate['neuron3'] += out.detach()/self.T
        out = self.conv3(out)
        out = self.bn3(out)

        # print('downsample: ', x.shape)
        if self.record_firing_rate:
            self.firing_rate['downsample'] += x.detach() / self.T
        identity = self.downsample(x)

        out += identity
        out = self.neuron3(out)

        out = self.maxpool(out)

        return out


class Jelly_ResNet12(nn.Module):

    def __init__(self, channels, tau=2.0, T=16, v_threshold=1.0, v_reset=0.0, record_firing_rate=False):
        super().__init__()
        self.record_firing_rate = record_firing_rate

        self.inplanes = 3
        self.T = T
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.neuron = neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset) #######################################

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]
        self.neuron_fc = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)

        if self.record_firing_rate:
            self.firing_rate = {'neuron': 0,
                                'layer1': self.layer1.firing_rate, 'layer2': self.layer2.firing_rate,
                                'layer3': self.layer3.firing_rate, 'layer4': self.layer4.firing_rate}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset_firing_rate(self):
        if self.record_firing_rate:
            for key, value in self.firing_rate.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.firing_rate[key][sub_key] = 0
                else:
                    self.firing_rate[key] = 0

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample, self.T, record_firing_rate=self.record_firing_rate)
        self.inplanes = planes
        return block

    def forward(self, x):
        # print('neuron:', x.shape)
        if self.record_firing_rate:
            self.firing_rate['neuron'] += x.detach() / self.T
        x = self.conv1(x)  # 输入层
        x = self.bn1(x)
        out_x = self.neuron(x)  # 编码

        out1 = self.layer1(out_x)

        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)

        # out = out4.view(out4.shape[0], out4.shape[1], -1).mean(dim=2)

        out_spikes_counter = out4  # 全连接层结果

        for t in range(1, self.T):
            if self.record_firing_rate:
                self.firing_rate['neuron'] += x.detach() / self.T
            out_x = self.neuron(x)  # 重新编码

            out1 = self.layer1(out_x)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)

            # out = out4.view(out4.shape[0], out4.shape[1], -1).mean(dim=2)
            out_spikes_counter += out4   # 全连接层结果， 用于返回分类信息

        return out_spikes_counter/self.T


@register('Jelly_resnet12')
def resnet12():
    return Jelly_ResNet12([64, 128, 256, 512])


@register('resnet12-wide')
def resnet12_wide():
    return Jelly_ResNet12([64, 160, 320, 640])

