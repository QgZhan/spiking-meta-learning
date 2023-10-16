import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
import utils.few_shot as fs
from clock_driven import functional


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='CKA',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def set_method(self, method):
        self.method = method

    def forward(self, x_shot, x_query):
        # print(x_shot.shape, x_query.shape)
        shot_shape = x_shot.shape[:3]  # 5：torch.Size([1, 5, 3])   2：[1, 2, 3]

        query_shape = x_query.shape[:2]  # 5：torch.Size([1, 75])     2：[1, 30]
        img_shape = x_query.shape[-3:]  # 5：torch.Size([1, 80, 80])   2：[1, 80, 80]

        # x_shot = x_shot.view(-1, *img_shape)  # 5：[15, 1, 80, 80]  2:[6,1,80,80]
        x_query = x_query.reshape(-1, *img_shape)  # 5：[25, 1, 80, 80]  2:[10, 1, 80, 80]

        # with torch.no_grad():
        #     x_shot = self.encoder(x_shot)
        #     functional.reset_net(self.encoder)
        x_query = self.encoder(x_query)

        # x_input = torch.cat([x_shot, x_query], dim=0)
        # x_tot = self.encoder(x_input)
        channel_dim = x_query.shape[-3]

        # x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(
        #     x_query):]  # 5：torch.Size([15, 6400]) torch.Size([25, 6400]) 2:torch.Size([6, 6400]) torch.Size([10, 6400])

        if self.method == 'CKA':
            x_shot = x_shot.view(*shot_shape, channel_dim, -1)
            x_query = x_query.view(*query_shape, channel_dim, -1)
        else:
            x_shot = x_shot.view(*shot_shape, -1)  # 5：[1, 5, 3, 6400]  2:[1,2,3,6400]
            x_query = x_query.view(*query_shape, -1)  # 5：[1, 25, 6400] 2:[1,10,6400]

        x_shot_label = None

        if self.method == 'cos':

            x_shot = x_shot.mean(dim=-2)  # 5:[1, 5, 6400]  #2:[1,2,6400]
            x_shot = F.normalize(x_shot, dim=-1)  # 5:[1, 5, 6400]  #2:[1,2,6400]
            x_query = F.normalize(x_query, dim=-1)  # 5:[1, 5, 6400]  #2:[1,2,6400]
            metric = 'dot'

        elif self.method == 'sqr':

            x_shot = x_shot.mean(dim=-2)  # 5:均值[1, 5, 6400]  #2:[1,2,6400]
            metric = 'sqr'

        elif self.method == 'CKA':
            # x_shot: (1, 5, 5, 512)
            # x_query: (1, 75, 512)

            x_shot = x_shot.mean(dim=2)  # 5:均值 [1, 5, 512]  #2:[1,2,512]

            source_shot_shape = x_shot.shape
            source_query_shape = x_query.shape

            x_shot = x_shot.view(x_shot.shape[0], x_shot.shape[1], -1)
            x_query = x_query.view(x_query.shape[0], x_query.shape[1], -1)

            x_shot = F.normalize(x_shot, dim=-1)  # 5:[1, 5, 512]  #2:[1,2,512]
            x_query = F.normalize(x_query, dim=-1)  # 5:[1, 75, 512] #2:[1,75,512]

            x_shot = x_shot.view(source_shot_shape)
            x_query = x_query.view(source_query_shape)

            metric = 'CKA'

        elif self.method == 'SVM':
            shot_shape = x_shot.shape
            x_shot = x_shot.view(shot_shape[0], -1, shot_shape[-1])  # 5:均值 [1, 5, 512]  #2:[1,2,512]
            x_shot = F.normalize(x_shot, dim=-1)  # 5:[1, 5, 512]  #2:[1,2,512]
            x_query = F.normalize(x_query, dim=-1)  # 5:[1, 75, 512] #2:[1,75,512]
            x_shot_label = fs.make_nk_label(shot_shape[1], shot_shape[2], ep_per_batch=shot_shape[0]).cuda()

            metric = 'SVM'

            logits = utils.compute_logits(
                x_query, x_shot,
                proto_label=x_shot_label,
                n_way=shot_shape[1], n_shot=shot_shape[2],
                metric=metric, temp=self.temp)
            return logits

        logits = utils.compute_logits(
            x_query, x_shot, metric=metric, temp=self.temp)
        return logits



