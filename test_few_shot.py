import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from clock_driven import functional


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(dataset[0][0].shape, len(dataset), dataset.n_classes))
    n_way = 5  #####################################################################################################
    n_shot, n_query = args.shot, 15
    n_batch = 50
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(dataset.label, n_batch, n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    # encoder1 = encoding.PoissonEncoder()  # 泊松编码

    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    print(model.encoder)
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)

            with torch.no_grad():
                if not args.sauc:
                    # 前向无梯度计算x_shot
                    shot_shape = x_shot.shape[:-3]  # 5：torch.Size([1, 5, 3])   2：[1, 2, 3]
                    img_shape = x_shot.shape[-3:]  # 5：torch.Size([1, 80, 80])   2：[1, 80, 80]
                    x_shot = x_shot.view(-1, *img_shape)  # 5：[15, 1, 80, 80]  2:[6,1,80,80]
                    with torch.no_grad():
                        x_shot = model.encoder(x_shot)
                        functional.reset_net(model)
                    channel_dim = x_shot.shape[-3]
                    x_shot = x_shot.view(*shot_shape, channel_dim, -1)

                    logits = model(x_shot, x_query).view(-1, n_way)
                    label = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                    functional.reset_net(model)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(*shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)
        #
        # for key, value in model.encoder.firing_rate.items():
        #     print(key)
        #     if isinstance(value, dict):
        #         for sub_key, sub_value in value.items():
        #             print('\t', sub_key, sub_value.shape, sub_value.mean()/len(loader))
        #             # print(sub_value)
        #     else:
        #         print(value.shape, value.mean()/len(loader))
        #         # print(value)

        # model.encoder.reset_firing_rate()

        print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)
