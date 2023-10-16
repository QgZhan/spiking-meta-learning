import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from clock_driven import functional


def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname+f"_all_way_RBF_CKA")  #classifier实验结果保存在save文件夹下
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'], **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))
    # if config.get('visualize_datasets'):
    #     utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val  验证集
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'], drop_last=True, num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))
    else:
        eval_val = True
        train_dataset, val_dataset = random_split(dataset=train_dataset,
                                                  lengths=[int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))],
                                                  generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, drop_last=True, num_workers=8,
                                  pin_memory=True)
        utils.log('train dataset: {} (x{})'.format(train_dataset[0][0].shape, len(train_dataset)))
        val_loader = DataLoader(val_dataset, config['batch_size'], drop_last=True, num_workers=8, pin_memory=True)
        utils.log('val dataset: {} (x{})'.format(val_dataset[0][0].shape, len(val_dataset)))

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 1
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'], **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0].shape, len(fs_dataset), fs_dataset.n_classes))
        # if config.get('visualize_datasets'):
        #     utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_ways = config['n_ways']
        n_query = config['n_query']
        n_shots = config['n_shots']

        fs_dict = {}
        for n, n_way in enumerate(n_ways):
            fs_dict[n_way] = []
            for n_shot in n_shots:
                fs_sampler = CategoriesSampler(fs_dataset.label, 100, n_way, n_shot, n_query,
                                               ep_per_batch=config['ep_per_batchs'][n])
                fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler, num_workers=8, pin_memory=True)
                fs_dict[n_way].append(fs_loader)
        # fs_loaders = []
        # for n_shot in n_shots:
        #     fs_sampler = CategoriesSampler(fs_dataset.label, 200, n_way, n_shot, n_query, ep_per_batch=config['ep_per_batch'])
        #     fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler, num_workers=8, pin_memory=True)
        #     fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    # encoder1 = encoding.PoissonEncoder()  # 泊松编码

    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1 + 1):
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, drop_last=True, num_workers=8, pin_memory=True)

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        if eval_fs:
            for n_way in n_ways:
                for n_shot in n_shots:
                    aves_keys += [f'fsa-{n_way}-way-{n_shot}-shot']
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()

            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            functional.reset_net(model)
            # break

        # eval
        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item())
                aves['va'].add(acc)
                functional.reset_net(model)
                # break

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for n, n_way in enumerate(n_ways):
                for i, n_shot in enumerate(n_shots):
                    np.random.seed(0)
                    for data, _ in tqdm(fs_dict[n_way][i], desc=f'fs-{n_way}-way-{n_shot}-shot', leave=False):
                        x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=config['ep_per_batchs'][n])
                        label = fs.make_nk_label(n_way, n_query, ep_per_batch=config['ep_per_batchs'][n]).cuda()
                        with torch.no_grad():
                            logits = fs_model(x_shot, x_query).view(-1, n_way)
                            acc = utils.compute_acc(logits, label)
                            functional.reset_net(fs_model)
                        aves[f'fsa-{n_way}-way-{n_shot}-shot'].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_way in n_ways:
                for n_shot in n_shots:
                    key = f'fsa-{n_way}-way-{n_shot}-shot'
                    log_str += ' {}-way-{}-shot: {:.4f}'.format(n_way, n_shot, aves[key])
                    writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

