# -*- coding: utf-8 -*-
# Author: 
#   Runsheng Xu <rxx3386@ucla.edu>
# Modified by:
#   Rongsong Li <rongsong.li@qq.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import time

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import logging_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    
    saved_path = opt.model_dir if opt.model_dir else train_utils.setup_train(hypes)
    # get logger
    logger = logging_utils.init_logging(saved_path, debug=False)

    multi_gpu_utils.init_distributed_mode(opt)

    logger.info('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, partname="train")
    opencood_validate_dataset = build_dataset(hypes, visualize=False, partname="val")
    logger.info("%d samples for training, %d samples for validation"%(
        opencood_train_dataset.__len__(),
        opencood_validate_dataset.__len__()
    ))

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    logger.info('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
        logger.info('{0}Continue Training from Epoch {1}{0}'.format(
            '-'*15, init_epoch
        ))

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        # saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)
    train_loss_list,valid_loss_list = [],[]

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    logger.info("Training Start")
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
        #     scheduler.step(epoch)
        # if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
        #     scheduler.step_update(epoch * num_steps + 0)
        # for param_group in optimizer.param_groups:
        #     logger.info("Epoch %d, learning rate: %.7f"%(epoch, param_group["lr"]))

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        train_ave_loss = []
        start_time2 = time.time()

        num_pr_list, num_gt_list = [],[]

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)
            train_ave_loss.append(final_loss.item())

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
        
        # reocrd loss
        train_loss_list.append(statistics.mean(train_ave_loss))

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            # record training detail
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)
            valid_loss_list.append(valid_ave_loss)
            logger.info("Epoch %d, train loss: %.3f, val loss: %.3f, elapsed time: %.1fs"%(
                epoch, train_loss_list[-1], valid_loss_list[-1], time.time()-start_time2
            ))
            logging_utils.draw_loss_figure(train_loss_list, 
                                            valid_loss_list, 
                                            save_dir=saved_path,
                                            init_epoch=init_epoch)

        # adjust learning rate
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            logger.info("Epoch %d, learning rate: %.7f"%(epoch, param_group["lr"]))

    logger.info("Training Finished, checkpoints saved to %s, elapsed time %.1fs"%(
        saved_path, time.time()-start_time
    ))


if __name__ == '__main__':
    main()
