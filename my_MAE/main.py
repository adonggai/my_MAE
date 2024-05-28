import argparse
import datetime
import json
import pathlib
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

import engine_pretrain
from model import MaskedAutoencoderViT
import torch.nn as nn
import timm.optim.optim_factory as optim_factory
from torch.optim.lr_scheduler import LambdaLR
import math
from dataset import read_data
import advanced_tools
import os
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='epochs to warmup LR')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--dataset_root', default='E:\\数据集\\QuaDriGa_2021.07.12_v2.6.1-0\\generalized_dataset\\10UEs100samplesUMiLOS_one_point.mat', type=str)

    # 以下是一些高级配置
    parser.add_argument('--num_workers', default=0, type=int, help='commonly set between half to twice of CPU cores')
    parser.add_argument('--pin_mem', action='store_true', help='if set True, data is stored in CUDA memory so that GPU loads data faster. But may cause out of memory.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')   # 这个东西是一种架构吗？
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    return parser


def NMSE(cur_net, test_loader, device):
    # 定义设备
    device = torch.device(device)
    cur_net.eval()
    # 测试过程
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            # 过模型，得到输出
            _, x_hat, _ = cur_net(x[0])
            x_hat = torch.reshape(x_hat, (x_hat.shape[0], 2, -1, x_hat.shape[-1]))
            delta_x = x_hat - x[0]
            mse = torch.sum(torch.pow(delta_x, 2), dim=1)
            power = torch.sum(torch.pow(x[0], 2), dim=1)
            nmse_list = torch.mean(mse / power, dim=(1, 2))
            if i == 0:
                all_nmse_list = nmse_list
            else:
                all_nmse_list = torch.cat((all_nmse_list, nmse_list), dim=0)
    nmse = torch.mean(all_nmse_list)
    return 10*torch.log10(torch.tensor(nmse))


def main(args):
    # 分布式训练初始化
    advanced_tools.init_distributed_mode(args)
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')   # 打印当前脚本所在的绝对路径
    print("{}".format(args).replace(', ', ',\n'))   # 打印参数列表

    device = torch.device(args.device)
    seed = args.seed + advanced_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 准备数据集
    # random_data_train = torch.randn(1000, 2, 32, 32, device=args.device)  # 先拿假的充数
    # random_data_test = torch.randn(100, 2, 32, 32, device=args.device)
    train_dataset, test_dataset = read_data(args.dataset_root, args.device)
    myTrainset = TensorDataset(train_dataset)
    myTestset = TensorDataset(test_dataset)
    if True:
        num_tasks = advanced_tools.get_world_size()
        global_rank = advanced_tools.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(myTrainset, num_replicas=num_tasks,
                                                            rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(myTrainset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)   # 定义记录到tensorboard的操作对象
    else:
        log_writer = None

    dataloader_train = DataLoader(myTrainset, sampler=sampler_train, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    # 准备模型和优化器
    myModel = MaskedAutoencoderViT(
        img_size=32,
        patch_size=4,
        in_chans=2,
        embed_dim=768,
        depth=8,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False
    ).to(device)
    model_without_ddp = myModel
    print("Model = %s" % str(model_without_ddp))
    eff_batch_size = args.batch_size * args.accum_iter * advanced_tools.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        myModel = torch.nn.parallel.DistributedDataParallel(myModel, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = myModel.module

    # 准备其他的训练控制器
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # weight decay是一种过拟合的手段，也就是在损失函数中加入权重的范数损失。一般是权重的L2范数。
    # 目的是让神经网络在学习时优先选择绝对值小的参数值，使得模型更加平滑，提高模型的泛化能力（from chatgpt）
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = advanced_tools.NativeScalerWithGradNormCount()   # 创建梯度裁剪器

    # 模型训练
    advanced_tools.load_model(args, model_without_ddp, optimizer, loss_scaler)   # 初始参数args.resume = ''的话应该不会进任务
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader_train.sampler.set_epoch(epoch)
        engine_pretrain.train_one_epoch(myModel, dataloader_train, optimizer, device, epoch, loss_scaler, log_writer, args)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            advanced_tools.save_model(args, myModel, model_without_ddp, optimizer, loss_scaler, epoch)
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        # if args.output_dir and advanced_tools.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #         with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
        #             f.write(json.dumps(log_stats) + '\n')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

