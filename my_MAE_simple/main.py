import argparse
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from model import MaskedAutoencoderViT
import torch.nn as nn
import timm.optim.optim_factory as optim_factory
from torch.optim.lr_scheduler import LambdaLR
import math
from dataset import read_data
from LossGrad import NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--seed', default=22, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--min_learning_rate', default=1e-6, type=float)
    parser.add_argument('--max_learning_rate', default=1e-3, type=float)
    parser.add_argument('--warm_up_iter', default=5, type=int)
    parser.add_argument('--T_max', default=20, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--dataset_root', default='E:\\数据集\\QuaDriGa_2021.07.12_v2.6.1-0\\generalized_dataset\\10UEs100samplesUMiLOS_one_point.mat', type=str)
    return parser


def NMSE(cur_net, test_loader, device):
    # 定义设备
    device = torch.device(device)
    cur_net.eval()
    # 测试过程
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            # 过模型，得到输出
            x = x[0].to(device)
            _, x_hat, _ = cur_net(x)
            x_hat = torch.reshape(x_hat, (x_hat.shape[0], 2, -1, x_hat.shape[-1]))
            x_hat = x_hat.cpu().numpy()
            x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
            x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
            x_hat_C = x_hat_real + 1j * (x_hat_imag)
            # 计算网络输出端和输入端之间的nmse
            x = x.cpu().numpy()
            x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
            x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
            x_C = x_real + 1j * (x_imag)
            power = np.sum(abs(x_C) ** 2, axis=1)
            mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
            nmse_list = mse / power
            if i == 0:
                all_nmse_list = nmse_list
            else:
                all_nmse_list = np.concatenate((all_nmse_list, nmse_list), axis=0)
    nmse = np.mean(all_nmse_list)
    return 10*torch.log10(torch.tensor(nmse))


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 准备数据集
    # random_data_train = torch.randn(1000, 2, 32, 32, device=args.device)  # 先拿假的充数
    # random_data_test = torch.randn(100, 2, 32, 32, device=args.device)
    train_dataset, test_dataset = read_data(args.dataset_root)
    trainLoader = DataLoader(TensorDataset(train_dataset), batch_size=args.batch_size)
    testLoader = DataLoader(TensorDataset(test_dataset), batch_size=args.batch_size)

    # 准备模型和优化器
    mymodel = MaskedAutoencoderViT(
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
    ).to(args.device)
    param_groups = optim_factory.add_weight_decay(mymodel, args.weight_decay)
    myoptim = AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))
    mylambda = lambda cur_iter: cur_iter / args.warm_up_iter if cur_iter < args.warm_up_iter else \
        (args.min_learning_rate + 0.5 * (args.max_learning_rate - args.min_learning_rate) * (
                    1.0 + math.cos((cur_iter - args.warm_up_iter) / (args.T_max - args.warm_up_iter) * math.pi)))
    myscheduler = LambdaLR(myoptim, mylambda)
    loss_scaler = NativeScaler()

    # 训练过程
    best_nmse = 10
    for epoch in range(args.epochs):
        time_start = time.time()
        mymodel.train()
        total_loss = 0
        for iter_, x in enumerate(trainLoader):
            x = x[0].to(args.device)
            with torch.cuda.amp.autocast():
                loss, _, _ = mymodel(x)
            total_loss += loss.item() * x.shape[0]
            loss_scaler(loss, myoptim, parameters=mymodel.parameters())
            myoptim.zero_grad()
            if iter_ % 100 == 0:
                print(f'{iter_}/{len(trainLoader)} of the {epoch} th epoch, loss={loss}')
        total_loss = total_loss / len(trainLoader.dataset)
        print(f'epoch{epoch}, loss={total_loss}')
        myscheduler.step()
        if epoch > 0 and epoch % 2 == 0:
            nmse = NMSE(mymodel, testLoader, args.device)
            print(f'test, NMSE={nmse}')
            if nmse < best_nmse:
                torch.save(mymodel.state_dict(), 'model.pth')
                best_nmse = nmse
                print('model saved!')
        time_end = time.time()
        print(f'spend{(time_end - time_start) // 3600}h{((time_end - time_start) % 3600) // 60}min{((time_end - time_start) % 3600) % 60}s for one epoch')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)

