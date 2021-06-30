import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from dataset.dataset import StftDataset
from networks.sound_pressure_net import Sound_Pressure_CNN

parser = argparse.ArgumentParser(description='Propert Sound_Pressure_CNN for single_sound_source in pytorch')
parser.add_argument('--train_image_dir', dest='train_image_dir',
                    help='The directory used to train the models',
                    default='/home2/zgx/data/sound_sources/stft/train/', type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory used to train the models',
                    default='/home2/zgx/data/single_sound_source_10000/', type=str)
parser.add_argument('--val_image_dir', dest='val_image_dir',
                    help='The directory used to evaluate the models',
                    default='/home2/zgx/data/sound_sources/stft/val/', type=str)

parser.add_argument('--model_dir', dest='model_dir',
                    help='The directory used to save the models',
                    default='/home2/zgx/data/AcousticNet/AcousticNet_models/',
                    type=str)
# /home2/zgx/data/repvgg_single_sound_source_with_sound_pressure_models/
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs', default=8, type=int, help='Batch size for dataloader')
parser.add_argument('--lr', '--learning-rate', default=0.001, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

wandb.init(
    project='sound_source_location',
    entity='joaquin_chou',
    name="Sound_Pressure_CNN_for_single_sound_source_with_pressure_with_smoothl1" + "-epoch" + str(
        args.epochs) + "-lr" + str(
        args.lr),
    config=args
)

# 选定显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
torch.cuda.set_device(0)

# 加载模型
model = Sound_Pressure_CNN()
wandb.watch(model)
# model = nn.DataParallel(model)
model.cuda()
# model = nn.parallel.DistributedDataParallel(model)

# 加载声源数据
train_dataloader = torch.utils.data.DataLoader(
    StftDataset(args.data_dir, args.train_image_dir),
    batch_size=args.bs, shuffle=True,
    num_workers=8, pin_memory=False)

val_dataloader = torch.utils.data.DataLoader(
    StftDataset(args.data_dir, args.val_image_dir),
    batch_size=args.bs, shuffle=True,
    num_workers=8, pin_memory=False)

# 定义loss函数
# criterion = torch.nn.L1Loss(reduction="mean")

# 定义优化器
# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# 定义学习率策略
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[10, 50, 80], gamma=0.1, last_epoch=- 1)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
#                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                           eps=1e-08)
# 保存距离
# coding=UTF-8
filename = '/home2/zgx/data/single_sound_source_only_pressure_models/distance_val_only_pressure.txt'
# 定义训练过程
print('===> Start Epoch {} End Epoch {}'.format(args.start_epoch, args.epochs + 1))

for epoch in range(args.start_epoch, args.epochs + 1):

    epoch_start_time = time.time()
    epoch_loss = 0.
    # 启动BN和dropout
    model.train()
    # location,
    for batch_idx, (stft_image, pressure, raw_sound_data) in enumerate(train_dataloader):
        input_var = stft_image.cuda()
        # location_var = torch.tensor(location, dtype=torch.float32).cuda()
        raw_sound_data = raw_sound_data.cuda()
        pressure_var = torch.tensor(pressure, dtype=torch.float32).cuda()

        # print("__________", input_var.dtype)
        # print("+++++++", location_var.dtype)
        # print("+++++++", pressure_var.dtype)
        # 计算输出
        # output_location = model(input_var)[0]
        # print("+++++++++++++++++", output_location.shape)
        output_pressure, output_sound, output_constraint = model(input_var, raw_sound_data)
        # print("+++++++++++++++++", output_pressure.shape)
        # torch.mean((output_location - location_var) ** 2) +
        # batch_loss = torch.mean((output_pressure - pressure_var) ** 2)

        kl = F.kl_div(output_constraint.softmax(dim=-1).log(), output_sound.softmax(dim=-1), reduction='sum')
        l1 = F.smooth_l1_loss(output_pressure, pressure_var, reduction="mean")
        batch_loss = l1 + kl

        # batch_loss = criterion(output_pressure, pressure_var)
        # batch_loss = 10 * torch.mean((output_location - location_var) ** 2) + F.smooth_l1_loss(output_pressure,
        #                                                                                        pressure_var,
        #                                                                                        reduction="mean")

        # compute gradient and do Adam step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbatch loss: {:.6f}\tAvg loss: {:.6f}'.
                format(
                epoch,
                (batch_idx + 1),
                len(train_dataloader),
                100. * (batch_idx + 1) / len(train_dataloader),
                batch_loss.item(), epoch_loss / (batch_idx + 1)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tl1 loss: {:.6f}\tkl loss: {:.6f}'.
                format(
                epoch,
                (batch_idx + 1),
                len(train_dataloader),
                100. * (batch_idx + 1) / len(train_dataloader),
                l1.item(), kl.item()))

    wandb.log({"train_loss": epoch_loss / (batch_idx + 1), "epoch": epoch})

    lr_scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.8f}\tLearningRate {:.4f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, lr_scheduler.get_lr()[0]))
    # , lr_scheduler.get_lr()[0]
    print("------------------------------------------------------------------")

    if (epoch % 10 == 0):
        # 编写验证集保存模型
        print("begin val")
        location_error = []
        location_rmse = 0.
        pressure_error = []
        pressure_rmse = 0.
        ## val stage
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.
            # val_location,
            for batch_idx, (stft_image, val_pressure, val_raw_sound_data) in enumerate(val_dataloader):
                val_input_var = stft_image.cuda()
                # val_location_var = torch.tensor(val_location, dtype=torch.float32).cuda()
                val_pressure_var = torch.tensor(val_pressure, dtype=torch.float32).cuda()
                val_raw_sound_data = val_raw_sound_data.cuda()

                # val_output_location = model(val_input_var)[0]
                val_output_pressure, val_output_sound, val_output_constraint = model(val_input_var, val_raw_sound_data)
                # output_pressure, output_sound, output_constraint = model(input_var, raw_sound_data)
                # batch_loss = criterion(output, target_var)
                # batch_loss = torch.mean((output_location - location_var) ** 2) + torch.mean(
                #     (output_pressure - pressure_var) ** 2)
                val_batch_loss = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                # val_batch_loss = criterion(val_output_pressure, val_pressure_var)
                # + torch.mean(
                #     (val_output_location - val_location_var) ** 2)
                # l1_loss = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                # l1_loss = criterion(val_output_pressure, val_pressure_var)
                # l2_loss = torch.mean((val_output_location - val_location_var) ** 2)
                # val_batch_loss = l1_loss
                # + l2_lossval_
                # location_error.append(l2_loss.cpu().numpy())
                kl_loss = F.kl_div(val_output_constraint.softmax(dim=-1).log(), val_output_sound.softmax(dim=-1), reduction='sum')
                l1_loss = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                pressure_error.append(l1_loss.cpu().numpy())

                total_val_loss += val_batch_loss.item()
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tbatch loss: {:.6f}\tAvg loss: {:.6f}'.
                    format(
                    epoch,
                    (batch_idx + 1),
                    len(val_dataloader),
                    100. * (batch_idx + 1) / len(val_dataloader),
                    val_batch_loss.item(), total_val_loss / (batch_idx + 1)))

                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tval l1 loss: {:.6f}\tval kl loss: {:.6f}'.
                    format(
                    epoch,
                    (batch_idx + 1),
                    len(val_dataloader),
                    100. * (batch_idx + 1) / len(val_dataloader),
                    l1_loss.item(), kl_loss.item()))

                with open(filename, 'a') as file_object:
                    # ___location:
                    # l2_loss.cpu().numpy(),
                    file_object.write(
                        'EPOCH{}___{}___pressure:{:4f}\n'.format(epoch, batch_idx + 1,
                                                                 l1_loss.cpu().numpy()))

        # location_rmse = np.mean(location_error)
        pressure_rmse = np.mean(pressure_error)
        with open(filename, 'a') as file_object:
            file_object.write(
                'EPOCH{}___location_rmse:{:4f}___pressure_rmse:{:4f}\n'.format(epoch, location_rmse, pressure_rmse))
        wandb.log({"val_loss": total_val_loss / (batch_idx + 1), "epoch": epoch})

    if (epoch % 10 == 0) or (epoch == args.epochs + 1):
        torch.save(model.state_dict(),
                   args.model_dir + 'single_sound_source_epoch_only_pressure' + str(epoch) + '.pth')
