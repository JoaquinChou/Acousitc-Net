import argparse
import wandb
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
from dataset.dataset import StftDataset
from networks.repvgg_with_sound_pressure_net import get_RepVGG_func_by_name

parser = argparse.ArgumentParser(description='Propert RepVGG_B0 for single_sound_source in pytorch')
parser.add_argument('--arch', dest='arch', metavar='ARCH', default='RepVGG-B0')
parser.add_argument('--train_image_dir', dest='train_image_dir',
                    help='The directory used to train the models',
                    default='/home2/zgx/data/sound_sources/stft/train/', type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory used to train the models',
                    default='/home2/zgx/data/sound_sources/single_sound_source_10000/', type=str)
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
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs', default=8, type=int, help='Batch size for dataloader')
parser.add_argument('--lr', '--learning-rate', default=0.01, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

args = parser.parse_args()

args.model_dir = args.model_dir + '{}/'.format(time.strftime('%m-%d-%H-%M',time.localtime(time.time())))
if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
wandb.init(
    project='sound_source_location',
    entity='joaquin_chou',
    name="repVGG_B0_pressure_and_location" + "-epoch" + str(
        args.epochs) + "-lr" + str(
        args.lr),
    config=args
)

# 选定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
# torch.cuda.set_device(1)

# 加载模型
RepVGG = get_RepVGG_func_by_name(args.arch)
model = RepVGG()
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
criterion_l1 = torch.nn.L1Loss(reduction="mean")

# 定义优化器
# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

# 定义学习率策略
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[10, 50, 100], gamma=0.1, last_epoch=- 1)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True,
#                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                           eps=1e-08)
# 保存距离
# coding=UTF-8
filename = args.model_dir + 'distance_val_repVGG_B0.txt'
# 定义训练过程
print('===> Start Epoch {} End Epoch {}'.format(args.start_epoch, args.epochs + 1))

for epoch in range(args.start_epoch, args.epochs + 1):

    epoch_start_time = time.time()
    epoch_loss = 0.
    # 启动BN和dropout
    model.train()

    for batch_idx, (stft_image, raw_sound_data, location, pressure) in enumerate(train_dataloader):
        input_var = stft_image.cuda()
        location_var = torch.tensor(location, dtype=torch.float32).cuda()
        raw_sound_data = raw_sound_data.cuda()
        pressure_var = torch.tensor(pressure, dtype=torch.float32).cuda()

        # print("__________1", raw_sound_data.shape)
        # print("+++++++", location_var.shape)
        # print("+++++++2", pressure_var.shape)
        # 计算输出
        output_location = model(input_var, raw_sound_data)[0]
        # print("+++++++++++++++++", output_location.shape)
        output_pressure = model(input_var, raw_sound_data)[1]
        # print("__________3", output_pressure.shape)
        l2 = torch.mean((output_location - location_var) ** 2) * 100
        # l1 = F.smooth_l1_loss(output_pressure, pressure_var, reduction="mean")
        l1 = criterion_l1(output_pressure, pressure_var)
        batch_loss = l1 + l2

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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tl1 loss: {:.6f}\tl2 loss: {:.6f}'.
                format(
                epoch,
                (batch_idx + 1),
                len(train_dataloader),
                100. * (batch_idx + 1) / len(train_dataloader),
                l1.item(), l2.item()))

    wandb.log({"train_loss": epoch_loss / (batch_idx + 1), "epoch": epoch})

    lr_scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.8f}\tLearningRate {:.4f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, lr_scheduler.get_lr()[0]))
    # , lr_scheduler.get_lr()[0]
    print("------------------------------------------------------------------")

    if (epoch % 5 == 0):
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
            for batch_idx, (stft_image, val_raw_sound_data, val_location, val_pressure) in enumerate(val_dataloader):
                val_input_var = stft_image.cuda()
                val_location_var = torch.tensor(val_location, dtype=torch.float32).cuda()
                val_pressure_var = torch.tensor(val_pressure, dtype=torch.float32).cuda()
                val_raw_sound_data = val_raw_sound_data.cuda()

                val_output_location = model(val_input_var, val_raw_sound_data)[0]
                val_output_pressure = model(val_input_var, val_raw_sound_data)[1]
                # val_l1 = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                val_l1 = criterion_l1(val_output_location, val_location_var)
                val_l2 = torch.mean((val_output_location - val_location_var) ** 2) * 100
                val_batch_loss = val_l1 + val_l2

                # l1_loss = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                # l1_loss = criterion(val_output_pressure, val_pressure_var)
                # l2_loss = torch.mean((val_output_location - val_location_var) ** 2)
                # val_batch_loss = l1_loss
                # + l2_lossval_
                # kl_loss = F.kl_div(val_output_constraint.softmax(dim=-1).log(), val_output_sound.softmax(dim=-1), reduction='sum')
                # l1_loss = F.smooth_l1_loss(val_output_pressure, val_pressure_var, reduction="mean")
                pressure_error.append(val_l1.cpu().numpy())
                location_error.append(val_l2.cpu().numpy())

                total_val_loss += val_batch_loss.item()
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tbatch loss: {:.6f}\tAvg loss: {:.6f}'.
                    format(
                    epoch,
                    (batch_idx + 1),
                    len(val_dataloader),
                    100. * (batch_idx + 1) / len(val_dataloader),
                    val_batch_loss.item(), total_val_loss / (batch_idx + 1)))

                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tval l1 loss: {:.6f}\tval l2 loss: {:.6f}'.
                    format(
                    epoch,
                    (batch_idx + 1),
                    len(val_dataloader),
                    100. * (batch_idx + 1) / len(val_dataloader),
                    val_l1.item(), val_l2.item()))

                with open(filename, 'a') as file_object:
                    file_object.write(
                        'EPOCH{}______location:{:4f}___pressure:{:4f}\n'.format(epoch, batch_idx + 1,
                                                                                val_l2.cpu().numpy(),
                                                                                val_l1.cpu().numpy()))

        location_rmse = np.mean(location_error)
        pressure_rmse = np.mean(pressure_error)
        with open(filename, 'a') as file_object:
            file_object.write(
                'EPOCH{}___location_rmse:{:4f}___pressure_rmse:{:4f}\n'.format(epoch, location_rmse, pressure_rmse))
        wandb.log({"val_loss": total_val_loss / (batch_idx + 1), "epoch": epoch})

    if (epoch % 10 == 0) or (epoch == args.epochs + 1):
        torch.save(model.state_dict(),
                   args.model_dir + 'single_sound_source_repVGG_B0_pressure_and_location_' + str(epoch) + '.pth')
