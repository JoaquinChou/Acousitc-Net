import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from networks.repvgg_with_sound_pressure_net import get_RepVGG_func_by_name
from dataset.dataset import StftDataset
from utils import load_checkpoint
import time
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch RepVGG Test')
parser.add_argument('--test_image_dir', dest='test_image_dir',
                    help='The directory used to test the models',
                    default='/home2/zgx/data/sound_sources/stft/test/', type=str)
parser.add_argument('--data_dir', dest='data_dir',
                    help='The directory used to test the models',
                    default='/home2/zgx/data/sound_sources/single_sound_source_10000/', type=str)
parser.add_argument('--result_dir',
                    default='/home2/zgx/data/sound_sources/test_deploy_RepVGG_B0_alpha_10/',
                    type=str, help='Directory for results')

parser.add_argument('--mode', metavar='MODE', default='train', choices=['train', 'deploy'], help='train or deploy')
parser.add_argument('--weights', dest='weights',
                    help='Path to weights',
                    default='/home2/zgx/data/AcousticNet/AcousticNet_models/08-15-13-09/single_sound_source_repVGG_B0_pressure_and_location_140.pth',
                    type=str)
parser.add_argument('--convert_weights', dest='convert_weights',
                    help='Path to convert_weights',
                    default='/home2/zgx/data/AcousticNet/AcousticNet_models/alpha_10/convert_single_sound_source_repVGG_B0_pressure_and_location_140.pth',
                    type=str)
parser.add_argument('--gpus', default='3', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for testloader')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-B0')

args = parser.parse_args()

# 选定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# 加载模型
repvgg_build_func = get_RepVGG_func_by_name(args.arch)
model = repvgg_build_func(deploy=args.mode == 'deploy')
if args.mode == 'train':
    load_checkpoint(model, args.weights)
elif args.mode == 'deploy':
    load_checkpoint(model, args.convert_weights)
model.cuda()

# 加载声源数据
test_loader = torch.utils.data.DataLoader(
    StftDataset(args.data_dir, args.test_image_dir),
    batch_size=1, shuffle=True,
    num_workers=4, pin_memory=False)

# 定义横纵坐标

location_mde = []
pressure_mae = []
location_mape = []
pressure_mape = []

# 保存距离
if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
filename = args.result_dir + 'distance.txt'
total_test_loss = 0.
time_count = []
model.eval()
with torch.no_grad():
    for i, (stft_image, test_raw_sound_data, test_location, test_pressure) in enumerate(tqdm(test_loader), 0):
        test_input_var = stft_image.cuda()
        test_location_var = torch.tensor(test_location, dtype=torch.float32).cuda()
        test_pressure_var = torch.tensor(test_pressure, dtype=torch.float32).cuda()

        test_raw_sound_data = test_raw_sound_data.cuda()

        # 计算输出
        start_time = time.time()
        test_output_location = model(test_input_var, test_raw_sound_data)[0]
        test_output_pressure = model(test_input_var, test_raw_sound_data)[1]

        np_output_location = np.around(test_output_location.cpu().numpy(), decimals=2)
        np_target_location = test_location.cpu().numpy()
        # print("!!!!!!!!5!", np_target_location)
        np_output_pressure = np.around(test_output_pressure.cpu().numpy(), decimals=2)
        np_target_pressure = test_pressure.cpu().numpy()
        # print("!!!!!!!!6!", np_target_pressure)

        location_mde.append(np.linalg.norm(np_target_location - np_output_location))
        location_mape.append(np.linalg.norm(np_target_location - np_output_location) / np.sqrt(
            np_target_location[0][0] ** 2 + np_target_location[0][1] ** 2 + 0.000001))
        pressure_mae.append(np.linalg.norm(np_output_pressure - np_target_pressure, ord=1))
        pressure_mape.append(np.linalg.norm(np_output_pressure - np_target_pressure, ord=1) / np_target_pressure[0][0])

        # 计算测试时间
        torch.cuda.synchronize()
        time_count.append(time.time() - start_time)
        # distance = np.linalg.norm(np_target_location - np_output_location) + np.linalg.norm(
        #     np_output_pressure - np_target_pressure)
        # distance = np.around(distance, decimals=2)

        # print("+++++++++1", np.around(distance, decimals=2))
        # print("+++++++++2", np_output)
        # print("+++++++++3", np_target)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 100)
        # print("___________", np_output_pressure)
        ax.plot(np_output_location[0][0], np_output_location[0][1], np_output_pressure[0][0], 'bo', label="points")
        ax.text(np_output_location[0][0], np_output_location[0][1], np_output_pressure[0][0],
                '({:.2f},{:.2f},{:.2f})'.format(np_output_location[0][0], np_output_location[0][1],
                                                np_output_pressure[0][0]),
                color='blue')
        ax.plot(np_target_location[0][0], np_target_location[0][1], np_target_pressure[0][0], 'ro', label="points")
        ax.text(np_target_location[0][0], np_target_location[0][1], np_target_pressure[0][0],
                '({:.2f},{:.2f},{:.2f})'.format(np_target_location[0][0], np_target_location[0][1],
                                                np_target_pressure[0][0]), color='red')
        # plt.plot(0, 0, 'ro', label="points")
        plt.savefig(args.result_dir + str(i + 1) + '.jpg')
        # plt.show()

        with open(filename, 'a') as file_object:
            file_object.write(
                '{}_real_location:x__{:4f}__y__{:4f}__real_pressure:{:4f}\n'.format(i, np_target_location[0][0],
                                                                                    np_target_location[0][1],
                                                                                    np_target_pressure[0][0]))
            file_object.write(
                '{}_predict_location:x__{:4f}__y__{:4f}__predict_pressure:{:4f}\n'.format(i, np_output_location[0][0],
                                                                                          np_output_location[0][1],
                                                                                          np_output_pressure[0][0]))

            file_object.write('{}_location_error:{:2f}___pressure_error:{:2f}\n'.format(i, np.linalg.norm(
                np_target_location - np_output_location), np.linalg.norm(
                np_output_pressure - np_target_pressure, ord=1)))

    print("******************", np.sum(time_count))
    with open(filename, 'a') as file_object:
        file_object.write(
            '1000_test_location_mde:{:2f}___pressure_mae:{:2f}\n'.format(np.mean(location_mde),
                                                                         np.mean(pressure_mae)))

        file_object.write(
            '1000_test_location_mape:{:2f}___pressure_mape:{:2f}\n'.format(np.mean(location_mape),
                                                                           np.mean(pressure_mape)))
