import argparse
import os
import torch
import sys
sys.path.append('..')
from networks.repvgg_with_sound_pressure_net import get_RepVGG_func_by_name, repvgg_model_convert


parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('--weights', dest='weights',
                    help='Path to weights',
                    default='/home2/zgx/data/AcousticNet/AcousticNet_models/single_sound_source_repVGG_B0_pressure_and_location_140.pth',
                    type=str)
parser.add_argument('--convert_weights', dest='convert_weights',
                    help='Path to convert_weights',
                    default='/home2/zgx/data/AcousticNet/AcousticNet_models/convert_single_sound_source_repVGG_B0_pressure_and_location_140.pth',
                    type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-B0')

args = parser.parse_args()


def convert(arch, model_dir, save_dir):
    repvgg_build_func = get_RepVGG_func_by_name(arch)
    train_model = repvgg_build_func(deploy=False)
    if os.path.isfile(model_dir):
        print("=> loading checkpoint '{}'".format(model_dir))
        checkpoint = torch.load(model_dir)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(model_dir))

    repvgg_model_convert(train_model, save_path=save_dir)


if __name__ == '__main__':
    convert_weights = convert(args.arch, args.weights, args.convert_weights)
    print("Convert completing !!!")
