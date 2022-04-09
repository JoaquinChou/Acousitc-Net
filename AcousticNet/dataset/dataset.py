import os
import warnings
import math
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils.image_utils import load_img

# ignore warnings
warnings.filterwarnings("ignore")


class StftDataset(Dataset):
    def __init__(self, data_dir, img_dir):
        super(StftDataset, self).__init__()

        # "single_sound_source_56[]": each sound source point contains 56 images.
        single_sound_source_56 = []
        single_sound_source_data = []

        for file_dir in os.listdir(img_dir):
            temp = []
            single_sound_source_data.append(os.path.join(data_dir, file_dir))
            single_sound_source_image_dir = os.path.join(img_dir, file_dir)
            for filename in os.listdir(single_sound_source_image_dir):
                temp.append(os.path.join(single_sound_source_image_dir, filename))

            # Since the image read by os.listdir is out of order, adjust it.
            temp.sort(key=lambda x: int(x.split('/')[-1][:-4]))
            single_sound_source_56.append(temp)

        self.single_sound_source_56 = single_sound_source_56
        self.single_sound_source_data = single_sound_source_data
        self.microphone_center_location = np.array([8.929e-6, -1.786e-6, 0])
        self.distance = 2.5

    def __len__(self):
        return len(self.single_sound_source_data)

    def __getitem__(self, index):

        f = h5py.File(self.single_sound_source_data[index], 'r')
        raw_sound_data = np.array([f['time_data'].value]).reshape((1024, 1, 56)).transpose(2, 0, 1)


        label = self.single_sound_source_data[index].split('_')
        x = float(label[-6])
        y = float(label[-4])
        p = float(label[-2])

        microphone_center_to_source = np.linalg.norm(np.array([x, y, self.distance]) - self.microphone_center_location)
        sound_pressure_level = 20 * math.log(p / (2 * math.pow(10, -5)) / math.pow(microphone_center_to_source, 2), 10)

        # concat the image--56×512×1024
        stft_images = 0
        for i in range(56):
            single_channel_stft_image = load_img(self.single_sound_source_56[index][i])
            single_channel_stft_image = np.array([single_channel_stft_image[:, :, 0]]).transpose(0, 2, 1)

            if i == 0:
                stft_images = single_channel_stft_image
            else:
                stft_images = np.concatenate((stft_images, single_channel_stft_image), axis=0)

        return stft_images.astype(np.float32), raw_sound_data, np.array([x, y]), np.array([sound_pressure_level])
