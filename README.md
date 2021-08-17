# Acoustic-Net: A Novel Neural Network for Sound Localization and Quantification [APVC 2021]



## Getting Started

- python = 3.6
- pytorch >= 1.1.0
- acoular

## Preparation

- Beamforming Dataset

  The Beamforming Dataset can be downloaded from the following link:

   https://drive.google.com/file/d/1wPeOIcgcrq52-LQXwKE1-VQCMUrIBZuw/view?usp=sharing .

  This dataset contains 4200 sound source samples with 2400 points for training, 800 points for validation and 1000 points for testing.  
  
  
  
- Data Prerprocessing

  To run our tasks,

  1. You need to add  the train, val, test set into a new folder named "`data_dir`". Then, you can get 2400 h5 files about acoustic sound data collected by 56 microphones. 

  2. Take the point `"x_0.00_y_-0.06_rms_0.69_sources.h5"` for an example. You can run the `./DET/stft.m` to change the original sound data into the 56 grey images, and save the 56 images in the folder named `"x_0.00_y_-0.06_rms_0.69_sources.h5"`. 
  3. To be able to use our dataloader (`./dataset/dataset.py`);
     - Each sample folder should contain 56  grey images and the original sound data as the Acoustic-Net input. For details please refer to `./dataset/dataset.py`
  
  

## Training 

```shell
python repvgg_with_sound_pressure_and_location.py --data_dir data_path \
--train_image_dir train_stft_image_path \
--val_image_dir val_stft_image_path \
--model_dir save_model_path
```

The parameters such as `"epoch, batchsize and train_dir" ` you can add them in the shell command.

## Pretrained Model

We provide pre-trained model for the Acoustic-Net architecture. You can download the model from here to reproduce the results.
