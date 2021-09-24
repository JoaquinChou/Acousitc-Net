# Acoustic-Net: A Novel Neural Network for Sound Localization and Quantification [APVC 2021]

The Acoustic-Net is designed to extract the features of amplitude change and phase difference by using raw sound data. Our proposed method is illustrated in this figure.

<img src="https://z3.ax1x.com/2021/09/23/40Fene.png" alt="40Fene.png" border="0" />

An Acoustic-Net is proposed to locate and quantify the acoustic source without reconstructing images and being restricted by CB map. The acoustic-Net, with RepVGG-B0 and shallow onedimensional convolution, is used to extract the amplitude and phase characteristics of original sound signals. The proposed method can be easily realized by the hardware devices with fast computing speed and limited memory utilization. The system is implemented in Pytorch.

## Getting Started

- python = 3.6
- pytorch >= 1.1.0
- acoular

## Preparation

- Beamforming Dataset

  The Beamforming Dataset can be downloaded from the following link:

   https://drive.google.com/file/d/1wPeOIcgcrq52-LQXwKE1-VQCMUrIBZuw/view?usp=sharing.

  This dataset contains 4200 sound source samples with 2400 points for training, 800 points for validation and 1000 points for testing.  
  
  
  
- Data Prerprocessing

  To run our tasks,

  1. You need to add  the train, val, test set (remove their folders) into a new folder named "`data_dir`". Then, you can get 4200 h5 files about acoustic sound data collected by 56 microphones. 

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



## Converted Model

After training, you can convert the model  by running:

```shell
cd utils
python convert_models.py --weights your_model_path convert_weights convert_model_path
```

 

## Pretrained Model

We provide pre-trained model for the Acoustic-Net architecture. You can download the model from `./Acousitc-Net/PretrainedModel/` to reproduce the results.

## Testing 

- Original Model

if you don't want to convert the training model, you can use the following codes to test your model.

```shell
python test_sound_source_pressure_and_location.py --data_dir data_path \
--test_image_dir your test_stft_image_path \
--result_dir save_visualization \
--weights your_model_path
```



- Converted model

if you finish converting the model, you can use the following codes to test your model.

```shell
python test_sound_source_pressure_and_location.py --data_dir data_path \
--test_image_dir your test_stft_image_path \
--result_dir save_visualization \
--mode deploy \
--convert_weights your_convert_model_path
```



## Citation
