from torch import nn


# Only Use CNN to train the sound pressure level
class Sound_Pressure_CNN(nn.Module):
    def __init__(self):
        super(Sound_Pressure_CNN, self).__init__()

        self.sound_conv1 = nn.Sequential(nn.Conv2d(in_channels=56, out_channels=32, kernel_size=(4, 1), stride=2),
                                         nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU())

        self.sound_conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=2),
                                         nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU())

        self.sound_conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=2),
                                         nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU())

        self.fc1 = nn.Linear(4064, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        # self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.sound_conv1(x)
        x = self.sound_conv2(x)
        x = self.sound_conv3(x)
        # print("%%%%%%%%%%%%%%%%%1", x.shape)
        x = x.view(x.size(0), -1)
        # print("%%%%%%%%%%%%%%%%%2", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
