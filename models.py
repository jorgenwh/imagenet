import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out(input_size, kernel_size, stride, padding):
    return int((input_size - kernel_size + 2*padding)/stride + 1)

def conv_tower_out(input_size, params: list[list[int]]):
    size = input_size
    for kernel_size, stride, padding in params:
        size = conv_out(size, kernel_size, stride, padding)
    return size


class AlexNet(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(AlexNet, self).__init__()

        self.conv_out_height = conv_tower_out(input_height, 
                [ [11, 4, 2], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0] ]
        )
        self.conv_out_width = conv_tower_out(input_width, 
                [ [11, 4, 2], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0] ]
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256*self.conv_out_height*self.conv_out_width, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*self.conv_out_height*self.conv_out_width)
        x = self.classifier(x)
        return x

