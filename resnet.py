import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_model_summary import summary


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        r += self.shortcut(x)
        r = F.relu(r)
        return r


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels*4, 
                kernel_size=1, 
                stride=1, 
                padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*4)

        if stride != 1 or in_channels != out_channels*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=out_channels*4)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = F.relu(self.bn2(self.conv2(r)))
        r = self.bn3(self.conv3(r))
        r += self.shortcut(x)
        r = F.relu(r)
        return r


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            *[BasicBlock(in_channels=64, out_channels=64) for _ in range(2)]
        )

        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            *[BasicBlock(in_channels=128, out_channels=128) for _ in range(1, 2)]
        )

        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            *[BasicBlock(in_channels=256, out_channels=256) for _ in range(1, 2)]
        )

        self.conv5_x = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2),
            *[BasicBlock(in_channels=512, out_channels=512) for _ in range(1, 2)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


class ResNet34(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            *[BasicBlock(in_channels=64, out_channels=64) for _ in range(3)]
        )

        self.conv3_x = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            *[BasicBlock(in_channels=128, out_channels=128) for _ in range(1, 4)]
        )

        self.conv4_x = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            *[BasicBlock(in_channels=256, out_channels=256) for _ in range(1, 6)]
        )

        self.conv5_x = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2),
            *[BasicBlock(in_channels=512, out_channels=512) for _ in range(1, 3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


class ResNet50(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            BottleneckBlock(in_channels=64, out_channels=64),
            *[BottleneckBlock(in_channels=256, out_channels=64) for _ in range(1, 3)]
        )

        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=128, stride=2),
            *[BottleneckBlock(in_channels=512, out_channels=128) for _ in range(1, 4)]
        )

        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=256, stride=2),
            *[BottleneckBlock(in_channels=1024, out_channels=256) for _ in range(1, 6)]
        )

        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=512, stride=2),
            *[BottleneckBlock(in_channels=2048, out_channels=512) for _ in range(1, 3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


class ResNet101(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(ResNet101, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            BottleneckBlock(in_channels=64, out_channels=64),
            *[BottleneckBlock(in_channels=256, out_channels=64) for _ in range(1, 3)]
        )

        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=128, stride=2),
            *[BottleneckBlock(in_channels=512, out_channels=128) for _ in range(1, 4)]
        )

        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=256, stride=2),
            *[BottleneckBlock(in_channels=1024, out_channels=256) for _ in range(1, 23)]
        )

        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=512, stride=2),
            *[BottleneckBlock(in_channels=2048, out_channels=512) for _ in range(1, 3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


class ResNet152(nn.Module):
    def __init__(self, input_channels=3, input_height=224, input_width=224, num_classes=1000):
        super(ResNet152, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            BottleneckBlock(in_channels=64, out_channels=64),
            *[BottleneckBlock(in_channels=256, out_channels=64) for _ in range(1, 3)]
        )

        self.conv3_x = nn.Sequential(
            BottleneckBlock(in_channels=256, out_channels=128, stride=2),
            *[BottleneckBlock(in_channels=512, out_channels=128) for _ in range(1, 8)]
        )

        self.conv4_x = nn.Sequential(
            BottleneckBlock(in_channels=512, out_channels=256, stride=2),
            *[BottleneckBlock(in_channels=1024, out_channels=256) for _ in range(1, 36)]
        )

        self.conv5_x = nn.Sequential(
            BottleneckBlock(in_channels=1024, out_channels=512, stride=2),
            *[BottleneckBlock(in_channels=2048, out_channels=512) for _ in range(1, 3)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        r = self.conv1(x)
        r = self.bn1(r)
        r = F.relu(r)
        r = self.maxpool(r)
        r = self.conv2_x(r)
        r = self.conv3_x(r)
        r = self.conv4_x(r)
        r = self.conv5_x(r)
        r = self.avgpool(r)
        r = torch.flatten(r, 1)
        r = self.fc(r)
        return r


def resnet(num_layers, input_channels, input_height, input_width, num_classes):
    if num_layers == 18:
        return ResNet18(input_channels, input_height, input_width, num_classes)
    elif num_layers == 34:
        return ResNet34(input_channels, input_height, input_width, num_classes)
    elif num_layers == 50:
        return ResNet50(input_channels, input_height, input_width, num_classes)
    elif num_layers == 101:
        return ResNet101(input_channels, input_height, input_width, num_classes)
    elif num_layers == 152:
        return ResNet152(input_channels, input_height, input_width, num_classes)
    else:
        raise ValueError("Unsupported ResNet model. Supported models are: 18, 34, 50, 101, 152")


if __name__ == "__main__":
    inp = torch.randn(128, 3, 224, 224)
    model = resnet(num_layers=50, input_channels=3, input_height=224, input_width=224, num_classes=1000)
    print(summary(model, inp))
    #model = model.to('cuda')
    #inp = inp.to('cuda')
    #out = model(inp)

    
