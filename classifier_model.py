from torch import nn

CLASS_COUNT = 22


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.relu = nn.ReLU()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        outputs = self.conv_layers(x)
        res = x
        if self.downsample is not None:
            res = self.downsample(res)

        outputs += res
        outputs = self.relu(outputs)
        return outputs


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2),
                nn.BatchNorm2d(128)
            )),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2),
                nn.BatchNorm2d(256)
            )),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            ResidualBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2),
                nn.BatchNorm2d(512)
            )),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1 * 1 * 512, CLASS_COUNT)
        )

        self.training_metrics_history = list()
        self.validation_metrics_history = list()

    def forward(self, x):
        outputs = self.layers(x)
        return outputs

