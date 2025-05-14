import torch
import torch.nn as nn

class SimpleConvEncoder(nn.Module):
    def __init__(self, input_channels=1, feature_dim=128):
        super(SimpleConvEncoder, self).__init__()
        # 输入形状：(batch, 1, 2500)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # -> (batch, 32, 2500)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)  # -> (batch, 64, 1250)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, feature_dim, kernel_size=3, stride=2, padding=1)  # -> (batch, 128, 625)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        # x: (batch, timesteps, channels) -> 转换为 (batch, channels, timesteps)
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class SimpleConvDecoder(nn.Module):
    def __init__(self, feature_dim=128, out_length=2500):
        super(SimpleConvDecoder, self).__init__()
        # 从 (batch, 128, 625) -> 通过转置卷积还原到 (batch, 1, 2500)
        self.deconv1 = nn.ConvTranspose1d(feature_dim, 64, kernel_size=4, stride=2, padding=1)  # -> (batch, 64, 1250)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)  # -> (batch, 1, 2500)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.deconv2(x)
        x = x.squeeze(1)  # 输出 (batch, 2500)
        return x
