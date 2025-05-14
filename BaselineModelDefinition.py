import torch
import torch.nn as nn
import numpy as np


class AFPredictionBaseline(nn.Module):
    """
    基线模型用于心房颤动(AF)预测，基于论文架构
    使用双向LSTM处理ECG特征序列
    """

    def __init__(self, input_size=134, hidden_size=128, num_layers=2, num_classes=2, dropout_rate=0.3):
        super(AFPredictionBaseline, self).__init__()

        # 双层双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # 增加dropout减少过拟合风险
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层用于分类
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # 批归一化层
        self.bn = nn.BatchNorm1d(hidden_size)

        # 温度缩放参数
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float))

        # 初始化参数 - 使用更安全的方法
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重，针对不同类型的层使用适当的初始化方法"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重不做自定义初始化，PyTorch已经有适当的默认初始化
                    continue
                elif 'fc' in name and param.dim() >= 2:
                    # 线性层权重使用Xavier初始化
                    nn.init.xavier_normal_(param)
                elif 'bn' in name:
                    # BatchNorm层权重使用常值初始化
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                # 所有偏置初始化为0
                nn.init.zeros_(param)

    def forward(self, x):
        # 通过LSTM处理序列数据
        out, _ = self.lstm(x)

        # 对时序维度进行平均池化
        out_pooled = torch.mean(out, dim=1)

        # 全连接层和激活函数
        out = self.fc1(out_pooled)
        out = self.relu(out)

        # 批归一化
        out = self.bn(out)

        # Dropout层
        out = self.dropout(out)

        # 最终分类层
        logits = self.fc2(out)

        # 应用温度缩放进行校准
        logits = logits / self.temperature.clamp(min=1e-6)

        return logits


def create_baseline_model(input_dim=134, hidden_dim=128, num_layers=2, num_classes=2, dropout_rate=0.3):
    """创建基线模型实例的辅助函数"""
    model = AFPredictionBaseline(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model