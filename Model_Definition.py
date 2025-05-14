import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks, welch


########################################
# 1. 简单CNN：替换原多分支CNN
########################################
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
        # 添加全局池化以获得与MultiBranchCNN相同的输出维度
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, timesteps, channels) -> 转换为 (batch, channels, timesteps)
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # 添加全局池化，获得 (batch, feature_dim, 1)
        x = self.global_pool(x)
        # 去除最后一个维度，得到 (batch, feature_dim)
        x = x.squeeze(-1)
        return x


########################################
# 2. HRV特征（示例：增加R峰检测、更多指标）
########################################
def compute_hrv_features(ecg, fs=250):
    # 标准化信号以提高R峰检测准确性
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)

    # 使用更强参数寻找R峰
    peaks, _ = find_peaks(ecg_norm,
                          height=0.4,  # 增加高度阈值
                          distance=int(0.25 * fs),  # 最小RR间隔
                          prominence=0.3)  # 添加波峰突出度参数

    if len(peaks) < 4:  # 需要足够的R峰才能计算有意义的HRV指标
        return [0, 0, 0, 0, 0, 0]

    # 计算RR间期(毫秒)
    rr = np.diff(peaks) / fs * 1000

    # 排除异常值（可能的错误检测）
    rr_filtered = rr[(rr > 300) & (rr < 2000)]  # 只保留合理范围内的RR间期

    if len(rr_filtered) < 3:
        return [0, 0, 0, 0, 0, 0]

    # 计算标准HRV指标
    mean_rr = np.mean(rr_filtered)
    sdnn = np.std(rr_filtered)
    rmssd = np.sqrt(np.mean(np.diff(rr_filtered) ** 2))
    nn50 = np.sum(np.abs(np.diff(rr_filtered)) > 50)
    pnn50 = nn50 / len(rr_filtered) if len(rr_filtered) > 0 else 0

    # 频域HRV
    if len(rr_filtered) >= 4:
        rri_times = np.cumsum(rr_filtered) / 1000
        fs_interp = 4
        time_interp = np.linspace(rri_times[0], rri_times[-1], num=len(rr_filtered) * fs_interp)
        rr_interp = np.interp(time_interp, rri_times, rr_filtered)
        f, pxx = welch(rr_interp, fs=fs_interp)
        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.4)
        lf_power = np.trapz(pxx[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(pxx[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0
        lf_hf = lf_power / hf_power if hf_power > 0 else 0
    else:
        lf_hf = 0

    return [mean_rr, sdnn, rmssd, nn50, pnn50, lf_hf]


########################################
# 3. 多层RNN + 池化 + 校准（Temperature Scaling）
########################################
class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim=134, hidden_dim=128, num_layers=2, num_classes=2):
        """
        - input_dim=134：6维HRV + 128维CNN输出
        - hidden_dim=128
        - num_layers=2：双层LSTM
        - num_classes=2：二分类
        """
        super(MultiLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # 时序池化（这里采用平均池化，也可用最大池化/注意力等）
        self.pool = nn.AdaptiveAvgPool2d((1, hidden_dim))  # 针对 (batch, seq_len, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # 校准：温度缩放
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        out, (h, c) = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        # 对时序维度进行平均池化 -> (batch, 1, hidden_dim)
        # out_pooled = self.pool(out.unsqueeze(1)).squeeze(1)  # (batch, hidden_dim)
        out_pooled = out.mean(dim=1)
        logits = self.fc(out_pooled)  # (batch, num_classes)
        # 校准：logits / temperature
        logits = logits / self.temperature.clamp(min=1e-6)
        return logits


# 添加到 Model_Definition.py 文件中

class AttentionLayer(nn.Module):
    """
    注意力层: 对LSTM输出的每个时间步分配权重
    """

    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # 注意力向量、矩阵和偏置项
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        """
        参数:
            lstm_output: [batch_size, seq_len, hidden_dim]
        返回:
            context: [batch_size, hidden_dim]
        """
        # lstm_output: [batch_size, seq_len, hidden_dim]
        # attention_weights: [batch_size, seq_len, 1]
        attention_weights = self.attention(lstm_output)

        # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # [batch_size, seq_len, hidden_dim] * [batch_size, seq_len, 1] = [batch_size, seq_len, hidden_dim]
        context = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_dim]

        return context, attention_weights


class MultiLayerLSTMWithAttention(nn.Module):
    """
    带有注意力机制的多层LSTM模型
    """

    def __init__(self, input_dim=134, hidden_dim=128, num_layers=2, num_classes=2):
        super(MultiLayerLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # 添加注意力层
        self.attention = AttentionLayer(hidden_dim)

        # 分类层
        self.fc = nn.Linear(hidden_dim, num_classes)

        # 校准：温度缩放
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # x: [batch_size, seq_len, input_dim]
        out, (h, c) = self.lstm(x)  # out: [batch_size, seq_len, hidden_dim]

        # 应用注意力机制
        context, attention_weights = self.attention(out)  # context: [batch_size, hidden_dim]

        # 分类
        logits = self.fc(context)  # [batch_size, num_classes]

        # 校准：logits / temperature
        logits = logits / self.temperature.clamp(min=1e-6)

        return logits, attention_weights