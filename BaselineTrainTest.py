import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import random
import time
import argparse
from scipy import signal
import matplotlib.pyplot as plt
from Model_Definition import (
    SimpleConvEncoder,
    compute_hrv_features,
    MultiLayerLSTM
)
from BaselineModelDefinition import create_baseline_model


# 设置随机种子，确保可重复性
def set_seed(seed=42):
    """设置所有随机数生成器的种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)  # 固定随机种子


# 创建早停器
class EarlyStopping:
    """
    早停类，用于监控验证性能并在性能不再提升时停止训练

    参数:
    patience: 在触发早停前等待改进的最大轮数
    min_delta: 被视为改进的最小变化量
    mode: 'min'(监控指标越低越好) 或 'max'(监控指标越高越好)
    verbose: 是否打印早停相关信息
    """

    def __init__(self, patience=7, min_delta=0.0001, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_scores = []

        # 设置指标比较函数
        if self.mode == "min":
            self._is_better = lambda score, best: score <= (best - self.min_delta)
        else:
            self._is_better = lambda score, best: score >= (best + self.min_delta)

    def __call__(self, val_score, model, save_path):
        """
        评估当前验证分数是否需要触发早停

        参数:
        val_score: 当前验证指标
        model: 当前训练的模型
        save_path: 保存最佳模型的路径

        返回:
        True: 如果应该停止训练
        False: 如果应该继续训练
        """
        score = val_score
        self.val_scores.append(score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, save_path)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(val_score, model, save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, val_score, model, save_path):
        """保存模型当验证性能改善时"""
        if self.verbose:
            metric_name = "loss" if self.mode == "min" else "metric"
            print(
                f'Validation {metric_name} improved ({self.best_score:.6f} --> {val_score:.6f}). Saving model to {save_path}')
        torch.save(model.state_dict(), save_path)


# Focal Loss用于处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # 类别权重
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用交叉熵，保留每个样本的损失值
        if self.weight is not None:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算pt (预测目标类的概率)
        pt = torch.exp(-ce_loss)

        # Focal Loss公式
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 添加alpha权重（对正类样本的关注）
        if targets.dim() > 1:
            # 多标签情况
            focal_loss = focal_loss * (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        else:
            # 单标签情况
            batch_size = targets.size(0)
            alpha_tensor = torch.ones(batch_size, device=targets.device)
            alpha_tensor[targets == 1] = self.alpha  # 对正类使用alpha
            alpha_tensor[targets == 0] = 1 - self.alpha  # 对负类使用1-alpha
            focal_loss = focal_loss * alpha_tensor

        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_pretrained_weights(cnn_model, pretrained_path):
    """
    将预训练的权重加载到SimpleConvEncoder模型中
    """
    try:
        # 加载预训练模型权重
        pretrained_weights = torch.load(pretrained_path)

        # 提取编码器部分的权重
        encoder_weights = {k.replace('encoder.', ''): v for k, v in pretrained_weights.items()
                           if k.startswith('encoder.')}

        # 创建新的状态字典
        new_state_dict = {}
        model_dict = cnn_model.state_dict()

        # 将预训练的权重映射到对应的层
        for name, param in model_dict.items():
            if name in encoder_weights:
                new_state_dict[name] = encoder_weights[name]
            else:
                new_state_dict[name] = param

        # 载入处理后的权重
        cnn_model.load_state_dict(new_state_dict)
        print(f"成功加载预训练权重到SimpleConvEncoder")

    except Exception as e:
        print(f"加载预训练权重时出错: {e}")
        print("继续使用随机初始化的权重")

    return cnn_model


class ECGDataset(Dataset):
    def __init__(self,
                 root_dirs,
                 target_fs,
                 seq_len):
        """
        ECG数据集类，支持多个不同采样率的数据源

        参数:
        root_dirs: 包含不同数据源信息的字典，格式为 {
            "采样率标识": {
                "path": "数据路径",
                "use_first_10_min": 是否只使用前10分钟
            }
        }
        target_fs: 目标采样率，所有数据将重采样到这个采样率
        seq_len: 每个样本的序列长度（30秒片段的数量）
        """
        self.samples = []
        self.labels = []
        self.target_fs = target_fs
        self.seq_len = seq_len
        self.class_counts = {"Low_risk": 0, "High_risk": 0}

        subfolders = ["High_risk", "Low_risk"]

        # 处理每个数据源
        for dataset_key, dataset_info in root_dirs.items():
            root_dir = dataset_info["path"]
            use_first_10_min = dataset_info["use_first_10_min"]

            # 根据标识符获取原始采样率
            original_fs = int(dataset_key.replace("Hz", ""))

            print(f"处理数据集: {root_dir}，采样率: {original_fs}Hz")

            for sub in subfolders:
                label = 0 if sub == "Low_risk" else 1
                files = glob.glob(os.path.join(root_dir, sub, "*.npy"))
                print(f"  子文件夹 {sub}: 发现 {len(files)} 个文件")

                for f in files:
                    ecg = np.load(f)
                    if ecg.ndim > 1:
                        ecg = ecg[:, 0]

                    # 处理不同时长
                    if use_first_10_min:
                        # 只使用前10分钟
                        ecg = ecg[:10 * 60 * original_fs]
                    # 对于非use_first_10_min的数据集，使用全部数据（假定已经是10分钟）

                    # 如果原始采样率不是目标采样率，进行重采样
                    if original_fs != target_fs:
                        ecg = resample_ecg(ecg, original_fs, target_fs)

                    # 窗口大小（30秒对应的采样点数）
                    window_size = 30 * target_fs

                    # 如果长度不够分割成seq_len个片段，则跳过
                    if len(ecg) < window_size * self.seq_len:
                        continue

                    segments = []
                    for i in range(seq_len):
                        seg = ecg[i * window_size: (i + 1) * window_size]
                        segments.append(seg)

                    self.samples.append(np.array(segments))  # (seq_len, window_size)
                    self.labels.append(label)
                    self.class_counts[sub] += 1

        print(f"数据集统计: {self.class_counts}")

        # 计算更强的类别权重用于处理严重不平衡
        self.class_weights = None
        if self.class_counts["Low_risk"] != 0 and self.class_counts["High_risk"] != 0:
            # 使用更强的权重调整，对正类(High_risk)给予更高权重
            weight_low = 1.0
            weight_high = 2.5  # 提高正类权重
            self.class_weights = torch.tensor([weight_low, weight_high], dtype=torch.float)
            print(f"增强类别权重: Low_risk={weight_low}, High_risk={weight_high}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def collate_fn(batch):
    data_list, label_list = [], []
    for ecg_segments, lab in batch:
        data_list.append(ecg_segments)
        label_list.append(lab)
    return torch.tensor(data_list, dtype=torch.float), torch.tensor(label_list, dtype=torch.long)


# 添加一个函数用于对ECG信号进行重采样
def resample_ecg(ecg, original_fs, target_fs):
    """
    将ECG信号从原始采样率重采样到目标采样率
    """
    try:
        # 计算重采样后的样本数量
        num_samples = int(len(ecg) * target_fs / original_fs)
        # 重采样信号
        resampled = signal.resample(ecg, num_samples)
        return resampled
    except Exception as e:
        print(f"重采样信号时出错: {e}")
        return ecg  # 如果重采样失败，返回原始信号


# 基线漂移校正和信号增强
def enhance_signal(ecg, fs=250):
    """对ECG信号进行基线漂移校正和滤波增强"""
    try:
        from scipy import signal as sig

        # 高通滤波去除基线漂移
        b, a = sig.butter(3, 0.5 / (fs / 2), 'high')
        ecg_filtered = sig.filtfilt(b, a, ecg)

        # 带通滤波保留有用频率范围 (一般ECG在0.5-40Hz范围内)
        b, a = sig.butter(3, [0.5 / (fs / 2), 40 / (fs / 2)], 'band')
        ecg_filtered = sig.filtfilt(b, a, ecg_filtered)

        return ecg_filtered
    except Exception as e:
        print(f"信号增强出错: {e}")
        return ecg  # 出错时返回原始信号


def extract_features(dataloader, cnn_model, device, fs=250):
    """
    从ECG数据中提取HRV和CNN特征，包含改进的错误处理和信号预处理
    """
    X_all, Y_all = [], []
    error_count = 0

    for ecg_segments, labels in tqdm(dataloader, desc="提取特征"):
        ecg_segments = ecg_segments.to(device)  # (batch, seq_len, window_size)
        b, s, w = ecg_segments.shape
        feat_list = []

        for i in range(b):
            seq_feats = []
            for j in range(s):
                try:
                    # 获取ECG片段数据
                    seg_data = ecg_segments[i, j, :].cpu().numpy()

                    # 应用信号增强
                    seg_data = enhance_signal(seg_data, fs)

                    # 信号预处理 - 标准化
                    seg_mean = np.mean(seg_data)
                    seg_std = np.std(seg_data)
                    if seg_std > 1e-6:  # 防止除以0
                        seg_data = (seg_data - seg_mean) / seg_std

                    # 提取HRV特征
                    hrv_feats = compute_hrv_features(seg_data, fs=fs)  # 6维

                    # 检查HRV特征是否有异常值
                    if np.any(np.isnan(hrv_feats)) or np.any(np.isinf(hrv_feats)):
                        hrv_feats = np.zeros_like(hrv_feats)
                        error_count += 1

                    # CNN输入形状: (1, w, 1)
                    seg_tensor = torch.tensor(seg_data, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)
                    with torch.no_grad():
                        cnn_out = cnn_model(seg_tensor).cpu().numpy().flatten()  # 128维

                    # 检查CNN输出特征是否有异常值
                    if np.any(np.isnan(cnn_out)) or np.any(np.isinf(cnn_out)):
                        cnn_out = np.zeros_like(cnn_out)
                        error_count += 1

                    # 合并特征
                    combined = np.concatenate([hrv_feats, cnn_out])  # 6 + 128 = 134维
                    seq_feats.append(combined)

                except Exception as e:
                    print(f"处理片段时出错: {e}")
                    error_count += 1
                    # 出现异常时使用零向量
                    seq_feats.append(np.zeros(134))

            feat_list.append(seq_feats)  # (seq_len, 134)
        X_all.append(np.array(feat_list))  # (batch, seq_len, 134)
        Y_all.append(labels.cpu().numpy())

    if error_count > 0:
        print(f"警告: 特征提取过程中有 {error_count} 个片段出现错误")

    X_all = np.concatenate(X_all, axis=0)  # (N, seq_len, 134)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all


def evaluate_model(model, data_loader, device, criterion=None):
    """评估模型性能，返回各种评估指标"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for feats, labs in tqdm(data_loader, desc="评估", leave=False):
            feats = feats.to(device)
            labs = labs.to(device)
            logits = model(feats)

            if criterion is not None:
                loss = criterion(logits, labs)
                total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    # 计算各种指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    results = {
        'loss': total_loss if criterion is not None else None,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

    return results


def train_and_evaluate_kfold(dataset, cnn_model, device, n_splits=5, use_baseline=True):
    """
    使用k折交叉验证训练和评估模型，带有增强的早停机制

    参数:
    dataset: ECG数据集
    cnn_model: CNN模型用于特征提取
    device: 计算设备(CPU/GPU)
    n_splits: 交叉验证的折数
    use_baseline: 是否使用基线模型
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 准备标签用于分层采样
    labels = [dataset[i][1] for i in range(len(dataset))]

    fold_results = []
    best_models = []

    # 用于保存训练历史的字典
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
        'val_recall': [],
        'val_precision': []
    }

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 50}\n第 {fold + 1}/{n_splits} 折\n{'=' * 50}")

        # 创建训练和测试数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, batch_size=4, sampler=train_subsampler, collate_fn=collate_fn)
        test_loader = DataLoader(dataset, batch_size=4, sampler=test_subsampler, collate_fn=collate_fn)

        # 特征提取
        print("提取训练特征...")
        X_train, Y_train = extract_features(train_loader, cnn_model, device, fs=dataset.target_fs)
        print("提取测试特征...")
        X_test, Y_test = extract_features(test_loader, cnn_model, device, fs=dataset.target_fs)

        # 标准化 - 使用稳健的方法
        b_train, s_train, d = X_train.shape
        X_train_2d = X_train.reshape(b_train * s_train, d)
        scaler = StandardScaler()
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_train = X_train_2d.reshape(b_train, s_train, d)

        b_test, s_test, d_test = X_test.shape
        X_test_2d = X_test.reshape(b_test * s_test, d_test)
        X_test_2d = scaler.transform(X_test_2d)
        X_test = X_test_2d.reshape(b_test, s_test, d_test)

        # 拆分训练集和验证集，以便应用早停策略
        train_size = int(0.8 * len(X_train))
        val_size = len(X_train) - train_size

        train_indices = list(range(len(X_train)))
        np.random.shuffle(train_indices)

        train_indices, val_indices = train_indices[:train_size], train_indices[train_size:]

        X_train_final = X_train[train_indices]
        Y_train_final = Y_train[train_indices]
        X_val = X_train[val_indices]
        Y_val = Y_train[val_indices]

        print(f"训练集大小: {X_train_final.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")

        # 创建数据加载器
        train_feats = [(torch.tensor(X_train_final[i], dtype=torch.float), torch.tensor(Y_train_final[i])) for i in
                       range(len(X_train_final))]
        val_feats = [(torch.tensor(X_val[i], dtype=torch.float), torch.tensor(Y_val[i])) for i in range(len(X_val))]
        test_feats = [(torch.tensor(X_test[i], dtype=torch.float), torch.tensor(Y_test[i])) for i in range(b_test)]

        # 使用加权采样解决类别不平衡
        train_labels = Y_train_final
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        # 增加批大小以提高稳定性
        batch_size = 16
        train_feat_loader = DataLoader(train_feats, batch_size=batch_size, sampler=sampler)
        val_feat_loader = DataLoader(val_feats, batch_size=batch_size, shuffle=False)
        test_feat_loader = DataLoader(test_feats, batch_size=batch_size, shuffle=False)

        # 创建模型 - 使用基线模型(默认双向LSTM)
        if use_baseline:
            # 创建基线双向LSTM模型，增加dropout进行正则化
            model = create_baseline_model(
                input_dim=134,  # 128 CNN特征 + 6 HRV特征
                hidden_dim=128,
                num_layers=2,
                num_classes=2,
                dropout_rate=0.3  # 增加dropout抑制过拟合
            ).to(device)
            print("使用双向LSTM基线模型，增加dropout")
        else:
            # 使用原始模型
            model = MultiLayerLSTM(input_dim=134, hidden_dim=128, num_layers=2, num_classes=2).to(device)
            print("使用原始模型架构")

        # 设置焦点损失和类别权重
        class_weights = dataset.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
            criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
            print("使用焦点损失(Focal Loss)和增强的类别权重")
        else:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print("使用焦点损失(Focal Loss)，无类别权重")

        # 优化器和学习率调度 - 增加权重衰减(weight decay)进行L2正则化
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)  # 增加weight_decay

        # 使用带预热的学习率调度器
        from torch.optim.lr_scheduler import OneCycleLR
        total_steps = 40 * len(train_feat_loader)  # 假设最多40轮
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            total_steps=total_steps,
            pct_start=0.3,  # 30%的时间用于预热
            div_factor=25,  # 初始学习率 = max_lr/div_factor
            final_div_factor=1000,  # 最终学习率 = max_lr/final_div_factor
            anneal_strategy='cos'  # 余弦退火
        )

        # 创建早停对象，监控F1分数
        early_stopping = EarlyStopping(
            patience=12,  # 增加耐心值
            min_delta=0.001,  # 最小改进值
            mode='max',  # 监控最大化F1值
            verbose=True
        )

        # 训练参数
        epochs = 50  # 设置更大的最大轮数，让早停决定何时停止
        best_model_path = f'best_model_fold{fold + 1}_baseline_enhanced.pth'

        # 用于记录当前折的训练历史
        fold_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_f1': [],
            'val_acc': [],
            'val_recall': [],
            'val_precision': []
        }

        # 训练循环
        for epoch in range(epochs):
            # 训练
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            batch_count = 0

            start_time = time.time()
            for feats, labs in tqdm(train_feat_loader, desc=f"训练 Epoch {epoch + 1}/{epochs}", leave=False):
                feats = feats.to(device)
                labs = labs.to(device)

                optimizer.zero_grad()
                logits = model(feats)
                loss = criterion(logits, labs)

                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

                # 更新学习率
                scheduler.step()

                # 计算训练准确率
                _, predicted = torch.max(logits, 1)
                total += labs.size(0)
                correct += (predicted == labs).sum().item()

            train_loss = total_loss / batch_count
            train_acc = correct / total
            epoch_time = time.time() - start_time

            # 验证
            model.eval()
            val_results = evaluate_model(model, val_feat_loader, device, criterion)

            # 记录训练和验证指标
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_results['loss'])
            fold_history['val_f1'].append(val_results['f1'])
            fold_history['val_acc'].append(val_results['accuracy'])
            fold_history['val_recall'].append(val_results['recall'])
            fold_history['val_precision'].append(val_results['precision'])

            # 打印当前轮的训练结果
            print(f"Epoch {epoch + 1}, Time: {epoch_time:.2f}s")
            print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(
                f"  验证: Loss={val_results['loss']:.4f}, F1={val_results['f1']:.4f}, Acc={val_results['accuracy']:.4f}")
            print(f"  验证: Precision={val_results['precision']:.4f}, Recall={val_results['recall']:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")

            # 早停检查 - 使用F1分数作为早停指标
            if early_stopping(val_results['f1'], model, best_model_path):
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break

        # 绘制训练过程中的指标变化
        plt.figure(figsize=(15, 10))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(fold_history['train_loss'], label='Train Loss')
        plt.plot(fold_history['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold + 1} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(fold_history['train_acc'], label='Train Acc')
        plt.plot(fold_history['val_acc'], label='Val Acc')
        plt.title(f'Fold {fold + 1} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # 绘制F1分数曲线
        plt.subplot(2, 2, 3)
        plt.plot(fold_history['val_f1'], label='F1 Score')
        plt.title(f'Fold {fold + 1} F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend()
        plt.grid(True)

        # 绘制精确率和召回率曲线
        plt.subplot(2, 2, 4)
        plt.plot(fold_history['val_precision'], label='Precision')
        plt.plot(fold_history['val_recall'], label='Recall')
        plt.title(f'Fold {fold + 1} Precision & Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'fold_{fold + 1}_training_history.png')
        plt.close()

        # 使用最佳模型进行最终评估
        try:
            print("载入最佳模型进行最终评估...")
            model.load_state_dict(torch.load(best_model_path))

            # 在测试集上评估
            final_results = evaluate_model(model, test_feat_loader, device)
            print("\n最终测试集评估结果:")
            print(f"  准确率: {final_results['accuracy']:.4f}")
            print(f"  精确率: {final_results['precision']:.4f}")
            print(f"  召回率: {final_results['recall']:.4f}")
            print(f"  F1分数: {final_results['f1']:.4f}")
            print(f"  混淆矩阵:\n{final_results['confusion_matrix']}")

            fold_results.append(final_results)
            best_models.append({
                'state_dict': model.state_dict(),
                'fold': fold,
                'f1': final_results['f1']
            })

            # 合并当前折的历史数据到总历史数据
            for key in history:
                if len(history[key]) <= fold:
                    history[key].append(fold_history[key])
                else:
                    history[key][fold] = fold_history[key]

        except Exception as e:
            print(f"加载最佳模型时出错: {e}")

    # 计算所有折的平均性能
    print("\n" + "=" * 50)
    print("Cross-Validation Results:")
    print("=" * 50)

    avg_acc = np.mean([res['accuracy'] for res in fold_results])
    avg_precision = np.mean([res['precision'] for res in fold_results])
    avg_recall = np.mean([res['recall'] for res in fold_results])
    avg_f1 = np.mean([res['f1'] for res in fold_results])

    std_acc = np.std([res['accuracy'] for res in fold_results])
    std_precision = np.std([res['precision'] for res in fold_results])
    std_recall = np.std([res['recall'] for res in fold_results])
    std_f1 = np.std([res['f1'] for res in fold_results])

    print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")

    # 保存性能最好的模型
    best_idx = np.argmax([model['f1'] for model in best_models])
    best_model = best_models[best_idx]
    model_type = "baseline_enhanced"
    torch.save({
        'state_dict': best_model['state_dict'],
        'fold': best_model['fold'],
        'f1': best_model['f1'],
        'avg_metrics': {
            'accuracy': avg_acc,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }
    }, f'best_model_final_{model_type}.pth')
    print(f"Best model saved (from fold {best_model['fold'] + 1}, F1: {best_model['f1']:.4f})")

    return fold_results, best_models, history


def evaluate_with_varying_windows(model, data_loader, device, max_windows=20):
    """
    评估使用不同数量窗口(从1到max_windows)时的模型性能

    参数:
    model: 训练好的LSTM模型
    data_loader: 测试数据加载器
    device: 设备(CPU/GPU)
    max_windows: 最大窗口数量(默认为20)

    返回:
    window_metrics: 包含每个窗口数量对应指标的字典
    """
    model.eval()

    # 用于存储不同窗口数量的评估结果
    window_metrics = {
        'window_counts': list(range(1, max_windows + 1)),
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # 收集所有测试数据
    all_features = []
    all_true_labels = []

    for feats, labs in tqdm(data_loader, desc="收集测试数据"):
        all_features.append(feats)
        all_true_labels.extend(labs.numpy())

    # 合并特征数据
    all_features = torch.cat(all_features, dim=0)
    all_true_labels = np.array(all_true_labels)

    # 检查最大窗口数
    actual_max_windows = all_features.shape[1]
    max_windows = min(max_windows, actual_max_windows)
    window_metrics['window_counts'] = list(range(1, max_windows + 1))

    # 对每个窗口数量进行评估
    for num_windows in range(1, max_windows + 1):
        predictions = []

        # 由于可能数据量大，分批处理
        batch_size = 32
        num_samples = all_features.shape[0]

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_feats = all_features[i:end_idx, :num_windows, :].to(device)

            with torch.no_grad():
                # 预测
                logits = model(batch_feats)
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())

        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_true = all_true_labels
        y_pred = np.array(predictions)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        # 存储结果
        window_metrics['accuracy'].append(accuracy)
        window_metrics['precision'].append(precision)
        window_metrics['recall'].append(recall)
        window_metrics['f1'].append(f1)

        print(
            f"Windows: {num_windows}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return window_metrics


def plot_window_metrics(window_metrics, save_path='window_metrics.png'):
    """
    绘制不同窗口数量下的性能指标折线图

    参数:
    window_metrics: 由evaluate_with_varying_windows返回的指标字典
    save_path: 图像保存路径
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(12, 8))

    # 绘制四条折线
    plt.plot(window_metrics['window_counts'], window_metrics['accuracy'], marker='o', label='Accuracy')
    plt.plot(window_metrics['window_counts'], window_metrics['precision'], marker='s', label='Precision')
    plt.plot(window_metrics['window_counts'], window_metrics['recall'], marker='^', label='Recall')
    plt.plot(window_metrics['window_counts'], window_metrics['f1'], marker='d', label='F1 Score')

    plt.title('Model Performance with Different Window Counts', fontsize=16)
    plt.xlabel('Number of 30-second Windows', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.xticks(window_metrics['window_counts'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 添加数据标签
    for i, acc in enumerate(window_metrics['accuracy']):
        plt.text(window_metrics['window_counts'][i], acc + 0.01, f'{acc:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Chart saved to: {save_path}")


def evaluate_window_impact(dataset, cnn_model, best_models, device):
    """
    评估窗口数量对模型性能的影响
    """
    print("\n" + "=" * 50)
    print("Performing window count impact analysis")
    print("=" * 50)

    # 选择性能最好的模型
    best_idx = np.argmax([model['f1'] for model in best_models])
    best_model_data = best_models[best_idx]
    print(f"Using best model from fold {best_model_data['fold'] + 1} (F1={best_model_data['f1']:.4f}) for analysis")

    # 准备测试集（使用10%的数据）
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 创建并加载模型
    model = create_baseline_model(input_dim=134, hidden_dim=128, num_layers=2, num_classes=2, dropout_rate=0.3).to(
        device)
    model.load_state_dict(best_model_data['state_dict'])
    model.eval()

    # 提取测试特征
    print("Extracting test features...")
    X_test, Y_test = extract_features(test_loader, cnn_model, device, fs=dataset.target_fs)

    # 标准化特征
    b_test, s_test, d_test = X_test.shape
    X_test_2d = X_test.reshape(b_test * s_test, d_test)
    scaler = StandardScaler()
    X_test_2d = scaler.fit_transform(X_test_2d)
    X_test = X_test_2d.reshape(b_test, s_test, d_test)

    # 创建特征数据集和加载器
    test_feats = [(torch.tensor(X_test[i], dtype=torch.float), torch.tensor(Y_test[i])) for i in range(b_test)]
    test_feat_loader = DataLoader(test_feats, batch_size=16, shuffle=False)

    # 执行窗口分析
    window_metrics = evaluate_with_varying_windows(model, test_feat_loader, device, max_windows=min(20, s_test))

    # 绘制结果
    plot_window_metrics(window_metrics, save_path='window_analysis_result_enhanced.png')

    return window_metrics


def main():
    # 设置种子确保可重复性
    set_seed(42)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train AF prediction model with enhanced early stopping')
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建CNN模型
    cnn_model = SimpleConvEncoder(feature_dim=128).to(device)

    # 预训练权重路径
    pretrained_path = "E:\AFprediction2025.04.13xiaorong\Combination_Training\signal_masked\masked_autoencoder.pth"

    # 加载预训练权重
    if os.path.exists(pretrained_path):
        print(f"Found pretrained weights: {pretrained_path}")
        cnn_model = load_pretrained_weights(cnn_model, pretrained_path)
    else:
        print(f"Pretrained weights not found: {pretrained_path}, using random initialization")

    # 加载数据集
    dataset = ECGDataset(
        root_dirs={
            "250Hz": {
                "path": "E:\AFprediction2025.04.13xiaorong\Combination_Training\data_augmentation\DATA2_augmented_2x",
                "use_first_10_min": True
            },
            "128Hz": {
                "path": "E:\AFprediction2025.04.13xiaorong\Combination_Training\data_augmentation\DATA3_augmented_2x",
                "use_first_10_min": False
            }
        },
        target_fs=250,  # 所有数据将重采样到250Hz
        seq_len=20
    )

    # 使用k折交叉验证，使用增强的早停机制
    print("Using enhanced bidirectional LSTM baseline model with improved early stopping")
    fold_results, best_models, history = train_and_evaluate_kfold(
        dataset, cnn_model, device, n_splits=5, use_baseline=True
    )

    # 分析窗口数量对性能的影响
    window_metrics = evaluate_window_impact(dataset, cnn_model, best_models, device)

    print("\nAnalysis complete! Results saved to 'window_analysis_result_enhanced.png'")


if __name__ == "__main__":
    main()