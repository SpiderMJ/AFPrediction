import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

# 从简化模型定义中导入编码器和解码器
from Model_Definition_masked import SimpleConvEncoder, SimpleConvDecoder


class MaskedECGDataset(Dataset):
    def __init__(self, root_dirs, target_fs, num_segments, max_files):

        self.segments = []
        self.target_fs = target_fs
        window_size = 30 * target_fs  # 30秒窗口

        # 处理多个数据集
        for dataset_name, dataset_info in root_dirs.items():
            path = dataset_info["path"]
            use_first_10_min = dataset_info["use_first_10_min"]

            # 确定采样率
            source_fs = int(dataset_name.replace("Hz", ""))

            # 获取High_risk文件夹的路径
            folder = os.path.join(path, "High_risk")
            files = glob.glob(os.path.join(folder, "*.npy"))
            print(f"[DEBUG] Found {len(files)} files in folder: {folder}")

            files = files[:max_files]  # 只加载前 max_files 个文件

            for f in files:
                ecg = np.load(f)
                if ecg.ndim > 1:
                    ecg = ecg[:, 0]

                # 根据参数决定是否只取前10分钟
                if use_first_10_min:
                    ecg = ecg[:10 * 60 * source_fs]

                # 如果源采样率与目标采样率不同，进行重采样
                if source_fs != target_fs:
                    # 计算重采样前后的长度比例
                    resampled_length = int(len(ecg) * (target_fs / source_fs))
                    ecg = signal.resample(ecg, resampled_length)

                # 计算重采样后的窗口大小
                for i in range(num_segments):
                    if (i + 1) * window_size <= len(ecg):
                        seg = ecg[i * window_size: (i + 1) * window_size].astype(np.float32)
                        mean = np.mean(seg)
                        std = np.std(seg)
                        seg = (seg - mean) / std if std > 0 else seg - mean
                        self.segments.append(seg)

        print(f"[DEBUG] Total segments loaded: {len(self.segments)}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        original = self.segments[idx]
        masked = original.copy()
        num_mask = int(len(original) * 0.2)
        mask_indices = np.random.choice(len(original), num_mask, replace=False)
        masked[mask_indices] = 0.0
        return torch.tensor(masked).unsqueeze(-1), torch.tensor(original)


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)  # (batch, feature_dim, 625)
        reconstruction = self.decoder(features)  # (batch, 2500)
        return reconstruction


def main():
    print("[DEBUG] Starting main()")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    dataset = MaskedECGDataset(
        root_dirs={
            "250Hz": {
                "path": r"E:\AFprediction2025.04.13xiaorong\Combination_Training\Database\DATA2_processed_filtered",
                "use_first_10_min": True
            },
            "128Hz": {
                "path": r"E:\AFprediction2025.04.13xiaorong\Combination_Training\Database\DATA3_processed_filtered",
                "use_first_10_min": False
            }
        },
        target_fs=250,
        num_segments=10,
        max_files=5
    )
    if len(dataset) == 0:
        print("[ERROR] No segments loaded. Check your dataset path or file format.")
        return
    else:
        print(f"[DEBUG] Dataset size: {len(dataset)} segments")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    total_batches = len(dataloader)
    print(f"[DEBUG] DataLoader created with {total_batches} batches.")

    encoder = SimpleConvEncoder(feature_dim=128).to(device)
    decoder = SimpleConvDecoder(feature_dim=128, out_length=30 * 250).to(device)
    model = MaskedAutoencoder(encoder, decoder).to(device)
    print("[DEBUG] Model created.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30  # 测试用少量epoch

    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (masked, original) in progress_bar:
            batch_start = time.time()
            masked, original = masked.to(device), original.to(device)
            optimizer.zero_grad()
            output = model(masked)
            loss = criterion(output, original)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * masked.size(0)
            progress_bar.set_postfix(loss=loss.item(), batch_time=f"{time.time() - batch_start:.2f}s")
        avg_loss = total_loss / len(dataset)
        epoch_time = time.time() - epoch_start
        print(f"[DEBUG] Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    print("[DEBUG] Training complete.")

    # -------------------- 保存模型权重 --------------------
    torch.save(model.state_dict(), "masked_autoencoder.pth")
    print("[DEBUG] Model weights saved to masked_autoencoder.pth")

    model.eval()
    with torch.no_grad():
        masked_example, original_example = dataset[0]
        print("[DEBUG] Visualizing sample index 0")
        masked_example = masked_example.unsqueeze(0).to(device)
        reconstructed_example = model(masked_example).cpu().numpy().flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(original_example.numpy(), label="Original")
    plt.plot(reconstructed_example, label="Reconstruction", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.title("Original vs. Reconstruction")
    plt.legend()
    plt.show()
    print("[DEBUG] Plot displayed.")


if __name__ == "__main__":
    main()