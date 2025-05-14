import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs=250, lowcut=0.5, highcut=40, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal, fs=250, notch_freq=50, quality=30):
    w0 = notch_freq / (fs / 2)
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal, axis=0)


source_dir = 'E:\AFprediction2025.03.05\AFprediction-JunMa\Database'  # 原分类数据所在目录
dest_dir = 'E:\AFprediction2025.03.05\AFprediction-JunMa\Database\DATA2_processed_filtered'  # 去噪后保存目录
os.makedirs(dest_dir, exist_ok=True)

subfolders = ['High_risk', 'Low_risk']
fs = 250

for subfolder in subfolders:
    input_path = os.path.join(source_dir, subfolder)
    output_path = os.path.join(dest_dir, subfolder)
    os.makedirs(output_path, exist_ok=True)

    for npy_file in glob.glob(os.path.join(input_path, '*.npy')):
        data = np.load(npy_file)

        # 先带通滤波，再进行陷波滤波去除工频干扰
        if data.ndim == 1:
            filtered_data = bandpass_filter(data, fs=fs)
            filtered_data = notch_filter(filtered_data, fs=fs, notch_freq=50, quality=30)
        else:
            # 多通道时对每个通道分别滤波
            filtered_data = []
            for ch in range(data.shape[1]):
                temp = bandpass_filter(data[:, ch], fs=fs)
                temp = notch_filter(temp, fs=fs, notch_freq=50, quality=30)
                filtered_data.append(temp)
            filtered_data = np.column_stack(filtered_data)

        output_file = os.path.join(output_path, os.path.basename(npy_file))
        np.save(output_file, filtered_data)
        print(f"去噪后数据已保存至: {output_file}")