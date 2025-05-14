import wfdb
import os
import glob
import numpy as np
import hashlib

folder = 'E:\AFprediction2025.03.05\pythonProject1\database\DATA2\\files'  # 替换为ECG记录所在文件夹路径
save_folder_afib = 'E:\AFprediction2025.03.05\pythonProject1\database\DATA2_processed\High_risk'  # 替换为保存AFIB片段的文件夹路径
save_folder_normal = 'E:\AFprediction2025.03.05\pythonProject1\database\DATA2_processed\Low_risk'  # 替换为保存正常片段的文件夹路径
os.makedirs(save_folder_afib, exist_ok=True)
os.makedirs(save_folder_normal, exist_ok=True)

atr_files = glob.glob(os.path.join(folder, '*.atr'))

for atr_file in atr_files:
    base = os.path.splitext(os.path.basename(atr_file))[0]
    record_path = os.path.join(folder, base)
    try:
        ann = wfdb.rdann(record_path, 'atr')
        sig, _ = wfdb.rdsamp(record_path)
    except Exception as e:
        print(f"读取 {record_path} 时出错: {e}")
        continue

    total_samples = sig.shape[0]
    segment_length = 30 * 60 * 250  # 30分钟 = 450000采样点
    first_part = 10 * 60 * 250  # 前10分钟 = 150000采样点

    # 提取所有包含"AFIB"的注释采样点
    afib_indices = [ann.sample[i] for i, note in enumerate(ann.aux_note) if 'AFIB' in note]
    afib_indices = np.unique(np.array(afib_indices))

    afib_segments = []
    normal_segments = []

    # 非重叠分段扫描整条记录
    for start in range(0, total_samples - segment_length + 1, segment_length):
        end = start + segment_length
        segment = sig[start:end]

        # 条件1：前10分钟内无AFIB，后20分钟至少有一次AFIB注释
        if not np.any((afib_indices >= start) & (afib_indices < start + first_part)) and \
                np.any((afib_indices >= start + first_part) & (afib_indices < end)):
            afib_segments.append((start, end, segment))

        # 条件2：正常片段——整段无AFIB注释
        if not np.any((afib_indices >= start) & (afib_indices < end)):
            # 使用连续差分的标准差作为噪声指标，数值越小说明干扰越小
            noise_metric = np.std(np.diff(segment, axis=0))
            normal_segments.append((start, end, segment, noise_metric))

    # 对正常片段按起始时间排序
    afib_segments.sort(key=lambda x: x[0])
    normal_segments.sort(key=lambda x: x[0])

    # 配对数量取较小者
    num_pairs = min(len(afib_segments), len(normal_segments))
    for i in range(num_pairs):
        afib_seg = afib_segments[i]
        normal_seg = normal_segments[i]

        afib_filename = f"{base}_{afib_seg[0] // 250:.0f}-{afib_seg[1] // 250:.0f}_afib.npy"
        normal_filename = f"{base}_{normal_seg[0] // 250:.0f}-{normal_seg[1] // 250:.0f}_normal.npy"
        np.save(os.path.join(save_folder_afib, afib_filename), afib_seg[2])
        np.save(os.path.join(save_folder_normal, normal_filename), normal_seg[2])

        print(
            f"记录 {base} 配对: AFIB片段 {afib_seg[0] // 250}-{afib_seg[1] // 250}秒，正常片段 {normal_seg[0] // 250}-{normal_seg[1] // 250}秒（噪声 {normal_seg[3]:.4f}）")
