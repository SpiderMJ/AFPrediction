import os
import glob
import numpy as np
import random
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm


class ECGAugmenter:
    """ECG信号增强类

    实现多种数据增强策略，特别针对心房颤动ECG信号
    """

    def __init__(self, fs=128):
        """
        参数:
            fs (int): 采样率，默认250Hz
        """
        self.fs = fs
        random.seed(42)  # 保证结果可重复
        np.random.seed(42)

    def add_gaussian_noise(self, ecg, snr_db=20):
        """添加高斯噪声

        参数:
            ecg (ndarray): 输入ECG信号
            snr_db (float): 信噪比，单位dB

        返回:
            ndarray: 增强后的信号
        """
        # 计算信号功率
        signal_power = np.mean(ecg ** 2)

        # 根据信噪比计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))

        # 生成噪声
        noise = np.random.normal(0, np.sqrt(noise_power), len(ecg))

        # 添加噪声
        return ecg + noise

    def add_laplacian_noise(self, ecg, snr_db=20):
        """添加拉普拉斯噪声

        参数:
            ecg (ndarray): 输入ECG信号
            snr_db (float): 信噪比，单位dB

        返回:
            ndarray: 增强后的信号
        """
        # 计算信号功率
        signal_power = np.mean(ecg ** 2)

        # 根据信噪比计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))

        # 计算拉普拉斯分布的scale参数
        scale = np.sqrt(noise_power / 2)

        # 生成拉普拉斯噪声
        noise = np.random.laplace(0, scale, len(ecg))

        # 添加噪声
        return ecg + noise

    def add_colored_noise(self, ecg, snr_db=20, beta=1.0):
        """添加有色噪声 (1/f^beta 噪声)

        参数:
            ecg (ndarray): 输入ECG信号
            snr_db (float): 信噪比，单位dB
            beta (float): 噪声颜色参数，1.0表示粉红噪声

        返回:
            ndarray: 增强后的信号
        """
        # 信号长度
        n = len(ecg)

        # 生成白噪声
        white_noise = np.random.normal(0, 1, n)

        # 转换为频域
        noise_fft = np.fft.rfft(white_noise)

        # 生成1/f谱
        f = np.fft.rfftfreq(n)
        f[0] = f[1]  # 避免除零
        fft_filter = f ** (-beta / 2)

        # 应用1/f滤波
        colored_fft = noise_fft * fft_filter

        # 转回时域
        colored_noise = np.fft.irfft(colored_fft, n)

        # 归一化
        colored_noise = colored_noise / np.std(colored_noise)

        # 计算信号功率
        signal_power = np.mean(ecg ** 2)

        # 根据信噪比计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))

        # 调整噪声幅度
        colored_noise = colored_noise * np.sqrt(noise_power)

        # 添加噪声
        return ecg + colored_noise

    def time_stretch(self, ecg, factor_range=(0.9, 1.1)):
        """时间拉伸或压缩

        参数:
            ecg (ndarray): 输入ECG信号
            factor_range (tuple): 拉伸因子范围，如(0.9, 1.1)表示时间轴变为原来的90%-110%

        返回:
            ndarray: 增强后的信号
        """
        # 随机选择拉伸因子
        factor = np.random.uniform(*factor_range)

        # 原始时间点
        x_original = np.arange(len(ecg))

        # 新的时间点
        x_new = np.linspace(0, len(ecg) - 1, int(len(ecg) * factor))

        # 插值
        interpolator = interp1d(x_original, ecg, kind='cubic', bounds_error=False, fill_value='extrapolate')

        # 获取新的信号
        ecg_stretched = interpolator(x_new)

        # 调整到原始长度
        if len(ecg_stretched) > len(ecg):
            # 截断
            ecg_stretched = ecg_stretched[:len(ecg)]
        else:
            # 填充
            padding = np.zeros(len(ecg) - len(ecg_stretched))
            ecg_stretched = np.concatenate([ecg_stretched, padding])

        return ecg_stretched

    def amplitude_scale(self, ecg, factor_range=(0.8, 1.2)):
        """振幅缩放

        参数:
            ecg (ndarray): 输入ECG信号
            factor_range (tuple): 缩放因子范围

        返回:
            ndarray: 增强后的信号
        """
        # 随机选择缩放因子
        factor = np.random.uniform(*factor_range)

        # 缩放信号
        return ecg * factor

    def add_baseline_wander(self, ecg, amplitude_ratio=0.05, freq_range=(0.05, 0.2)):
        """添加基线漂移

        参数:
            ecg (ndarray): 输入ECG信号
            amplitude_ratio (float): 基线漂移幅度与信号幅度的比例
            freq_range (tuple): 基线漂移频率范围，单位Hz

        返回:
            ndarray: 增强后的信号
        """
        # 信号长度
        n = len(ecg)

        # 信号幅度
        amplitude = np.max(ecg) - np.min(ecg)

        # 基线漂移幅度
        wandering_amplitude = amplitude * amplitude_ratio

        # 随机选择基线漂移频率
        freq = np.random.uniform(*freq_range)

        # 生成基线漂移
        t = np.arange(n) / self.fs
        wandering = wandering_amplitude * np.sin(2 * np.pi * freq * t)

        # 添加基线漂移
        return ecg + wandering

    def sliding_window(self, ecg, window_length=None, overlap_ratio=0.5):
        """滑动窗口，生成重叠片段

        参数:
            ecg (ndarray): 输入ECG信号
            window_length (int): 窗口长度，默认为None使用1/3信号长度
            overlap_ratio (float): 窗口重叠比例

        返回:
            list: 窗口片段列表
        """
        if window_length is None:
            window_length = len(ecg) // 3

        # 计算步长
        step = int(window_length * (1 - overlap_ratio))

        # 生成窗口
        windows = []
        for i in range(0, len(ecg) - window_length + 1, step):
            windows.append(ecg[i:i + window_length])

        return windows

    def segment_shuffle(self, ecg, n_segments=5):
        """内部片段交换

        参数:
            ecg (ndarray): 输入ECG信号
            n_segments (int): 分段数量

        返回:
            ndarray: 增强后的信号
        """
        # 将信号分为n段
        segments = np.array_split(ecg, n_segments)

        # 随机打乱片段顺序（但保持一定连贯性）
        # 这里我们只交换相邻片段以保持信号的整体形态
        for i in range(1, len(segments), 2):
            if i < len(segments) - 1:
                segments[i], segments[i + 1] = segments[i + 1], segments[i]

        # 重新组合
        return np.concatenate(segments)

    def signal_inversion(self, ecg):
        """信号反转

        参数:
            ecg (ndarray): 输入ECG信号

        返回:
            ndarray: 增强后的信号
        """
        # 反转信号，同时保持中心位置
        mean_val = np.mean(ecg)
        return 2 * mean_val - ecg

    def apply_random_augmentation(self, ecg, label, strength='medium'):
        """应用随机组合的增强

        参数:
            ecg (ndarray): 输入ECG信号
            label (int): 标签
            strength (str): 增强强度，可选'light', 'medium', 'strong'

        返回:
            list: [(增强信号1, 标签), (增强信号2, 标签), ...]
        """
        # 根据强度设置参数
        if strength == 'light':
            noise_snr_range = (25, 35)
            time_factor_range = (0.95, 1.05)
            amp_factor_range = (0.9, 1.1)
            baseline_amp_ratio = 0.03
            n_augmentations = 2
        elif strength == 'medium':
            noise_snr_range = (15, 30)
            time_factor_range = (0.9, 1.1)
            amp_factor_range = (0.8, 1.2)
            baseline_amp_ratio = 0.05
            n_augmentations = 3
        else:  # strong
            noise_snr_range = (10, 25)
            time_factor_range = (0.85, 1.15)
            amp_factor_range = (0.7, 1.3)
            baseline_amp_ratio = 0.08
            n_augmentations = 4

        results = []

        # 应用基本增强，每个原始信号生成n_augmentations个变体
        for i in range(n_augmentations):
            # 复制原始信号
            aug_ecg = ecg.copy()

            # 随机选择应用的增强方法
            # 确保每个变体使用不同的增强组合
            augmentation_types = []

            # 1. 随机添加一种噪声
            noise_type = np.random.choice([
                'gaussian', 'laplacian', 'colored', 'none'
            ], p=[0.3, 0.3, 0.3, 0.1])
            augmentation_types.append(noise_type)

            if noise_type == 'gaussian':
                snr = np.random.uniform(*noise_snr_range)
                aug_ecg = self.add_gaussian_noise(aug_ecg, snr_db=snr)
            elif noise_type == 'laplacian':
                snr = np.random.uniform(*noise_snr_range)
                aug_ecg = self.add_laplacian_noise(aug_ecg, snr_db=snr)
            elif noise_type == 'colored':
                snr = np.random.uniform(*noise_snr_range)
                beta = np.random.uniform(0.5, 1.5)  # 随机1/f噪声颜色
                aug_ecg = self.add_colored_noise(aug_ecg, snr_db=snr, beta=beta)

            # 2. 随机应用时间拉伸
            if np.random.rand() < 0.7:
                aug_ecg = self.time_stretch(aug_ecg, factor_range=time_factor_range)
                augmentation_types.append('time_stretch')

            # 3. 随机应用振幅缩放
            if np.random.rand() < 0.7:
                aug_ecg = self.amplitude_scale(aug_ecg, factor_range=amp_factor_range)
                augmentation_types.append('amplitude_scale')

            # 4. 随机添加基线漂移
            if np.random.rand() < 0.6:
                aug_ecg = self.add_baseline_wander(aug_ecg,
                                                   amplitude_ratio=baseline_amp_ratio,
                                                   freq_range=(0.05, 0.2))
                augmentation_types.append('baseline_wander')

            # 5. 较低概率应用片段交换，保持信号连贯性
            if np.random.rand() < 0.1 and not 'time_stretch' in augmentation_types:  # 降低到0.1
                aug_ecg = self.segment_shuffle(aug_ecg, n_segments=np.random.randint(3, 5))  # 降低分段数量
                augmentation_types.append('segment_shuffle')

            # 6. 很低概率应用信号反转
            if np.random.rand() < 0.1:
                aug_ecg = self.signal_inversion(aug_ecg)
                augmentation_types.append('signal_inversion')

            # 添加到结果中
            results.append((aug_ecg, label))

        return results

    def apply_sliding_window_augmentation(self, ecg, label, window_size=None, overlap=0.5):
        """应用滑动窗口增强

        参数:
            ecg (ndarray): 输入ECG信号
            label (int): 标签
            window_size (int): 窗口大小，默认为None使用信号长度的1/3
            overlap (float): 窗口重叠比例

        返回:
            list: [(窗口1, 标签), (窗口2, 标签), ...]
        """
        windows = self.sliding_window(ecg, window_length=window_size, overlap_ratio=overlap)
        return [(window, label) for window in windows if len(window) == (window_size or len(ecg) // 3)]

    def visualize_augmentations(self, original, augmented_list, fs=128, save_path=None):
        """可视化原始信号和增强信号

        参数:
            original (ndarray): 原始ECG信号
            augmented_list (list): 增强信号列表
            fs (int): 采样率
            save_path (str): 保存路径，默认为None不保存

        返回:
            None
        """
        n_aug = len(augmented_list)
        plt.figure(figsize=(12, 6 + 2 * n_aug))

        # 计算时间轴
        time = np.arange(len(original)) / fs

        # 绘制原始信号
        plt.subplot(n_aug + 1, 1, 1)
        plt.plot(time, original)
        plt.title('Original Signal')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # 绘制增强信号
        for i, (aug_sig, _) in enumerate(augmented_list):
            plt.subplot(n_aug + 1, 1, i + 2)
            aug_time = np.arange(len(aug_sig)) / fs
            plt.plot(aug_time, aug_sig)
            plt.title(f'Augmented Signal {i + 1}')
            plt.ylabel('Amplitude')
            plt.grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100)
            plt.close()
        else:
            plt.show()


def augment_ecg_dataset(source_dir, output_dir, fs=128, visualize=False, balanced=True):
    """对ECG数据集进行增强

    参数:
        source_dir (str): 源数据目录，包含"High_risk"和"Low_risk"子目录
        output_dir (str): 输出目录
        fs (int): 采样率
        visualize (bool): 是否可视化一些增强样例
        balanced (bool): 是否平衡两个类别的样本数

    返回:
        tuple: (原始样本数, 增强后样本数)
    """
    # 确保输出目录存在
    high_risk_out = os.path.join(output_dir, 'High_risk')
    low_risk_out = os.path.join(output_dir, 'Low_risk')

    os.makedirs(high_risk_out, exist_ok=True)
    os.makedirs(low_risk_out, exist_ok=True)

    if visualize:
        vis_dir = os.path.join(output_dir, 'visualization')
        os.makedirs(vis_dir, exist_ok=True)

    # 创建增强器
    augmenter = ECGAugmenter(fs=fs)

    # 记录统计信息
    original_count = {'High_risk': 0, 'Low_risk': 0}
    augmented_count = {'High_risk': 0, 'Low_risk': 0}

    # 读取和增强每个类别的数据
    for category in ['High_risk', 'Low_risk']:
        category_dir = os.path.join(source_dir, category)
        files = glob.glob(os.path.join(category_dir, '*.npy'))
        original_count[category] = len(files)

        print(f"处理{category}类数据，共{len(files)}个文件")

        # 处理每个文件
        for file_idx, file_path in enumerate(tqdm(files)):
            try:
                # 加载ECG数据
                ecg = np.load(file_path)

                # 确保是单导联数据
                if ecg.ndim > 1:
                    ecg = ecg[:, 0]  # 取第一个导联

                # 文件基本名（不含路径和扩展名）
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                # 标签编码
                label = 1 if category == 'High_risk' else 0

                # 应用基本的"复制+变异"增强，每个样本生成3-4个变体
                strength = 'medium'  # 默认中等强度

                # 对于两种情况使用不同增强策略
                augmented_samples = []

                # 1. 应用随机组合增强
                aug_samples = augmenter.apply_random_augmentation(ecg, label, strength=strength)
                augmented_samples.extend(aug_samples)

                # 2. 应用滑动窗口增强（每个文件生成2-3个重叠窗口）
                window_size = 30 * fs  # 30秒窗口
                window_samples = augmenter.apply_sliding_window_augmentation(
                    ecg, label, window_size=window_size, overlap=0.7
                )
                # 最多只保留3个窗口
                if len(window_samples) > 3:
                    window_samples = window_samples[:3]
                augmented_samples.extend(window_samples)

                # 保存增强后的样本
                out_dir = high_risk_out if category == 'High_risk' else low_risk_out

                # 可视化第一个文件的增强结果
                if file_idx == 0 and visualize:
                    vis_path = os.path.join(vis_dir, f"{category}_augmentation_example.png")
                    augmenter.visualize_augmentations(ecg, augmented_samples[:4], fs=fs, save_path=vis_path)

                # 保存增强后的样本
                for i, (aug_ecg, aug_label) in enumerate(augmented_samples):
                    output_file = os.path.join(out_dir, f"{base_name}_aug{i}.npy")
                    np.save(output_file, aug_ecg)
                    augmented_count[category] += 1

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")

    # 如果需要平衡类别
    if balanced:
        # 找出样本较多的类别
        max_count = max(augmented_count['High_risk'], augmented_count['Low_risk'])
        min_count = min(augmented_count['High_risk'], augmented_count['Low_risk'])
        min_category = 'High_risk' if augmented_count['High_risk'] < augmented_count['Low_risk'] else 'Low_risk'

        if max_count > min_count:
            # 需要为少数类增加样本
            print(f"平衡类别，为{min_category}类增加样本，从{min_count}个到{max_count}个")

            # 找出所有少数类的文件
            min_cat_dir = os.path.join(output_dir, min_category)
            min_cat_files = glob.glob(os.path.join(min_cat_dir, '*.npy'))

            # 计算需要增加的样本数
            samples_to_add = max_count - min_count

            # 随机选择一些样本进行复制和轻微变化
            selected_files = np.random.choice(min_cat_files, samples_to_add, replace=True)

            for i, file_path in enumerate(tqdm(selected_files)):
                try:
                    # 加载样本
                    ecg = np.load(file_path)

                    # 生成轻微变化
                    # 注意：这里我们只应用非常轻微的增强，避免造成太多的人工特征
                    snr = np.random.uniform(30, 40)  # 非常轻微的噪声
                    aug_ecg = augmenter.add_gaussian_noise(ecg, snr_db=snr)

                    # 轻微时间拉伸
                    if np.random.rand() < 0.5:
                        factor = np.random.uniform(0.98, 1.02)
                        x_original = np.arange(len(aug_ecg))
                        x_new = np.linspace(0, len(aug_ecg) - 1, int(len(aug_ecg) * factor))
                        interpolator = interp1d(x_original, aug_ecg, bounds_error=False, fill_value='extrapolate')
                        aug_ecg = interpolator(x_new)
                        # 调整长度
                        if len(aug_ecg) > len(ecg):
                            aug_ecg = aug_ecg[:len(ecg)]
                        else:
                            aug_ecg = np.pad(aug_ecg, (0, len(ecg) - len(aug_ecg)))

                    # 保存新生成的样本
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_file = os.path.join(min_cat_dir, f"{base_name}_bal{i}.npy")
                    np.save(output_file, aug_ecg)
                    augmented_count[min_category] += 1

                except Exception as e:
                    print(f"平衡样本时处理文件 {file_path} 出错: {str(e)}")

    # 打印统计信息
    print("\n数据增强完成！")
    print(f"原始样本数: High_risk={original_count['High_risk']}, Low_risk={original_count['Low_risk']}")
    print(f"增强后样本数: High_risk={augmented_count['High_risk']}, Low_risk={augmented_count['Low_risk']}")
    print(f"总样本数: 原始={sum(original_count.values())}, 增强后={sum(augmented_count.values())}")
    print(f"增强倍数: {sum(augmented_count.values()) / max(1, sum(original_count.values())):.2f}倍")

    return original_count, augmented_count


if __name__ == "__main__":
    # 设置路径
    source_directory = 'E:\AFprediction2025.03.05\Combination_Training\Database\DATA3_processed_filtered'
    output_directory = 'E:\AFprediction2025.03.05\Combination_Training\Database\DATA3_augmented'

    # 运行数据增强
    orig_count, aug_count = augment_ecg_dataset(
        source_dir=source_directory,
        output_dir=output_directory,
        fs=128,
        visualize=True,  # 生成可视化效果
        balanced=True  # 平衡两个类别的样本数
    )