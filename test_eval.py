import cv2
import numpy as np
import librosa
import torch
import lpips
#import mediapipe as mp
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP库
INCEPTION_SIZE = (299, 299)  # Inception-v3输入尺寸
# =============================================================================
# 配置参数
# =============================================================================
VIDEO1_PATH = 'D:/Python/team/DINet2/eval/examples/videocrop2.mp4'
VIDEO2_PATH = 'D:/Python/team/DINet2/eval/result/wav2lip/2.mp4'
SAMPLE_INTERVAL = 5  # 每5帧采样一次以加速计算
AUDIO_SAMPLE_RATE = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# 工具函数
# =============================================================================
def extract_frames(video_path, interval=SAMPLE_INTERVAL):
    """提取视频帧并采样"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转为RGB
        frame_count += 1
    cap.release()
    return frames


def extract_audio(video_path):
    """提取音频波形"""
    audio, sr = librosa.load(video_path, sr=AUDIO_SAMPLE_RATE)
    return audio, sr


def resize_frames(frames, target_size):
    """调整帧尺寸以对齐视频"""
    return [cv2.resize(frame, target_size) for frame in frames]


# =============================================================================
# 视觉质量评估
# =============================================================================
def calculate_visual_metrics(frames1, frames2):
    """计算SSIM、PSNR、LPIPS"""
    #ssim_scores = []
    psnr_scores = []
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
    lpips_scores = []

    for frame1, frame2 in tqdm(zip(frames1, frames2), desc='Processing frames'):
        # # 检查图像尺寸并动态设置 win_size
        # min_height = min(frame1.shape[0], frame2.shape[0])
        # min_width = min(frame1.shape[1], frame2.shape[1])
        # win_size = 7
        # if min_height < 7 or min_width < 7:
        #     win_size = 3  # 如果图像尺寸小于7x7，使用更小的窗口
        #
        # # 确保输入是多通道图像（如RGB）
        # if frame1.ndim == 2 or frame2.ndim == 2:
        #     raise ValueError("图像必须是多通道（如RGB）格式")
        #
        # # 计算SSIM（兼容新旧版本scikit-image）
        # try:
        #     ssim_val = ssim(
        #         frame1, frame2,
        #         channel_axis=-1,  # 新版本参数
        #         data_range=255,
        #         win_size=win_size
        #     )
        # except TypeError:
        #     # 回退到旧版本参数 multichannel=True
        #     ssim_val = ssim(
        #         frame1, frame2,
        #         multichannel=True,  # 旧版本参数
        #         data_range=255,
        #         win_size=win_size
        #     )
        # ssim_scores.append(ssim_val)

        # PSNR
        mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32))  **  2)
        psnr_val = 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')
        psnr_scores.append(psnr_val)

        # LPIPS（需转为Tensor）
        frame1_tensor = torch.from_numpy(frame1).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        frame2_tensor = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0 * 2 - 1
        with torch.no_grad():
            lpips_val = loss_fn(frame1_tensor, frame2_tensor).item()
        lpips_scores.append(lpips_val)

    return {
        # 'SSIM': np.mean(ssim_scores),
        'PSNR': np.mean(psnr_scores),
        'LPIPS': np.mean(lpips_scores)
    }


class FIDCalculator:
    def __init__(self):
        self.model = inception_v3(pretrained=True, transform_input=False).eval().to(DEVICE)
        self.model.fc = torch.nn.Identity()  # 获取2048维特征

    def preprocess_image(self, image):
        """图像预处理：调整尺寸、归一化"""
        # 调整尺寸至299x299
        resized = cv2.resize(image, INCEPTION_SIZE)
        # 转换通道顺序 [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        # 归一化到[-1, 1]（与Inception训练一致）
        tensor = (tensor / 255.0) * 2 - 1
        return tensor.unsqueeze(0).to(DEVICE)  # 添加batch维度

    def extract_features(self, frames):
        """批量提取特征"""
        features = []
        with torch.no_grad():
            for frame in tqdm(frames, desc='Extracting FID features'):
                tensor = self.preprocess_image(frame)
                feature = self.model(tensor)
                features.append(feature.cpu().numpy())
        return np.concatenate(features, axis=0)

    @staticmethod
    def calculate_fid(real_features, fake_features):
        """计算FID分数"""
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

        # 均值差异
        diff = mu_real - mu_fake
        mean_term = np.dot(diff, diff)

        # 协方差项
        cov_sqrt, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        cov_term = np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)

        return mean_term + cov_term


# =============================================================================
# 视听同步评估（伪代码，需替换为实际SyncNet实现）
# =============================================================================
# class SyncNetWrapper:
#     """伪代码：实际实现需参考 https://github.com/joonson/syncnet_python"""
#
#     def __init__(self):
#         pass
#
#     def compare(self, video_frame, audio_segment):
#         # 此处应返回LSE-D和LSE-C
#         return np.random.rand(), np.random.rand()  # 示例随机值
#
#
# def calculate_av_sync(frames, audio, sample_rate):
#     """计算LSE-D和LSE-C"""
#     syncnet = SyncNetWrapper()
#     window_size = int(sample_rate * 0.2)  # 每200ms音频窗口
#     lse_d, lse_c = [], []
#
#     for i, frame in tqdm(enumerate(frames), desc='Processing AV sync'):
#         start = i * window_size
#         end = start + window_size
#         if end > len(audio):
#             break
#         audio_segment = audio[start:end]
#         d, c = syncnet.compare(frame, audio_segment)
#         lse_d.append(d)
#         lse_c.append(c)
#
#     return {
#         'LSE-D': np.mean(lse_d),
#         'LSE-C': np.mean(lse_c)
#     }

# =============================================================================
# 主流程
# =============================================================================
def main():
    # 初始化FID计算器
    fid_calculator = FIDCalculator()
    # 步骤1: 提取视频帧并调整尺寸
    frames1 = extract_frames(VIDEO1_PATH)
    frames2 = extract_frames(VIDEO2_PATH)
    target_size = (frames1[0].shape[1], frames1[0].shape[0])
    frames2 = resize_frames(frames2, target_size)

    # 步骤2: 计算视觉质量指标
    visual_metrics = calculate_visual_metrics(frames1, frames2)
    print("\n视觉质量指标:")
    for k, v in visual_metrics.items():
        print(f"{k}: {v:.4f}")

    # 步骤3: 计算FID
    print("\n计算FID...")
    real_features = fid_calculator.extract_features(frames1)
    fake_features = fid_calculator.extract_features(frames2)
    fid_score = fid_calculator.calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score:.2f}")

    # 步骤3: 提取音频并计算视听同步
    # audio1, sr1 = extract_audio(VIDEO1_PATH)
    # av_sync_metrics1 = calculate_av_sync(frames1, audio1, sr1)
    # audio2, sr2 = extract_audio(VIDEO2_PATH)
    # av_sync_metrics2 = calculate_av_sync(frames2, audio2, sr2)
    #
    # print("\n视听同步指标 (视频1):")
    # for k, v in av_sync_metrics1.items():
    #     print(f"{k}: {v:.4f}")
    # print("\n视听同步指标 (视频2):")
    # for k, v in av_sync_metrics2.items():
    #     print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    main()