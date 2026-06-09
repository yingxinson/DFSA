# #import cv2
# #print(cv2.__version__)
# #print(cv2.getBuildInformation())
#
#
# # 测试视频写入功能
# #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器标签
# #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
# #frame = [ [255]*640 for _ in range(480) ]  # 创建测试帧
# #out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
# #out.release()
# #import tensorflow as tf
# #rint("TF Version:", tf.__version__)
# #print("GPU Devices:", tf.config.list_physical_devices('GPU'))
# import numpy as np
# import os
#
# # deep_speech_dir = "./asserts/training_data/split_video_25fps_deepspeech"
# # for file in os.listdir(deep_speech_dir):
# #     data = np.load(os.path.join(deep_speech_dir, file),allow_pickle=True)
# #     print(f"{file}: shape={data.shape}")
#
# import torch
#
#
# def print_model_keys(checkpoint_path):
#     try:
#         # 加载检查点文件
#         checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#
#         # 检查是否是state_dict格式
#         if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint
#
#         # 打印所有键
#         print("Model keys:")
#         for key in state_dict.keys():
#             print(key)
#
#     except FileNotFoundError:
#         print(f"Error: 文件未找到 - {checkpoint_path}")
#     except Exception as e:
#         print(f"加载检查点时发生错误: {str(e)}")
#
#
# # 使用示例
# if __name__ == "__main__":
#     checkpoint_path = "H:/BaiduNetdiskDownload/depth4netG_model_epoch_22.pth"  # 替换实际路径
#     print_model_keys(checkpoint_path)
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coordinate_grid_2d(size, dtype):
    """创建二维坐标网格"""
    h, w = size
    x = torch.linspace(-1, 1, w, dtype=dtype)
    y = torch.linspace(-1, 1, h, dtype=dtype)
    grid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    return grid  # [H, W, 2]


def bicubic_grid_sample(input, grid, align_corners=True):
    """自定义双三次插值实现"""
    # 使用PyTorch内置的grid_sample配合调整参数
    # 注意：实际双三次插值需要自定义实现，这里为简化使用双线性插值替代
    return F.grid_sample(input, grid, mode='bicubic', padding_mode='border', align_corners=align_corners)


class AdaAT(nn.Module):
    def __init__(self, para_ch=384, feature_ch=256):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch

        # 增强参数生成网络
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(para_ch * 2, para_ch),
            nn.LayerNorm(para_ch)
        )

        # 动态参数生成组件
        self.scale = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Tanh()
        )
        self.rotation = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Tanh()
        )
        self.translation = nn.Sequential(
            nn.Linear(para_ch, 2 * feature_ch),
            nn.Tanh()
        )

        # 延迟初始化空间注意力（需要知道特征图尺寸）
        self.spatial_attention = None

    def _init_spatial_attention(self, h, w):
        """延迟初始化空间注意力层"""
        self.spatial_attention = nn.Sequential(
            nn.Linear(self.para_ch, h * w),
            nn.Sigmoid()
        ).to(next(self.parameters()).device)

    def forward(self, feature_map, para_code):
        batch, c, h, w = feature_map.size()

        # 延迟初始化（首次运行时初始化）
        if self.spatial_attention is None:
            self._init_spatial_attention(h, w)

        # 增强参数生成
        para_code = self.commn_linear(para_code)

        # 生成空间注意力图
        attention = self.spatial_attention(para_code).view(batch, 1, h, w)

        # 参数范围调整
        scale = (self.scale(para_code) + 1.5).unsqueeze(-1)  # [B, C, 1]
        angle = self.rotation(para_code) * (torch.pi / 2)  # [B, C]
        translation = self.translation(para_code).view(batch, self.feature_ch, 2) * 1.5  # [B, C, 2]

        # 构造旋转矩阵
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_matrix = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1)  # [B, C, 4]
        rot_matrix = rot_matrix.view(batch, c, 2, 2)  # [B, C, 2, 2]

        # 创建坐标网格
        grid = make_coordinate_grid_2d((h, w), feature_map.dtype)
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)  # [B, H, W, 2]

        # 应用仿射变换
        transformed_grid = torch.einsum('bchwj,bcjkl->bchwl',
                                        grid.view(batch, 1, h, w, 2),
                                        rot_matrix.view(batch, c, 1, 1, 2, 2))
        transformed_grid = transformed_grid * scale.view(batch, c, 1, 1, 1)
        transformed_grid += translation.view(batch, c, 1, 1, 2)

        # 应用空间注意力混合
        transformed_grid = grid.unsqueeze(1) * (1 - attention.unsqueeze(-1)) + \
                           transformed_grid * attention.unsqueeze(-1)

        # 调整维度进行采样
        transformed_grid = transformed_grid.reshape(batch * c, h, w, 2)
        feature_map_sampled = feature_map.unsqueeze(2)  # [B, C, 1, H, W]

        # 双三次插值采样
        trans_feature = F.grid_sample(
            feature_map_sampled,
            transformed_grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=True
        ).squeeze(2)

        # 残差连接
        return trans_feature * 0.7 + feature_map * 0.3


# 测试用例
if __name__ == "__main__":
    # 模拟输入
    batch_size = 4
    feature = torch.randn(batch_size, 256, 64, 64)  # [B, C, H, W]
    code = torch.randn(batch_size, 384)  # [B, para_ch]

    # 初始化模块
    model = AdaAT()

    # 前向传播
    output = model(feature, code)
    print(f"输入形状: {feature.shape}")
    print(f"输出形状: {output.shape}")
