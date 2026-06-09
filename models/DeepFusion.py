
import torch
import torch.nn as nn
import torch.nn.functional as F



class AdaptiveLowPassFilter(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.generate_kernel = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, in_channels * kernel_size ** 2, 1)
        )
        self.norm = nn.InstanceNorm2d(in_channels)  # 添加归一化

    def forward(self, high_feat, low_feat):
        B, C, H, W = high_feat.shape
        x = torch.cat([high_feat, low_feat], 1)
        kernel = self.generate_kernel(x)  # [B, C*k^2, H, W]
        kernel = F.softmax(kernel.view(B, C, self.kernel_size ** 2, H, W), dim = 2)  # 通道级归一化

        high_unf = F.unfold(high_feat, self.kernel_size, padding=self.padding)  # [B, C*k^2, L]
        high_unf = high_unf.view(B, C, self.kernel_size ** 2, H, W)

        out = (kernel * high_unf).sum(dim=2)  # [B, C, H, W]
        return self.norm(out)  # 保留特征分布


class OffsetGenerator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pyramid = nn.Sequential(
            nn.Conv2d(in_channels * 2, 128, 3, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear')  # 上采样恢复分辨率
        )
        self.offset_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, high, low):
        x = torch.cat([high, low], 1)
        pyramid_feat = self.pyramid(x)  # 多尺度特征
        offset = self.offset_conv(pyramid_feat) * 0.5  # 限制偏移在[-0.5,0.5]
        return offset

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.avg_pool(x).view(b, c)
        weight = self.fc(avg).view(b, c, 1, 1)
        return x * weight.expand_as(x)

class DeepFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpf = AdaptiveLowPassFilter(in_channels)
        self.offset_gen = OffsetGenerator(in_channels)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.Sigmoid()  # 输出[0,1]权重
        )

        # 高频增强
        self.high_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            ChannelAttention(in_channels)
        )

        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, high_feat, low_feat):
        # 高频增强
        enhanced_high = self.high_enhance(high_feat)

        # 低频处理
        low_filtered = self.alpf(enhanced_high, low_feat)

        offset = self.offset_gen(enhanced_high, low_filtered)

        # 形变对齐
        B, C, H, W = enhanced_high.shape
        grid = self._create_grid(B, H, W, enhanced_high.device)
        #grid = grid + offset.permute(0, 2, 3, 1) / torch.tensor([W / 2, H / 2], device=enhanced_high.device)
        scale = torch.tensor([W / 2, H / 2], device=enhanced_high.device).view(1, 1, 1, 2)
        grid = grid + offset.permute(0, 2, 3, 1) / scale
        low_warped = F.grid_sample(low_filtered, grid, mode='bilinear')

        # 动态门控融合
        gate = self.gate(torch.cat([enhanced_high, low_warped], 1))
        fused = gate * low_warped + (1 - gate) * enhanced_high  # 高频主导

        return self.final_conv(fused) + high_feat  # 残差连接原始高频

    def _create_grid(self, B, H, W, device):
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device),
                              indexing='ij')
        return torch.stack((x, y), -1).unsqueeze(0).repeat(B, 1, 1, 1)


