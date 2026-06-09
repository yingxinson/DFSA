# import torch
# from torch import nn
# import torch.nn.functional as F
# import math
# import cv2
# import numpy as np
# from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
# from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
#
#
# def make_coordinate_grid_3d(spatial_size, type):
#     '''
#         generate 3D coordinate grid
#     '''
#     d, h, w = spatial_size
#     x = torch.arange(w).type(type)
#     y = torch.arange(h).type(type)
#     z = torch.arange(d).type(type)
#     x = (2 * (x / (w - 1)) - 1)
#     y = (2 * (y / (h - 1)) - 1)
#     z = (2 * (z / (d - 1)) - 1)
#     yy = y.view(1, -1, 1).repeat(d, 1, w)
#     xx = x.view(1, 1, -1).repeat(d, h, 1)
#     zz = z.view(-1, 1, 1).repeat(1, h, w)
#     meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
#     return meshed, zz
#
#
# class ResBlock1d(nn.Module):
#     '''
#         basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock1d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv1d(in_features, out_features, 1)
#         self.norm1 = BatchNorm1d(in_features)
#         self.norm2 = BatchNorm1d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class ResBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock2d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv2d(in_features, out_features, 1)
#         self.norm1 = BatchNorm2d(in_features)
#         self.norm2 = BatchNorm2d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class UpBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(UpBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = F.interpolate(x, scale_factor=2)
#         out = self.conv(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out
#
#
# class DownBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(DownBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=2)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class DownBlock2d(nn.Module):
#     # basic block
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
#         super(DownBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=stride)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # print(f"\n[DownBlock2d] Input shape: {x.shape}")  # 新增
#         out = self.conv(x)
#         # print(f"[DownBlock2d] Output shape: {out.shape}")  # 新增
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(SameBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(SameBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
# def make_coordinate_grid_2d(spatial_size, type):
#     """生成2D坐标网格"""
#     h, w = spatial_size
#     x = torch.linspace(-1, 1, w).type(type)
#     y = torch.linspace(-1, 1, h).type(type)
#     yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)
#     grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
#     return grid
# class AdaAT(nn.Module):
#     """优化后的2D自适应仿射变换"""
#
#     def __init__(self, para_ch, feature_ch):
#         super(AdaAT, self).__init__()
#         self.para_ch = para_ch
#         self.feature_ch = feature_ch
#
#         # 共享参数生成网络
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU()
#         )
#
#         # 各变换参数生成器
#         self.scale = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Sigmoid()
#         )
#         self.rotation = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Tanh()
#         )
#         self.translation = nn.Sequential(
#             nn.Linear(para_ch, 2 * feature_ch),
#             nn.Tanh()
#         )
#
#     def forward(self, feature_map, para_code):
#         batch, c, h, w = feature_map.shape  # 输入维度: [B, C, H, W]
#
#         # 1. 参数生成
#         para_code = self.commn_linear(para_code)
#         scale = self.scale(para_code) * 2  # [B, C]
#         angle = self.rotation(para_code) * math.pi  # [B, C]
#         translation = self.translation(para_code).view(batch, c, 2)  # [B, C, 2]
#
#         # 2. 生成基础坐标网格
#         grid = make_coordinate_grid_2d((h, w), feature_map.dtype)  # [H, W, 2]
#         grid = grid.unsqueeze(0).to(feature_map.device)  # [1, H, W, 2]
#
#         # 3. 构造旋转矩阵
#         cos_a = torch.cos(angle).view(batch, c, 1, 1, 1)  # [B, C, 1, 1, 1]
#         sin_a = torch.sin(angle).view(batch, c, 1, 1, 1)
#         rotation_matrix = torch.stack([
#             torch.cat([cos_a, -sin_a], dim=-1),
#             torch.cat([sin_a, cos_a], dim=-1)
#         ], dim=-2)  # [B, C, 1, 1, 2, 2]
#
#         # 4. 应用变换（广播机制实现）
#         # 扩展网格到batch和通道维度
#         grid = grid.expand(batch, c, h, w, 2)  # [B, C, H, W, 2]
#
#         # 旋转和缩放
#         transformed = torch.matmul(
#             rotation_matrix,
#             grid.unsqueeze(-1)
#         ).squeeze(-1) * scale.view(batch, c, 1, 1, 1)
#
#         # 平移
#         transformed += translation.view(batch, c, 1, 1, 2)
#
#         # 5. 通道并行采样
#         # 重组维度进行批量采样
#         transformed = transformed.view(batch * c, h, w, 2)
#         input_feature = feature_map.view(batch * c, 1, h, w)
#
#         # 执行网格采样
#         output = F.grid_sample(
#             input_feature,
#             transformed,
#             align_corners=False,
#             padding_mode='border'
#         )
#
#         return output.view(batch, c, h, w)
#
#
#
# class AdaAT1(nn.Module):
#     """ Adaptive Affine Transformation (保持3D网格结构) """
#
#     def __init__(self, para_ch=256, feature_ch=256):
#         super().__init__()
#         self.feature_ch = feature_ch
#
#         # 参数生成网络
#         self.param_net = nn.Sequential(
#             nn.Linear(para_ch, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU()
#         )
#
#         # 动态参数生成器
#         self.scale_gen = nn.Sequential(
#             nn.Linear(512, feature_ch),
#             nn.Sigmoid()
#         )
#         self.rotate_gen = nn.Sequential(
#             nn.Linear(512, feature_ch),
#             nn.Tanh()
#         )
#         self.translate_gen = nn.Sequential(
#             nn.Linear(512, 2 * feature_ch),
#             nn.Tanh()
#         )
#
#     def forward(self, feature_map, para_code):
#         B, C, H, W = feature_map.shape
#
#         # 参数生成
#         para_code = self.param_net(para_code)
#
#         # 缩放参数
#         scale = self.scale_gen(para_code).view(B, C, 1, 1) * 2
#
#         # 旋转参数
#         angle = self.rotate_gen(para_code).view(B, C) * math.pi
#         cos_a = torch.cos(angle).view(B, C, 1, 1)
#         sin_a = torch.sin(angle).view(B, C, 1, 1)
#
#         # 平移参数
#         translate = self.translate_gen(para_code).view(B, C, 2)
#
#         # 生成坐标网格
#         grid_xy, grid_z = make_coordinate_grid_3d(
#             spatial_size=(1, H, W),  # depth=1
#             type=feature_map.dtype
#         )
#
#         # 扩展 grid_xy 到 [B, C, H, W, 2]
#         grid_xy = grid_xy.unsqueeze(0).to(feature_map.device)  # [1,1,H,W,2]
#         grid_xy = grid_xy.expand(B, C, -1, -1, -1)  # [B,C,H,W,2]
#
#         # 构建旋转矩阵
#         rotation_matrix = torch.stack([
#             cos_a, -sin_a,
#             sin_a, cos_a
#         ], dim=-1).view(B, C, 1, 1, 2, 2)  # [B,C,1,1,2,2]
#
#         # 应用旋转缩放
#         transformed = torch.matmul(
#             rotation_matrix,
#             grid_xy.unsqueeze(-1)  # [B,C,H,W,2,1]
#         ).squeeze(-1)  # [B,C,H,W,2]
#
#         # 应用缩放和平移
#         transformed = transformed * scale.unsqueeze(-1) + translate.view(B, C, 1, 1, 2)
#
#         # 扩展 grid_z 到 [B,C,H,W,1]
#         grid_z = grid_z.unsqueeze(0).to(feature_map.device)  # [1,1,H,W]
#         grid_z = grid_z.unsqueeze(0).expand(B, C, -1, -1)  # [B,C,H,W]
#         grid_z = grid_z.unsqueeze(-1)  # [B,C,H,W,1]
#
#         # 拼接完整网格
#         full_grid = torch.cat([transformed, grid_z], dim=-1)  # [B,C,H,W,3]
#
#         # 采样特征
#         trans_feature = F.grid_sample(
#             input=feature_map.unsqueeze(1),  # [B,1,C,H,W]
#             grid=full_grid,
#             mode='bilinear',
#             padding_mode='border',
#             align_corners=True
#         ).squeeze(1)  # [B,C,H,W]
#
#         return trans_feature
# '''
# class AdaAT(nn.Module):
#
#      #  AdaAT operator
#
#
#     def __init__(self, para_ch=384, feature_ch=256):  #(self, para_ch, feature_ch)
#         super(AdaAT, self).__init__()
#         self.para_ch = para_ch
#         self.feature_ch = feature_ch
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU(inplace=True)
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Sigmoid()
#         )
#         self.rotation = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Tanh()
#         )
#         self.translation = nn.Sequential(
#             nn.Linear(para_ch, 2 * feature_ch),
#             nn.Tanh()
#         )
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature_map, para_code):
#         batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
#
#         para_code = self.commn_linear(para_code)
#
#         scale = self.scale(para_code).unsqueeze(-1) * 2
#         angle = self.rotation(para_code).unsqueeze(-1) * torch.pi  #
#         rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
#         rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
#         translation = self.translation(para_code).view(batch, self.feature_ch, 2)
#
#         grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
#         grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
#         grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
#         scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
#         translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
#         full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
#         trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid, mode='bilinear',padding_mode='border',align_corners=True).squeeze(1)
#         return trans_feature
# '''
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 全连接层（含降维与恢复）
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#
#         # 平均池化分支
#         avg_out = self.avg_pool(x).view(B, C)
#         avg_out = self.fc(avg_out).view(B, C, 1, 1)
#
#         # 最大池化分支
#         max_out = self.max_pool(x).view(B, C)
#         max_out = self.fc(max_out).view(B, C, 1, 1)
#
#         # 合并注意力权重
#         return x * (avg_out + max_out)
# class CrossModalFusion(nn.Module):
#     def __init__(self, img_ch=3, depth_ch=1, out_ch=4):
#         super().__init__()
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(img_ch + depth_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ChannelAttention(out_ch)  # 使用定义好的通道注意力
#         )
#
#     def forward(self, img, depth):
#         x = torch.cat([img, depth], dim=1)
#         return self.fuse_conv(x)
#
#
# class AudioVisualAttention(nn.Module):
#     def __init__(self, img_dim=128, audio_dim=128):
#         super().__init__()
#         self.query = nn.Linear(img_dim, 64)
#         self.key = nn.Linear(audio_dim, 64)
#         self.value = nn.Linear(audio_dim, img_dim)
#
#     def forward(self, img_feat, audio_feat):
#         # img_feat: [B,128], audio_feat: [B,128]
#         q = self.query(img_feat)  # [B,64]
#         k = self.key(audio_feat)  # [B,64]
#         v = self.value(audio_feat)  # [B,128]
#         attn = F.softmax(q @ k.T, dim=1)  # [B,B]
#         return img_feat + (attn @ v)  # [B,128]
# class DFSA(nn.Module):
#     def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
#         super(DFSA, self).__init__()
#         self.source_in_conv = nn.Sequential(
#             SameBlock2d(4, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1)
#         )
#
#         # 1x1 卷积来调整 ref_img 通道数
#         #self.ref_channel_adjust = nn.Conv2d(ref_channel, 3, kernel_size=1, stride=1, padding=0)
#
#         self.ref_in_conv = nn.Sequential(
#             SameBlock2d(20, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1),
#         )
#
#         # self.trans_conv = nn.Sequential(
#         #     # 20 →10
#         #     SameBlock2d(512, 128, kernel_size=3, padding=1),
#         #     SameBlock2d(128, 128, kernel_size=11, padding=5),
#         #     SameBlock2d(128, 128, kernel_size=11, padding=5),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 10 →5
#         #     SameBlock2d(128, 128, kernel_size=7, padding=3),
#         #     SameBlock2d(128, 128, kernel_size=7, padding=3),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 5 →3
#         #     SameBlock2d(128, 128, kernel_size=3, padding=1),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 3 →2
#         #     SameBlock2d(128, 128, kernel_size=3, padding=1),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         # )
#
#         # 修改trans_conv结构减少下采样
#         self.trans_conv = nn.Sequential(
#             SameBlock2d(512, 256, kernel_size=3, padding=1),
#             ResBlock2d(256, 256, kernel_size=3, padding=1),
#             DownBlock2d(256, 256, kernel_size=3, padding=1),  # 仅2次下采样
#             ResBlock2d(256, 128, kernel_size=3, padding=1),
#             ResBlock2d(128, 128, kernel_size=3, padding=1),
#         )
#
#         self.audio_encoder = nn.Sequential(
#             SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             SameBlock1d(128, 128, kernel_size=3, padding=1)
#         )
#
#         appearance_conv_list = []
#         for i in range(2):
#             appearance_conv_list.append(
#                 nn.Sequential(
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                 )
#             )
#         self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
#
#         self.adaAT = AdaAT(para_ch=256, feature_ch=256)
#
#         self.out_conv = nn.Sequential(
#             SameBlock2d(512, 128, kernel_size=3, padding=1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             ResBlock2d(128, 128, 3, 1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
#             nn.Sigmoid()
#         )
#         self.global_avg2d = nn.AdaptiveAvgPool2d(1)
#         self.global_avg1d = nn.AdaptiveAvgPool1d(1)
#
#         self.depth_encoder = depth_encoder
#         self.depth_decoder = depth_decoder
#
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.ReLU())
#         self.av_attn = AudioVisualAttention()
#         self.source_fusion = CrossModalFusion(img_ch=3, depth_ch=1, out_ch=4)
#         self.ref_fusion = CrossModalFusion(img_ch=15, depth_ch=5, out_ch=20)
#         # self.out_conv = nn.Sequential(
#         #     SameBlock2d(512, 256, kernel_size=3),
#         #     nn.PixelShuffle(2),  # 亚像素卷积提升分辨率
#         #     ResBlock2d(64, 64, kernel_size=3),
#         #     nn.Conv2d(64, 3, kernel_size=7, padding=3))
#
#     # def normalize_depth_input(self, img):
#     #     # 假设输入是 [0,1]，转为标准ImageNet归一化
#     #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
#     #     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
#     #     return (img - mean) / std
#
#
#     def forward(self, source_img, ref_img, audio_feature):
#         # # 输入 ref_img 之前，调整通道数
#         # #device = torch.device("cuda:0")
#         # #source_img = source_img.to(device)
#         # #ref_img = ref_img.to(device)
#         # #audio_feature = audio_feature.to(device)
#         # ref_img = self.ref_channel_adjust(ref_img)
#         # print(f"ref_img_3: {ref_img.shape}")
#         # ref_img = self.normalize_depth_input(ref_img)  # 先归一化
#         # print(f"ref_img_3_normal: {ref_img.shape}")
#         # #depth_encoder = depth_encoder.to(device)
#         # #depth_decoder = depth_decoder.to(device)
#         # # 统一调整输入尺寸到32的倍数（示例调整为128x96）
#         # def adjust_size(x, target_h=128, target_w=96):
#         #     return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
#         #     # 调整输入尺寸
#         # source_img_adj = adjust_size(source_img)
#         # ref_img_adj = adjust_size(ref_img)
#         #     # 用调整后的尺寸进行深度估计
#         # print(next(depth_encoder.parameters()).device)  # 应为cuda:0
#         # print(source_img.device)  # 应与模型一致
#         # with torch.no_grad():  # 关闭梯度以节省显存
#         #     print("Before source depth_encoder")
#         #     depth_source = depth_decoder(depth_encoder(source_img_adj))[("disp", 0)]
#         #     print("After source depth_encoder")
#         #     print("Before ref depth_encoder")
#         #     depth_ref = depth_decoder(depth_encoder(ref_img_adj))[("disp", 0)]
#         #     print("After ref depth_encoder")
#         #
#         #     # 插值回原尺寸（如果需要与原图拼接）
#         # depth_source = adjust_size(depth_source, source_img.shape[2], source_img.shape[3])
#         # depth_ref = adjust_size(depth_ref, ref_img.shape[2], ref_img.shape[3])
#         #
#         # # 后续拼接和网络前向保持不变
#         # source_img = torch.cat((source_img, depth_source), 1)
#         # ref_img = torch.cat((ref_img, depth_ref), 1)
#         # print(f"[Input] source_img_depth_add: {source_img.shape}, ref_img_depth_add: {ref_img.shape}")
#
#         ## source image encoder
#         source_img.float()
#         ref_img.float()
#         # print(f"[Input] source_img: {source_img.shape}, ref_img: {ref_img.shape}")
#
#         # 输入 ref_img 之前，调整通道数
#         #ref_img = self.ref_channel_adjust(ref_img)  # 1
#         # print(f"ref_img_3: {ref_img.shape}")
#         #ref_img = self.normalize_depth_input(ref_img)  # 先归一化  #1
#         # print(f"ref_img_3: {ref_img.shape}")
#
#
#         # 调整ref
#         # 假设 ref_img 形状为 [Batch, 15, H, W]
#         batch_size, _, H, W = ref_img.shape
#         # 拆分为5个独立3通道图像 [Batch, 5, 3, H, W]
#         ref_imgs = ref_img.view(batch_size, 5, 3, H, W)
#         # 方法一：通道拼接+降维 (适合decoder接受高维输入)
#         depth_features = []
#         for i in range(5):
#             feat_encoder = self.depth_encoder(ref_imgs[:, i])  # [B, C, H', W']
#             feat_decoder = self.depth_decoder(feat_encoder)
#             feat_features = feat_decoder[("disp", 0)]
#             #print("feat_decoder", feat_features.shape)
#             depth_features.append(feat_features)
#             #depth_features.append(feat_decoder)
#         # 拼接特征并降维
#         combined_feat = torch.cat(depth_features, dim=1)  # [B, 5C, H', W']
#         #print("combined_feat", combined_feat.shape)
#         #fusion_conv = nn.Conv2d(5 * 1, 1, kernel_size=1)  # 添加适配层
#         #final_feat = fusion_conv(combined_feat)
#         #depth_ref = combined_feat[("disp", 0)]
#         depth_ref = combined_feat
#         #print("depth_ref0", depth_ref.shape)
#         depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",align_corners=True)
#         #depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bicubic",align_corners=True)
#         #print("depth_ref", depth_ref.shape)
#         #ref_img = torch.cat((ref_img, depth_ref), 1)
#
#         ref_img = self.ref_fusion(ref_img, depth_ref)
#         #print("ref_img_add", ref_img.shape)
#
#
#         # print("depth",next(self.depth_encoder.parameters()).device)  # 应为cuda:0
#         # print("source",source_img.device)  # 应与模型一致
#
#         # print("Before source depth_encoder")
#         outputs = self.depth_decoder(self.depth_encoder(source_img))
#         depth_source = outputs[("disp", 0)]
#         # print("After source depth_encoder")
#         # print("depth",next(self.depth_encoder.parameters()).device)  # 应为cuda:0
#         # print("source",source_img.device)  # 应与模型一致
#         # print("Before ref depth_encoder")
#         # outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
#         # outputs = self.depth_decoder(self.depth_encoder(ref_img))  # 1
#         # depth_ref = outputs[("disp", 0)]   # 1
#         # print("After ref  depth_encoder")
#         # print(f"[Input] source_img_depth: {depth_source.shape}, ref_img_depth: {depth_ref.shape}")
#         #depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode="bilinear",align_corners=True)
#         # 改用自适应插值或引导滤波
#         depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode='bilinear', align_corners=True)
#         #depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",align_corners=True) #  1
#         # print(f"[Input] source_img_depth_interpolate: {depth_source.shape}, ref_img_depth: {depth_ref.shape}")
#
#         #source_img = torch.cat((source_img, depth_source), 1)
#         source_img = self.source_fusion(source_img, depth_source)
#         # ref_img = torch.cat((driving[:, :, 0], depth_driving), 1)
#         # ref_img = torch.cat((ref_img, depth_ref), 1)
#         # print(f"[Input] source_img_depth_add: {source_img.shape}, ref_img_depth_add: {ref_img.shape}")
#
#         # [Input] source_img: torch.Size([24, 3, 104, 80]), ref_img: torch.Size([24, 15, 104, 80])
#         # [After source_in_conv] shape: torch.Size([24, 256, 26, 20])
#         # [After ref_in_conv] shape: torch.Size([24, 256, 26, 20])
#         # [After trans_conv] img_para shape: torch.Size([24, 128, 2, 2])  cat
#         # [After audio_encoder] audio_para shape: torch.Size([24, 128, 2])
#         # [After concat] trans_para shape: torch.Size([24, 256])
#         # [After AdaAT] ref_trans_feature shape: torch.Size([24, 256, 26, 20])
#
#         # print(f"[Input] source_img: {source_img.shape}, ref_img: {ref_img.shape}")
#         ## source image encoder
#
#         # print(f"[Before source_in_conv]")
#         source_in_feature = self.source_in_conv(source_img)
#         #print(f"[After source_in_conv] shape: {source_in_feature.shape}")
#         ## reference image encoder
#         # print(f"[Before ref_in_conv] ")
#         ref_in_feature = self.ref_in_conv(ref_img)
#         #print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
#
#         ## alignment encoder
#         img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
#         #print(f"[After trans_conv] img_para shape: {img_para.shape}")
#         img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)  # 四维变二维
#         #print(f"[After trans_conv2wei] img_para shape: {img_para.shape}")
#
#         ## audio encoder
#         audio_para = self.audio_encoder(audio_feature)
#         #print(f"[After audio_encoder] audio_para shape: {audio_para.shape}")
#         audio_para = self.global_avg1d(audio_para).squeeze(2)
#         #print(f"[audio_encoder_global] audio_para shape: {audio_para.shape}")
#         ## concat alignment feature and audio feature
#         #trans_para = torch.cat([img_para, audio_para], 1)
#         # 修改forward中的融合部分
#         trans_para = torch.cat([
#             self.av_attn(img_para, audio_para),
#             audio_para
#         ], dim=1)
#         #print(f"[trans_para] trans_para shape: {trans_para.shape}")
#         trans_para = self.fusion_fc(trans_para)
#         #print(f"[trans_para fusion_fc] trans_para shape: {trans_para.shape}")
#         ## use AdaAT do spatial deformation on reference feature maps
#         ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
#         #print(f"[Aref_trans_feature] : {ref_trans_feature.shape}")
#         #print(f"[trans_para] : {trans_para.shape}")
#         ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
#         # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
#         ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
#         ## feature decoder
#         merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
#         out = self.out_conv(merge_feature)
#         return out
#
#
#
# #
# # [Input] source_img: torch.Size([12, 3, 104, 80]), ref_img: torch.Size([12, 15, 104, 80])
# # [Input] source_img: torch.Size([12, 3, 104, 80]), ref_img: torch.Size([12, 15, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # Before source depth_encoder
# # After source depth_encoder
# # Before ref depth_encoder
# # After ref  depth_encoder
# # [Input] source_img_depth: torch.Size([12, 1, 128, 96]), ref_img_depth: torch.Size([12, 1, 128, 96])
# # [Input] source_img_depth_interpolate: torch.Size([12, 1, 104, 80]), ref_img_depth: torch.Size([12, 1, 104, 80])
# # [Input] source_img_depth_add: torch.Size([12, 4, 104, 80]), ref_img_depth_add: torch.Size([12, 4, 104, 80])
# # source_in_conv_before
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # Before source depth_encoder
#
#
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# import math
# import cv2
# import numpy as np
# from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
# from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
#
#
# def make_coordinate_grid_3d(spatial_size, type):
#     '''
#         generate 3D coordinate grid
#     '''
#     d, h, w = spatial_size
#     x = torch.arange(w).type(type)
#     y = torch.arange(h).type(type)
#     z = torch.arange(d).type(type)
#     x = (2 * (x / (w - 1)) - 1)
#     y = (2 * (y / (h - 1)) - 1)
#     z = (2 * (z / (d - 1)) - 1)
#     yy = y.view(1, -1, 1).repeat(d, 1, w)
#     xx = x.view(1, 1, -1).repeat(d, h, 1)
#     zz = z.view(-1, 1, 1).repeat(1, h, w)
#     meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
#     return meshed, zz
#
#
# class ResBlock1d(nn.Module):
#     '''
#         basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock1d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv1d(in_features, out_features, 1)
#         self.norm1 = BatchNorm1d(in_features)
#         self.norm2 = BatchNorm1d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class ResBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock2d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv2d(in_features, out_features, 1)
#         self.norm1 = BatchNorm2d(in_features)
#         self.norm2 = BatchNorm2d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class UpBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(UpBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = F.interpolate(x, scale_factor=2)
#         out = self.conv(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out
#
#
# class DownBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(DownBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=2)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class DownBlock2d(nn.Module):
#     # basic block
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
#         super(DownBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=stride)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # print(f"\n[DownBlock2d] Input shape: {x.shape}")  # 新增
#         out = self.conv(x)
#         # print(f"[DownBlock2d] Output shape: {out.shape}")  # 新增
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(SameBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(SameBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
#
# class AdaAT(nn.Module):
#
#      #  AdaAT operator
#
#
#     def __init__(self, para_ch=384, feature_ch=256):  #(self, para_ch, feature_ch)
#         super(AdaAT, self).__init__()
#         self.para_ch = para_ch
#         self.feature_ch = feature_ch
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU()
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Sigmoid()
#         )
#         self.rotation = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Tanh()
#         )
#         self.translation = nn.Sequential(
#             nn.Linear(para_ch, 2 * feature_ch),
#             nn.Tanh()
#         )
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature_map, para_code):
#         batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
#         para_code = self.commn_linear(para_code)
#         scale = self.scale(para_code).unsqueeze(-1) * 2
#         angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  #
#         rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
#         rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
#         translation = self.translation(para_code).view(batch, self.feature_ch, 2)
#         grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
#         grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
#         grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
#         scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
#         translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
#         full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
#         trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid, mode='bilinear',padding_mode='border',align_corners=True).squeeze(1)
#
#         return trans_feature
#
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=4):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 全连接层（含降维与恢复）
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         B, C, H, W = x.size()
#
#         # 平均池化分支
#         avg_out = self.avg_pool(x).view(B, C)
#         avg_out = self.fc(avg_out).view(B, C, 1, 1)
#
#         # 最大池化分支
#         max_out = self.max_pool(x).view(B, C)
#         max_out = self.fc(max_out).view(B, C, 1, 1)
#
#         # 合并注意力权重
#         return x * (avg_out + max_out)
#
#
# class CrossModalFusion(nn.Module):
#     def __init__(self, img_ch=3, depth_ch=1, out_ch=4):
#         super().__init__()
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(img_ch + depth_ch, out_ch, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ChannelAttention(out_ch)  # 使用定义好的通道注意力
#         )
#
#     def forward(self, img, depth):
#         x = torch.cat([img, depth], dim=1)
#         return self.fuse_conv(x)
#
#
# class AudioVisualAttention(nn.Module):
#     def __init__(self, img_dim=128, audio_dim=128):
#         super().__init__()
#         self.query = nn.Linear(img_dim, 64)
#         self.key = nn.Linear(audio_dim, 64)
#         self.value = nn.Linear(audio_dim, img_dim)
#
#     def forward(self, img_feat, audio_feat):
#         # img_feat: [B,128], audio_feat: [B,128]
#         q = self.query(img_feat)  # [B,64]
#         k = self.key(audio_feat)  # [B,64]
#         v = self.value(audio_feat)  # [B,128]
#         attn = F.softmax(q @ k.T, dim=1)  # [B,B]
#         return img_feat + (attn @ v)  # [B,128]
#
#
# class DFSA(nn.Module):
#     def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
#         super(DFSA, self).__init__()
#         self.source_in_conv = nn.Sequential(
#             SameBlock2d(4, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1)
#         )
#
#         # 1x1 卷积来调整 ref_img 通道数
#         # self.ref_channel_adjust = nn.Conv2d(ref_channel, 3, kernel_size=1, stride=1, padding=0)
#
#         self.ref_in_conv = nn.Sequential(
#             SameBlock2d(20, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1),
#         )
#
#         # self.trans_conv = nn.Sequential(
#         #     # 20 →10
#         #     SameBlock2d(512, 128, kernel_size=3, padding=1),
#         #     SameBlock2d(128, 128, kernel_size=11, padding=5),
#         #     SameBlock2d(128, 128, kernel_size=11, padding=5),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 10 →5
#         #     SameBlock2d(128, 128, kernel_size=7, padding=3),
#         #     SameBlock2d(128, 128, kernel_size=7, padding=3),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 5 →3
#         #     SameBlock2d(128, 128, kernel_size=3, padding=1),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         #     # 3 →2
#         #     SameBlock2d(128, 128, kernel_size=3, padding=1),
#         #     DownBlock2d(128, 128, kernel_size=3, padding=1),
#         # )
#
#         # 修改trans_conv结构减少下采样
#         self.trans_conv = nn.Sequential(
#             SameBlock2d(512, 256, kernel_size=3, padding=1),
#             ResBlock2d(256, 256, kernel_size=3, padding=1),
#             DownBlock2d(256, 256, kernel_size=3, padding=1),  # 仅2次下采样
#             ResBlock2d(256, 128, kernel_size=3, padding=1),
#             ResBlock2d(128, 128, kernel_size=3, padding=1),
#         )
#
#         self.audio_encoder = nn.Sequential(
#             SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             SameBlock1d(128, 128, kernel_size=3, padding=1)
#         )
#
#         appearance_conv_list = []
#         for i in range(2):
#             appearance_conv_list.append(
#                 nn.Sequential(
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                 )
#             )
#         self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
#         self.adaAT = AdaAT(para_ch=384, feature_ch=256)
#         self.out_conv = nn.Sequential(
#             SameBlock2d(512, 128, kernel_size=3, padding=1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             ResBlock2d(128, 128, 3, 1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
#             nn.Sigmoid()
#         )
#         self.global_avg2d = nn.AdaptiveAvgPool2d(1)
#         self.global_avg1d = nn.AdaptiveAvgPool1d(1)
#
#         self.depth_encoder = depth_encoder
#         self.depth_decoder = depth_decoder
#
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(256, 384),
#             nn.ReLU())
#         self.av_attn = AudioVisualAttention()
#         self.source_fusion = CrossModalFusion(img_ch=3, depth_ch=1, out_ch=4)
#         self.ref_fusion = CrossModalFusion(img_ch=15, depth_ch=5, out_ch=20)
#         # self.out_conv = nn.Sequential(
#         #     SameBlock2d(512, 256, kernel_size=3),
#         #     nn.PixelShuffle(2),  # 亚像素卷积提升分辨率
#         #     ResBlock2d(64, 64, kernel_size=3),
#         #     nn.Conv2d(64, 3, kernel_size=7, padding=3))
#
#     # def normalize_depth_input(self, img):
#     #     # 假设输入是 [0,1]，转为标准ImageNet归一化
#     #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
#     #     std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
#     #     return (img - mean) / std
#
#     def forward(self, source_img, ref_img, audio_feature):
#         # # 输入 ref_img 之前，调整通道数
#         # #device = torch.device("cuda:0")
#         # #source_img = source_img.to(device)
#         # #ref_img = ref_img.to(device)
#         # #audio_feature = audio_feature.to(device)
#         # ref_img = self.ref_channel_adjust(ref_img)
#         # print(f"ref_img_3: {ref_img.shape}")
#         # ref_img = self.normalize_depth_input(ref_img)  # 先归一化
#         # print(f"ref_img_3_normal: {ref_img.shape}")
#         # #depth_encoder = depth_encoder.to(device)
#         # #depth_decoder = depth_decoder.to(device)
#         # # 统一调整输入尺寸到32的倍数（示例调整为128x96）
#         # def adjust_size(x, target_h=128, target_w=96):
#         #     return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)
#         #     # 调整输入尺寸
#         # source_img_adj = adjust_size(source_img)
#         # ref_img_adj = adjust_size(ref_img)
#         #     # 用调整后的尺寸进行深度估计
#         # print(next(depth_encoder.parameters()).device)  # 应为cuda:0
#         # print(source_img.device)  # 应与模型一致
#         # with torch.no_grad():  # 关闭梯度以节省显存
#         #     print("Before source depth_encoder")
#         #     depth_source = depth_decoder(depth_encoder(source_img_adj))[("disp", 0)]
#         #     print("After source depth_encoder")
#         #     print("Before ref depth_encoder")
#         #     depth_ref = depth_decoder(depth_encoder(ref_img_adj))[("disp", 0)]
#         #     print("After ref depth_encoder")
#         #
#         #     # 插值回原尺寸（如果需要与原图拼接）
#         # depth_source = adjust_size(depth_source, source_img.shape[2], source_img.shape[3])
#         # depth_ref = adjust_size(depth_ref, ref_img.shape[2], ref_img.shape[3])
#         #
#         # # 后续拼接和网络前向保持不变
#         # source_img = torch.cat((source_img, depth_source), 1)
#         # ref_img = torch.cat((ref_img, depth_ref), 1)
#         # print(f"[Input] source_img_depth_add: {source_img.shape}, ref_img_depth_add: {ref_img.shape}")
#
#         ## source image encoder
#         source_img.float()
#         ref_img.float()
#         # print(f"[Input] source_img: {source_img.shape}, ref_img: {ref_img.shape}")
#
#         # 输入 ref_img 之前，调整通道数
#         # ref_img = self.ref_channel_adjust(ref_img)  # 1
#         # print(f"ref_img_3: {ref_img.shape}")
#         # ref_img = self.normalize_depth_input(ref_img)  # 先归一化  #1
#         # print(f"ref_img_3: {ref_img.shape}")
#
#         # 调整ref
#         # 假设 ref_img 形状为 [Batch, 15, H, W]
#         batch_size, _, H, W = ref_img.shape
#         # 拆分为5个独立3通道图像 [Batch, 5, 3, H, W]
#         ref_imgs = ref_img.view(batch_size, 5, 3, H, W)
#         # 方法一：通道拼接+降维 (适合decoder接受高维输入)
#         depth_features = []
#         for i in range(5):
#             feat_encoder = self.depth_encoder(ref_imgs[:, i])  # [B, C, H', W']
#             feat_decoder = self.depth_decoder(feat_encoder)
#             feat_features = feat_decoder[("disp", 0)]
#             # print("feat_decoder", feat_features.shape)
#             depth_features.append(feat_features)
#             # depth_features.append(feat_decoder)
#         # 拼接特征并降维
#         combined_feat = torch.cat(depth_features, dim=1)  # [B, 5C, H', W']
#         # print("combined_feat", combined_feat.shape)
#         # fusion_conv = nn.Conv2d(5 * 1, 1, kernel_size=1)  # 添加适配层
#         # final_feat = fusion_conv(combined_feat)
#         # depth_ref = combined_feat[("disp", 0)]
#         depth_ref = combined_feat
#         # print("depth_ref0", depth_ref.shape)
#         depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",
#                                   align_corners=True)
#         # depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bicubic",align_corners=True)
#         # print("depth_ref", depth_ref.shape)
#         # ref_img = torch.cat((ref_img, depth_ref), 1)
#
#         ref_img = self.ref_fusion(ref_img, depth_ref)
#         # print("ref_img_add", ref_img.shape)
#
#         # print("depth",next(self.depth_encoder.parameters()).device)  # 应为cuda:0
#         # print("source",source_img.device)  # 应与模型一致
#
#         # print("Before source depth_encoder")
#         outputs = self.depth_decoder(self.depth_encoder(source_img))
#         depth_source = outputs[("disp", 0)]
#         # print("After source depth_encoder")
#         # print("depth",next(self.depth_encoder.parameters()).device)  # 应为cuda:0
#         # print("source",source_img.device)  # 应与模型一致
#         # print("Before ref depth_encoder")
#         # outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
#         # outputs = self.depth_decoder(self.depth_encoder(ref_img))  # 1
#         # depth_ref = outputs[("disp", 0)]   # 1
#         # print("After ref  depth_encoder")
#         # print(f"[Input] source_img_depth: {depth_source.shape}, ref_img_depth: {depth_ref.shape}")
#         # depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode="bilinear",align_corners=True)
#         # 改用自适应插值或引导滤波
#         depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode='bilinear',
#                                      align_corners=True)
#         # depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",align_corners=True) #  1
#         # print(f"[Input] source_img_depth_interpolate: {depth_source.shape}, ref_img_depth: {depth_ref.shape}")
#
#         # source_img = torch.cat((source_img, depth_source), 1)
#         source_img = self.source_fusion(source_img, depth_source)
#         # ref_img = torch.cat((driving[:, :, 0], depth_driving), 1)
#         # ref_img = torch.cat((ref_img, depth_ref), 1)
#         # print(f"[Input] source_img_depth_add: {source_img.shape}, ref_img_depth_add: {ref_img.shape}")
#
#         # [Input] source_img: torch.Size([24, 3, 104, 80]), ref_img: torch.Size([24, 15, 104, 80])
#         # [After source_in_conv] shape: torch.Size([24, 256, 26, 20])
#         # [After ref_in_conv] shape: torch.Size([24, 256, 26, 20])
#         # [After trans_conv] img_para shape: torch.Size([24, 128, 2, 2])  cat
#         # [After audio_encoder] audio_para shape: torch.Size([24, 128, 2])
#         # [After concat] trans_para shape: torch.Size([24, 256])
#         # [After AdaAT] ref_trans_feature shape: torch.Size([24, 256, 26, 20])
#
#         # print(f"[Input] source_img: {source_img.shape}, ref_img: {ref_img.shape}")
#         ## source image encoder
#
#         # print(f"[Before source_in_conv]")
#         source_in_feature = self.source_in_conv(source_img)
#         # print(f"[After source_in_conv] shape: {source_in_feature.shape}")
#         ## reference image encoder
#         # print(f"[Before ref_in_conv] ")
#         ref_in_feature = self.ref_in_conv(ref_img)
#         # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
#
#         ## alignment encoder
#         img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
#         # print(f"[After trans_conv] img_para shape: {img_para.shape}")
#         img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)  # 四维变二维
#         # print(f"[After trans_conv2wei] img_para shape: {img_para.shape}")
#
#         ## audio encoder
#         audio_para = self.audio_encoder(audio_feature)
#         # print(f"[After audio_encoder] audio_para shape: {audio_para.shape}")
#         audio_para = self.global_avg1d(audio_para).squeeze(2)
#         ## concat alignment feature and audio feature
#         # trans_para = torch.cat([img_para, audio_para], 1)
#         # 修改forward中的融合部分
#         trans_para = torch.cat([
#             self.av_attn(img_para, audio_para),
#             audio_para
#         ], dim=1)
#         trans_para = self.fusion_fc(trans_para)
#         # print(f"[After concat] trans_para shape: {trans_para.shape}")
#         ## use AdaAT do spatial deformation on reference feature maps
#         ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
#         ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
#         # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
#         ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
#         ## feature decoder
#         merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
#         out = self.out_conv(merge_feature)
#         return out
#
# #
# # [Input] source_img: torch.Size([12, 3, 104, 80]), ref_img: torch.Size([12, 15, 104, 80])
# # [Input] source_img: torch.Size([12, 3, 104, 80]), ref_img: torch.Size([12, 15, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # Before source depth_encoder
# # After source depth_encoder
# # Before ref depth_encoder
# # After ref  depth_encoder
# # [Input] source_img_depth: torch.Size([12, 1, 128, 96]), ref_img_depth: torch.Size([12, 1, 128, 96])
# # [Input] source_img_depth_interpolate: torch.Size([12, 1, 104, 80]), ref_img_depth: torch.Size([12, 1, 104, 80])
# # [Input] source_img_depth_add: torch.Size([12, 4, 104, 80]), ref_img_depth_add: torch.Size([12, 4, 104, 80])
# # source_in_conv_before
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # ref_img_3: torch.Size([12, 3, 104, 80])
# # Before source depth_encoder
#
#
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# import math
# import cv2
# import numpy as np
# from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
# from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
#
#
# def make_coordinate_grid_3d(spatial_size, type):
#     '''
#         generate 3D coordinate grid
#     '''
#     d, h, w = spatial_size
#     x = torch.arange(w).type(type)
#     y = torch.arange(h).type(type)
#     z = torch.arange(d).type(type)
#     x = (2 * (x / (w - 1)) - 1)
#     y = (2 * (y / (h - 1)) - 1)
#     z = (2 * (z / (d - 1)) - 1)
#     yy = y.view(1, -1, 1).repeat(d, 1, w)
#     xx = x.view(1, 1, -1).repeat(d, h, 1)
#     zz = z.view(-1, 1, 1).repeat(1, h, w)
#     meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
#     return meshed, zz
#
#
# class ResBlock1d(nn.Module):
#     '''
#         basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock1d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv1d(in_features, out_features, 1)
#         self.norm1 = BatchNorm1d(in_features)
#         self.norm2 = BatchNorm1d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class ResBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(ResBlock2d, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                                padding=padding)
#         if out_features != in_features:
#             self.channel_conv = nn.Conv2d(in_features, out_features, 1)
#         self.norm1 = BatchNorm2d(in_features)
#         self.norm2 = BatchNorm2d(in_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.norm1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.norm2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.in_features != self.out_features:
#             out += self.channel_conv(x)
#         else:
#             out += x
#         return out
#
#
# class UpBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(UpBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = F.interpolate(x, scale_factor=2)
#         out = self.conv(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out
#
#
# class DownBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(DownBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=2)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class DownBlock2d(nn.Module):
#     # basic block
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
#         super(DownBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, stride=stride)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # print(f"\n[DownBlock2d] Input shape: {x.shape}")  # 新增
#         out = self.conv(x)
#         # print(f"[DownBlock2d] Output shape: {out.shape}")  # 新增
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock1d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size, padding):
#         super(SameBlock1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm1d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class SameBlock2d(nn.Module):
#     '''
#             basic block
#     '''
#
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1):
#         super(SameBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding)
#         self.norm = BatchNorm2d(out_features)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out
#
#
# class AdaAT(nn.Module):
#     #  AdaAT operator
#     def __init__(self, para_ch, feature_ch):  # (self, para_ch, feature_ch)
#         super(AdaAT, self).__init__()
#         self.para_ch = para_ch
#         self.feature_ch = feature_ch
#         self.commn_linear = nn.Sequential(
#             nn.Linear(para_ch, para_ch),
#             nn.ReLU()
#         )
#         self.scale = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Sigmoid()
#         )
#         self.rotation = nn.Sequential(
#             nn.Linear(para_ch, feature_ch),
#             nn.Tanh()
#         )
#         self.translation = nn.Sequential(
#             nn.Linear(para_ch, 2 * feature_ch),
#             nn.Tanh()
#         )
#         self.tanh = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, feature_map, para_code):
#         batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
#         para_code = self.commn_linear(para_code)
#         scale = self.scale(para_code).unsqueeze(-1) * 2
#         angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  #
#         rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
#         rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
#         translation = self.translation(para_code).view(batch, self.feature_ch, 2)
#         grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
#         grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
#         grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
#         scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
#         translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
#         trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
#         full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
#         trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
#
#         return trans_feature
#
#
#
# class DFSA(nn.Module):
#     def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
#         super(DFSA, self).__init__()
#         self.source_in_conv = nn.Sequential(
#             SameBlock2d(3, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1)
#         )
#         self.ref_in_conv = nn.Sequential(
#             SameBlock2d(15, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1),
#         )
#         self.depth_conv = nn.Sequential(
#             SameBlock2d(1, 64, kernel_size=7, padding=3),
#             DownBlock2d(64, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 256, kernel_size=3, padding=1),
#         )
#
#         self.trans_conv = nn.Sequential(
#             # 20 →10
#             SameBlock2d(512, 128, kernel_size=3, padding=1),
#             SameBlock2d(128, 128, kernel_size=11, padding=5),
#             SameBlock2d(128, 128, kernel_size=11, padding=5),
#             DownBlock2d(128, 128, kernel_size=3, padding=1),
#             # 10 →5
#             SameBlock2d(128, 128, kernel_size=7, padding=3),
#             SameBlock2d(128, 128, kernel_size=7, padding=3),
#             DownBlock2d(128, 128, kernel_size=3, padding=1),
#             # 5 →3
#             SameBlock2d(128, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 128, kernel_size=3, padding=1),
#             # 3 →2
#             SameBlock2d(128, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 128, kernel_size=3, padding=1),
#         )
#
#         self.audio_encoder = nn.Sequential(
#             SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             SameBlock1d(128, 128, kernel_size=3, padding=1)
#         )
#
#         appearance_conv_list = []
#         for i in range(2):
#             appearance_conv_list.append(
#                 nn.Sequential(
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                 )
#             )
#         self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
#         self.adaAT = AdaAT(para_ch=256, feature_ch=256)
#         self.out_conv = nn.Sequential(
#             SameBlock2d(512, 128, kernel_size=3, padding=1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             ResBlock2d(128, 128, 3, 1),
#             UpBlock2d(128, 128, kernel_size=3, padding=1),
#             nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
#             nn.Sigmoid()
#         )
#         self.global_avg2d = nn.AdaptiveAvgPool2d(1)
#         self.global_avg1d = nn.AdaptiveAvgPool1d(1)
#
#         self.depth_encoder = depth_encoder
#         self.depth_decoder = depth_decoder
#
#         self.audio_fc = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.Sigmoid()
#         )
#
#     def forward(self, source_img, ref_img, audio_feature):
#         ## source image encoder
#         source_img.float()
#         ref_img.float()
#         # 调整ref
#         # 假设 ref_img 形状为 [Batch, 15, H, W]
#         # batch_size, _, H, W = ref_img.shape
#         # # 拆分为5个独立3通道图像 [Batch, 5, 3, H, W]
#         # ref_imgs = ref_img.view(batch_size, 5, 3, H, W)
#         # # 方法一：通道拼接+降维 (适合decoder接受高维输入)
#         # depth_features = []
#         # for i in range(5):
#         #     feat_encoder = self.depth_encoder(ref_imgs[:, i])  # [B, C, H', W']
#         #     feat_decoder = self.depth_decoder(feat_encoder)
#         #     feat_features = feat_decoder[("disp", 0)]
#         #     depth_features.append(feat_features)
#         # # 拼接特征并降维
#         # # combined_feat = torch.cat(depth_features, dim=1)  # [B, 5C, H', W']
#         # # depth_ref = combined_feat
#         #
#         # depth_stack = torch.stack(depth_features, dim=1).squeeze(2)
#         # depth_ref = torch.mean(depth_stack, dim=1, keepdim=True)  # [B, 1, H, W]
#         # depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",
#         #                           align_corners=True)
#         # # ref_img = torch.cat([ref_img, depth_ref],dim=1)
#         #
#         # # print("Before source depth_encoder")
#         # outputs = self.depth_decoder(self.depth_encoder(source_img))
#         # depth_source = outputs[("disp", 0)]
#         # # depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode="bilinear",align_corners=True)
#         # # 改用自适应插值或引导滤波
#         # depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode='bilinear',
#         #                              align_corners=True)
#         # # source_img = torch.cat([source_img, depth_source],dim=1)
#
#         ## source image encoder
#         alpha = 0.1
#         # print(f"[Before source_in_conv]")
#         source_in_feature = self.source_in_conv(source_img)
#         # source_depth_feature = self.depth_conv(depth_source)
#         # source_in_feature = source_in_feature + alpha * source_depth_feature  # 加权融合而不是拼接
#
#         # print(f"[After source_in_conv] shape: {source_in_feature.shape}")
#         ## reference image encoder
#         # print(f"[Before ref_in_conv] ")
#         ref_in_feature = self.ref_in_conv(ref_img)
#         # ref_depth_feature = self.depth_conv(depth_ref)
#         # # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
#         # ref_in_feature = ref_in_feature + alpha * ref_depth_feature  # 加权融合而不是拼接
#
#         ## alignment encoder
#         img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
#         # print(f"[After trans_conv] img_para shape: {img_para.shape}")
#         img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)  # 四维变二维
#         # print(f"[After trans_conv2wei] img_para shape: {img_para.shape}")
#
#         ## audio encoder
#         audio_para = self.audio_encoder(audio_feature)
#         # print(f"[After audio_encoder] audio_para shape: {audio_para.shape}")
#         audio_para = self.global_avg1d(audio_para).squeeze(2)
#         #audio_para = self.audio_fc(audio_para) * audio_para  # 加权增强
#
#         ## concat alignment feature and audio feature
#         trans_para = torch.cat([img_para, audio_para], 1)
#         # print(f"[After concat] trans_para shape: {trans_para.shape}")
#
#         ## use AdaAT do spatial deformation on reference feature maps
#         ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
#         ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
#         # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
#         ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
#
#         ## feature decoder
#         merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
#         out = self.out_conv(merge_feature)
#         return out
#
# '''
# class LearnableWaveletConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_low = nn.Conv2d(in_channels, out_channels, 2, stride=2)
#         self.conv_high = nn.Conv2d(in_channels, 3 * out_channels, 2, stride=2)
#
#         # 用Haar小波初始化
#         with torch.no_grad():
#             # LL分量
#             self.conv_low.weight.data = torch.ones_like(self.conv_low.weight) * 0.5
#             # 高频分量(LH, HL, HH)
#             high_weights = torch.tensor([[0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
#             self.conv_high.weight.data[:3 * out_channels] = high_weights.repeat(out_channels, 1, 1, 1).float()
#
#     def forward(self, x):
#         ll = self.conv_low(x)
#         high = self.conv_high(x)  # [B,3C,H/2,W/2]
#         return torch.cat([ll, high], dim=1)  # [B,4C,H/2,W/2]
#
#
# class InverseWaveletConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
#
#         with torch.no_grad():
#             self.deconv.weight.data = torch.eye(2).repeat(out_channels, in_channels // 4, 1, 1).float() * 0.5
#
#     def forward(self, x):
#         return self.deconv(x)
#
#
# class WaveletFusion(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.low_gate = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.high_gate = nn.Sequential(
#             nn.Conv2d(3 * channels, channels, 3, padding=1),
#             nn.Sigmoid()
#         )
#         self.high_transform = nn.Conv2d(3 * channels, channels, 3, padding=1)
#
#     def forward(self, x):
#         ll, lh, hl, hh = torch.chunk(x, 4, dim=1)
#
#         # 低频处理
#         low_feat = ll * self.low_gate(ll)
#
#         # 高频融合
#         high_cat = torch.cat([lh, hl, hh], dim=1)
#         high_feat = self.high_transform(high_cat) * self.high_gate(high_cat)
#
#         return torch.cat([low_feat, high_feat], dim=1)
#
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel//reduction),
#             nn.ReLU(),
#             nn.Linear(channel//reduction, channel),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         avg = self.avg_pool(x).view(b,c)
#         max_ = self.max_pool(x).view(b,c)
#         avg_out = self.fc(avg).view(b,c,1,1)
#         max_out = self.fc(max_).view(b,c,1,1)
#         return x * (avg_out + max_out)
#
# # 在ResBlock2d中插入
# class EnhancedResBlock2d(ResBlock2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.ca = ChannelAttention(self.out_features)
#
#     def forward(self, x):
#         out = super().forward(x)
#         return self.ca(out)
#
# class EnhancedAdaAT(AdaAT):
#     def __init__(self, para_ch, feature_ch):
#         super().__init__(para_ch, feature_ch)
#         # 多尺度参数生成（增加权重归一化）
#         self.multi_scale = nn.Sequential(
#             nn.Linear(para_ch, para_ch // 2),
#             nn.ReLU(),
#             nn.Linear(para_ch // 2, 3),
#             nn.Softmax(dim=1)  # 保证权重和为1
#         )
#
#     def forward(self, feature_map, para_code):
#         # 原始单尺度变换
#         base_trans = super().forward(feature_map, para_code)
#
#         # 多尺度处理
#         B, C, H, W = feature_map.shape
#
#         # 生成多尺度金字塔（包含原始尺度）
#         pyramid = []
#         for i in range(3):
#             # 下采样因子：1, 2, 4
#             if i == 0:
#                 pyramid.append(feature_map)  # 原始尺度
#             else:
#                 pyramid.append(F.avg_pool2d(feature_map, kernel_size=2 ** i, stride = 2 ** i))
#
#         # 对各尺度进行变换并上采样回原始尺寸
#         trans_pyramid = []
#         for i, p in enumerate(pyramid):
#             # 执行AdaAT变换
#             trans_feat = super().forward(p, para_code)
#
#             # 上采样到原始尺寸
#             if i > 0:
#                 trans_feat = F.interpolate(
#                     trans_feat,
#                     size=(H, W),
#                     mode='bilinear',
#                     align_corners=False
#                 )
#             trans_pyramid.append(trans_feat)
#
#         # 获取可学习的尺度权重 [B,3]
#         scale_weights = self.multi_scale(para_code)  # 已用softmax归一化
#
#         # 加权融合（扩展权重维度匹配特征图）
#         weighted_features = []
#         for w, t in zip(scale_weights.unbind(dim=1), trans_pyramid):
#             weighted_features.append(w.view(B, 1, 1, 1) * t)
#
#         final_trans = sum(weighted_features)
#
#         # 残差连接
#         return base_trans + final_trans
#
# class DFSA(nn.Module):
#     def __init__(self, source_channel=3, ref_channel=15, audio_channel=256):
#         super().__init__()
#
#         # --------------------- 源图像编码器 ---------------------
#         self.source_in_conv = nn.Sequential(
#             LearnableWaveletConv(3, 64),  # [B,256,H/2,W/2]
#             WaveletFusion(64),
#             DownBlock2d(256, 128, kernel_size=3, padding=1),
#             LearnableWaveletConv(128, 128),  # [B,512,H/4,W/4]
#             WaveletFusion(128),
#             DownBlock2d(512, 256, kernel_size=3, padding=1)
#         )
#
#         # --------------------- 参考图像编码器 ---------------------
#         self.ref_in_conv = nn.Sequential(
#             LearnableWaveletConv(15, 64),  # [B,256,H/2,W/2]
#             WaveletFusion(64),
#             DownBlock2d(256, 128, kernel_size=3, padding=1),
#             LearnableWaveletConv(128, 128),  # [B,512,H/4,W/4]
#             WaveletFusion(128),
#             DownBlock2d(512, 256, kernel_size=3, padding=1)
#         )
#
#         # --------------------- 对齐变换器 ---------------------
#         self.trans_conv = nn.Sequential(
#             SameBlock2d(512, 128, kernel_size=3, padding=1),
#             SameBlock2d(128, 128, kernel_size=11, padding=5),
#             DownBlock2d(128, 128, kernel_size=3, padding=1),
#             DownBlock2d(128, 128, kernel_size=3, padding=1)
#         )
#
#         # --------------------- 音频编码器 ---------------------
#         self.audio_encoder = nn.Sequential(
#             DownBlock1d(256, 128, 5, 2),
#             ResBlock1d(128, 128, 3, 1),
#             DownBlock1d(128, 128, 3, 1),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         #self.audio_fc = nn.Linear(128, 256)
#
#         # --------------------- 自适应变形模块 ---------------------
#         self.adaAT = AdaAT(256, 256)
#         # self.appearance_conv = nn.Sequential(
#         #     ResBlock2d(256, 256, 3, 1),
#         #     ResBlock2d(256, 256, 3, 1)
#         # )
#         appearance_conv_list = []
#         for i in range(2):
#             appearance_conv_list.append(
#                 nn.Sequential(
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                     ResBlock2d(256, 256, 3, 1),
#                 )
#             )
#         self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
#         # # 替换原始ResBlock2d
#         # self.appearance_conv_list = nn.ModuleList([
#         #     nn.Sequential(
#         #         EnhancedResBlock2d(256,256,3,1),  # 使用增强版
#         #         EnhancedResBlock2d(256,256,3,1),
#         #         EnhancedResBlock2d(256,256,3,1),
#         #         EnhancedResBlock2d(256,256,3,1)
#         #     ) for _ in range(2)
#         # ])
#
#         # --------------------- 解码器 ---------------------
#         self.decoder = nn.Sequential(
#             InverseWaveletConv(768, 256),  # 512+256
#             ResBlock2d(256, 128, 3, 1),
#             UpBlock2d(128, 64),
#             InverseWaveletConv(192, 64),
#             ResBlock2d(64, 32, 3, 1),
#             nn.Conv2d(32, 3, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#         # --------------------- 通用模块 ---------------------
#         self.global_avg2d = nn.AdaptiveAvgPool2d(1)
#         self.global_avg1d = nn.AdaptiveAvgPool1d(1)
#
#     def _encode_image(self, x, is_ref=False):
#         features = []
#         conv_path = self.ref_in_conv if is_ref else self.source_in_conv
#         for layer in conv_path:
#             if isinstance(layer, LearnableWaveletConv):
#                 x = layer(x)
#                 x = F.relu(x)
#                 features.append(x)
#             elif isinstance(layer, WaveletFusion):
#                 x = layer(x)
#             else:
#                 x = layer(x)
#         return x, features
#
#     def forward(self, source_img, ref_img, audio_feature):
#         # --------------------- 双路小波编码 ---------------------
#         src_feat, src_wavelets = self._encode_image(source_img) # 256
#         ref_feat, ref_wavelets = self._encode_image(ref_img, is_ref=True)
#
#         # --------------------- 对齐特征提取 ---------------------
#         trans_feat = self.trans_conv(torch.cat([src_feat, ref_feat], dim=1))
#         img_para = self.global_avg2d(trans_feat).squeeze()      #128
#
#         # --------------------- 音频特征提取 ---------------------
#         audio_para = self.audio_encoder(audio_feature).squeeze() # 128
#         #audio_para = self.audio_fc(audio_para)     # 256
#
#         # --------------------- 自适应变形 ---------------------
#         trans_para = torch.cat([img_para, audio_para], dim=1)  # 128+128=256
#         ref_trans = self.appearance_conv_list[0](ref_feat)
#         warped_ref = self.adaAT(ref_trans, trans_para)
#         warped_ref = self.appearance_conv_list[1](warped_ref)  #256
#
#         # --------------------- 多尺度特征融合 ---------------------
#         fused_feat = torch.cat([src_feat, warped_ref], dim=1)  # 256+256=512
#
#         # 融合小波特征
#         for sw, rw in zip(src_wavelets, ref_wavelets):
#             fused_feat += F.interpolate(sw + rw, fused_feat.shape[2:])
#
#         # --------------------- 小波解码 ---------------------
#         return self.decoder(fused_feat)
#
# '''
import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
import depth
from models.FreqFusion import FreqFusion
def make_coordinate_grid_3d(spatial_size, type):
    '''
        generate 3D coordinate grid
    '''
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed, zz


class ResBlock1d(nn.Module):
    '''
        basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class ResBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out

class UpBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock1d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=2)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class DownBlock2d(nn.Module):
    # basic block

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(f"\n[DownBlock2d] Input shape: {x.shape}")  # 新增
        out = self.conv(x)
        # print(f"[DownBlock2d] Output shape: {out.shape}")  # 新增
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock1d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock2d(nn.Module):
    '''
            basic block
    '''

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class AdaAT(nn.Module):
     #  AdaAT operator
    def __init__(self, para_ch, feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(
            nn.Linear(para_ch, para_ch),
            nn.ReLU()
        )
        self.scale = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Sigmoid()
        )
        self.rotation = nn.Sequential(
            nn.Linear(para_ch, feature_ch),
            nn.Tanh()
        )
        self.translation = nn.Sequential(
            nn.Linear(para_ch, 2 * feature_ch),
            nn.Tanh()
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, feature_map, para_code):

        batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)

        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  #
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)

        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)

        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)

        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation

        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid, mode='bilinear').squeeze(1)

        return trans_feature

class AdaIN_Block(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(AdaIN_Block, self).__init__()
        self.eps = eps
        # 通道注意：这里 style 特征会用于生成 gamma 和 beta
        self.style_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(1),  # [B, C]
            nn.Linear(channels, channels * 2),  # 输出 gamma 和 beta
        )

    def calc_mean_std(self, feat):
        # 计算通道维度上的均值和标准差：用于归一化 content
        B, C, H, W = feat.size()
        feat_var = feat.view(B, C, -1).var(dim=2, unbiased=False) + self.eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content, style):
        # 计算 style 的 gamma 和 beta
        style_stats = self.style_fc(style)  # [B, 2C]
        gamma, beta = style_stats.chunk(2, dim=1)  # [B, C] x2
        gamma = gamma.view(-1, content.size(1), 1, 1)
        beta = beta.view(-1, content.size(1), 1, 1)

        # 内容归一化
        content_mean, content_std = self.calc_mean_std(content)
        normalized_content = (content - content_mean) / content_std

        # 应用 gamma 和 beta
        return normalized_content * gamma + beta

class DFSA(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
        super(DFSA, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1)
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.depth_source_conv = nn.Sequential(
            SameBlock2d(1, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.depth_ref_conv = nn.Sequential(
            SameBlock2d(5, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )

        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )

        # 保留原trans_conv的部分层作为适配器
        # self.adain_conv = nn.Sequential(
        #     SameBlock2d(512, 256, kernel_size=3, padding=1),  # 输出通道改为256
        #     AdaIN_Block(256)
        # )

        self.adain_block = AdaIN_Block(channels=256)  # 假设特征通道为256

        self.trans_conv1 = nn.Sequential(
            # 初始输入尺寸 [280, 256, 26, 20] -> 输出 [280, 128, 26, 20]
            SameBlock2d(256, 128, kernel_size=3, padding=1),  # 通道数从 256 降到 128
            # 保持空间尺寸 [280, 128, 26, 20] -> 输出 [280, 128, 26, 20]
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            # 空间下采样 [280, 128, 26, 20] -> 输出 [280, 128, 13, 10]
            DownBlock2d(128, 128, kernel_size=3, padding=1),

            # 继续减少空间尺寸 [280, 128, 13, 10] -> 输出 [280, 128, 7, 5]
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

            # 再次减少空间尺寸 [280, 128, 7, 5] -> 输出 [280, 128, 4, 3]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

            # 最终下采样到 [280, 128, 2, 2]
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )



        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )
        # # 替换原audio_encoder   transformer
        # self.audio_encoder = AudioTransformer(
        #     input_dim=audio_channel,  # 对应audio_channel=29
        #     d_model=128,  # 保持与原模型通道数一致
        #     nhead=8,
        #     num_layers=4
        # )

        # # 替换原audio_encoder CNN+transformer
        # self.audio_encoder = HybridAudioEncoder(
        #     input_dim=audio_channel,
        #     output_dim=128  # 保持与原模型兼容
        # )

        # self.fusion_source = DynamicChannelFusion(256)
        # self.fusion_ref = DynamicChannelFusion(256)

        # 添加交叉注意力模块
        self.cross_attention_img2audio = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.cross_attention_audio2img = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # 添加FreqFusion模块
        #self.freq_fusion = FreqFusion(256)

        # 融合替换cat
        # self.cross_attn = CrossAttentionFusion(256)
        # self.spatial_gate = SpatialGateFusion()

        # 修改现有的融合模块
        # self.fusion_source = nn.ModuleList([
        #     DynamicChannelFusion(256),
        #     FreqFusion(256, reduction_ratio=8)
        # ])

        # self.fusion_ref = nn.ModuleList([
        #     self.fusion_ref,
        #     self.freq_fusion
        # ])

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)

        self.adaAT = AdaAT(para_ch=256, feature_ch=256)
        #self.adaAT1 = AdaAT1(para_ch=256, feature_ch=256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),     # 512
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        #self.out_conv = DFSADecoder()
        #self.out_conv = HybridDecoder()
        #self.out_conv = Decoder0()


        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder


    def forward(self, source_img, ref_img, audio_feature):

        ######################################################depth image encoder###############################################
        source_img.float()
        ref_img.float()
        # 调整ref
        # 假设 ref_img 形状为 [Batch, 15, H, W]
        batch_size, _, H, W = ref_img.shape
        # 拆分为5个独立3通道图像 [Batch, 5, 3, H, W]
        ref_imgs = ref_img.view(batch_size, 5, 3, H, W)
        # 方法一：通道拼接+降维 (适合decoder接受高维输入)
        depth_features = []
        for i in range(5):
            feat_encoder = self.depth_encoder(ref_imgs[:, i])  # [B, C, H', W']
            feat_decoder = self.depth_decoder(feat_encoder)
            feat_features = feat_decoder[("disp", 0)]
            depth_features.append(feat_features)
        # 拼接特征并降维
        combined_feat = torch.cat(depth_features, dim=1)  # [B, 5C, H', W']
        depth_ref = combined_feat

        depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",align_corners=True)
        #print("depth_ref_inter",depth_ref.shape)


        # print("Before source depth_encoder")
        outputs = self.depth_decoder(self.depth_encoder(source_img))
        depth_source = outputs[("disp", 0)]
        # 改用自适应插值或引导滤波
        depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode='bilinear',align_corners=True)
        #print("depth_ref_inter", depth_ref.shape)

        ################################################### source image encoder#################################################
        alpha = 1
        # print(f"[Before source_in_conv]")
        source_in_feature = self.source_in_conv(source_img)  # [280,256,26,20]
        #print("source_in_feature", source_in_feature.shape)
        source_depth_feature = self.depth_source_conv(depth_source)  # [280,256,26,20]
        #print("source_depth_feature", source_depth_feature.shape)
        source_in_feature = source_in_feature + alpha * source_depth_feature  # 加权融合而不是拼接
        #source_in_feature = self.fusion_source(source_in_feature, source_depth_feature)  # 动态注意力融合
        #source_in_feature = self.freq_fusion(source_in_feature, source_depth_feature)   # [280,256,26,20]
        #print(" source_in_feature_fusion",  source_in_feature.shape)
        #source_in_feature = self.fusion_source[1](source_in_feature, source_depth_feature)

        # print(f"[After source_in_conv] shape: {source_in_feature.shape}")
        ###########################################reference image encoder######################################################
        # print(f"[Before ref_in_conv] ")
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_depth_feature = self.depth_ref_conv(depth_ref)
        # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
        ref_in_feature = ref_in_feature + alpha * ref_depth_feature  # 加权融合而不是拼接
        #ref_in_feature = self.fusion_ref(ref_in_feature, ref_depth_feature)  # 动态注意力融合
        #ref_in_feature = self.freq_fusion(ref_in_feature, ref_depth_feature)   # [280,256,26,20]
        #print(" ref_in_feature_fusion", ref_in_feature.shape)
        #ref_in_feature = self.fusion_ref[1](ref_in_feature, ref_depth_feature)

        ######################################### alignment encoder#############################################################
        #原
        # combined = torch.cat([source_in_feature, ref_in_feature], 1)    #([270, 512, 26, 20])
        # print(f"[After trans_conv1] combined shape: {combined.shape}")
        # img_para = self.trans_conv(combined)     # ([270, 128, 2, 2])
        # print(f"[After trans_conv1] img_para shape: {img_para.shape}")
        #
        # # AdaIN融合替代原trans_conv
        # #combined = torch.cat([source_in_feature, ref_in_feature], 1)  # [B,512,H,W]
        # #print(f"[combined] combined shape: {combined.shape}")
        # img_para = self.adain_conv[0](combined)  # 初始卷积   #[280,256,26,20]
        # print(f"[img_para] img_para shape: {img_para.shape}")
        # img_para = adaptive_instance_normalization(img_para, ref_in_feature)  # 关键融合  # [280,256,26,20]
        # img_para = self.trans_conv1(img_para)   # [280,128,2,2]
        # print(f"[img_para_trans_conv] img_para shape: {img_para.shape}")

        img_para = self.adain_block(content=source_in_feature, style=ref_in_feature)
        #print(f"[img_para] img_para shape: {img_para.shape}")
        img_para = self.trans_conv1(img_para)
        #print(f"[img_para_trans_conv] img_para shape: {img_para.shape}")


        #cross_attn
        # fused_features = self.cross_attn(source_in_feature, ref_in_feature)
        # img_para = self.trans_conv(fused_features)


        #print(f"[After trans_conv2] img_para shape: {img_para.shape}")
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)  # 四维变二维  # [280,128]
        #print(f"[After trans_conv_global] img_para shape: {img_para.shape}")

        ############################################ audio encoder############################################################
        # 原audio
        audio_para = self.audio_encoder(audio_feature)
        #print(f"[After audio_encoder] audio_para shape: {audio_para.shape}")
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        #print(f"[After audio_encoder_global] audio_para shape: {audio_para.shape}")

        # 纯transfomer
        # 确保音频输入形状正确
        # 原输入形状：[B, 29, T] (T为时间步长)
        # Transformer需要 [B, T, 29] → 转置维度
        # audio_para = self.audio_encoder(audio_feature.permute(0, 2, 1))  # [B, T,29]→[B,128]

        # CNN+Transfomre
        #audio_para = self.audio_encoder(audio_feature)  # 直接输出[B,128]

        ########################################### concat alignment feature and audio feature###################################
        # 原
        #trans_para = torch.cat([img_para, audio_para], 1)   # [280,256]
        #print(f"[After concat] trans_para shape: {trans_para.shape}")

        # 替换原来的拼接操作为交叉注意力融合
        img_para = img_para  # [B, 128]
        audio_para = audio_para  # [B, 128]

        # 转换为序列格式 [B, 1, 128]
        img_seq = img_para.unsqueeze(1)
        audio_seq = audio_para.unsqueeze(1)

        # 图像关注音频
        attended_img, _ = self.cross_attention_img2audio(query=img_seq,key=audio_seq,value=audio_seq)
        attended_img = attended_img.squeeze(1)  # [B, 128]

        # 音频关注图像
        attended_audio, _ = self.cross_attention_audio2img(query=audio_seq, key=img_seq,value=img_seq)
        attended_audio = attended_audio.squeeze(1)  # [B, 128]

        # 拼接融合结果
        trans_para = torch.cat([attended_img, attended_audio], dim=1)  # [B, 256]

        ############################### use AdaAT do spatial deformation on reference feature maps#############################
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)   # feature_map, para_code
        #ref_trans_feature = self.adaAT(ref_trans_feature, trans_para,ref_in_feature)
        # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)

        ################################################### feature decoder#####################################################
        # 原
        # print(f"[After AdaAT] source_in_feature shape: {source_in_feature.shape}")
        # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)


        #merge_feature = self.spatial_gate(source_in_feature, ref_trans_feature)

        out = self.out_conv(merge_feature)

        return out





