import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d

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
    yy = y.view(1,-1, 1).repeat(d,1, w)
    xx = x.view(1,1, -1).repeat(d,h, 1)
    zz = z.view(-1,1,1).repeat(1,h,w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed,zz

class ResBlock1d(nn.Module):
    '''
        basic block
    '''
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features,out_features,1)
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
    def __init__(self, in_features,out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                               padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features,out_features,1)
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
                              padding=padding,stride=2)
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class DownBlock2d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out

class SameBlock1d(nn.Module):
    '''
            basic block
    '''
    def __init__(self, in_features, out_features,  kernel_size, padding):
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
    def __init__(self, in_features, out_features,  kernel_size=3, padding=1):
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
    '''
       AdaAT operator
    '''
    def __init__(self,  para_ch,feature_ch):
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

    def forward(self, feature_map,para_code):
        batch,d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159#
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

        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
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
    def __init__(self, source_channel,ref_channel,audio_channel):
        super(DFSA, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel,64,kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128,256,kernel_size=3, padding=1)
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        # self.trans_conv = nn.Sequential(
        #     # 20 →10
        #     SameBlock2d(512, 128, kernel_size=3, padding=1),
        #     SameBlock2d(128, 128, kernel_size=11, padding=5),
        #     SameBlock2d(128, 128, kernel_size=11, padding=5),
        #     DownBlock2d(128, 128, kernel_size=3, padding=1),
        #     # 10 →5
        #     SameBlock2d(128, 128, kernel_size=7, padding=3),
        #     SameBlock2d(128, 128, kernel_size=7, padding=3),
        #     DownBlock2d(128, 128, kernel_size=3, padding=1),
        #     # 5 →3
        #     SameBlock2d(128, 128, kernel_size=3, padding=1),
        #     DownBlock2d(128, 128, kernel_size=3, padding=1),
        #     # 3 →2
        #     SameBlock2d(128, 128, kernel_size=3, padding=1),
        #     DownBlock2d(128, 128, kernel_size=3, padding=1),
        #
        # )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )

        # 外观特征卷积列表网格
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
        self.adaAT = AdaAT(256, 256)

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

        # 添加交叉注意力模块
        self.cross_attention_img2audio = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.cross_attention_audio2img = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)

        # 将组合的特征转化为视频
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

    def forward(self, source_img,ref_img,audio_feature):
        ## source image encoder
        source_in_feature = self.source_in_conv(source_img)

        ## reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img)

        ################################################################################### alignment encoder
        #原
        #img_para = self.trans_conv(torch.cat([source_in_feature,ref_in_feature],1))


        img_para = self.adain_block(content=source_in_feature, style=ref_in_feature)
        # print(f"[img_para] img_para shape: {img_para.shape}")
        img_para = self.trans_conv1(img_para)

        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)

        ## audio encoder
        audio_para = self.audio_encoder(audio_feature)
        # 通过全局自适应池化 (global_avg1d) 和 squeeze，将音频特征缩小为固定大小
        audio_para = self.global_avg1d(audio_para).squeeze(2)

        #################################################################### concat alignment feature and audio feature
        # 原
        #trans_para = torch.cat([img_para,audio_para],1)     #(1,256)


        #print("trans_para",trans_para.shape)
        # # 替换原来的拼接操作为交叉注意力融合
        img_para = img_para  # [B, 128]
        audio_para = audio_para  # [B, 128]

        # 转换为序列格式 [B, 1, 128]
        img_seq = img_para.unsqueeze(1)
        audio_seq = audio_para.unsqueeze(1)

        # 图像关注音频
        attended_img, _ = self.cross_attention_img2audio(query=img_seq, key=audio_seq, value=audio_seq)
        attended_img = attended_img.squeeze(1)  # [B, 128]

        # 音频关注图像
        attended_audio, _ = self.cross_attention_audio2img(query=audio_seq, key=img_seq, value=img_seq)
        attended_audio = attended_audio.squeeze(1)  # [B, 128]

        # 拼接融合结果
        trans_para = torch.cat([attended_img, attended_audio], dim=1)  # [B, 256]

        #################################################################### use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)  # (1,256,104,80)
        #print("ref_trans_feature",ref_trans_feature.shape)

        ## feature decoder
        merge_feature = torch.cat([source_in_feature,ref_trans_feature],1)   # (1,512,104,80)
        #print("merge_feature",merge_feature.shape)


        out = self.out_conv(merge_feature)
        return out



