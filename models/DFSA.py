import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
#import AdaIN
from models.DeepFusion import DeepFusion
import os

#from models.AdaIN import calc_mean_std,adaptive_instance_normalization
import torchvision


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

class ParametricAdaIN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(ParametricAdaIN, self).__init__()
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

class DINet(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
        super(DINet, self).__init__()

        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder

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
        # )



        self.adain_block = ParametricAdaIN(channels=256)  # 假设特征通道为256

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

        # 添加交叉注意力模块
        self.cross_attention_img2audio = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.cross_attention_audio2img = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # 添加FreqFusion模块
        self.freq_fusion = DeepFusion(256)

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

        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),     # 512
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )

        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
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

        ################################################### source  image encoder  #################################################
        alpha = 1
        # print(f"[Before source_in_conv]")
        source_in_feature = self.source_in_conv(source_img)  # [280,256,26,20]
        #print("source_in_feature", source_in_feature.shape)
        source_depth_feature = self.depth_source_conv(depth_source)  # [280,256,26,20]
        #print("source_depth_feature", source_depth_feature.shape)
        #source_in_feature = source_in_feature + alpha * source_depth_feature  # 加权融合而不是拼接
        ################################################### source image fusion#################################################
        source_in_feature = self.freq_fusion(source_in_feature, source_depth_feature)   # [280,256,26,20]
        #print(" source_in_feature_fusion",  source_in_feature.shape)
        # print(f"[After source_in_conv] shape: {source_in_feature.shape}")


        ###########################################reference image encoder######################################################
        # print(f"[Before ref_in_conv] ")
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_depth_feature = self.depth_ref_conv(depth_ref)
        # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
        #ref_in_feature = ref_in_feature + alpha * ref_depth_feature  # 加权融合而不是拼接
        ###########################################reference image fusion######################################################
        ref_in_feature = self.freq_fusion(ref_in_feature, ref_depth_feature)   # [280,256,26,20]
        #print(" ref_in_feature_fusion", ref_in_feature.shape)

        ######################################################可视化################################################
        import torchvision.utils as vutils

        save_dir = "./debug_vis/freq_fusion/"
        os.makedirs(save_dir, exist_ok=True)
        # ====== 添加可视化输出 ======
        # 假设只保存第一张（batch=0）的部分通道
        if not self.training:  # 或者加条件，比如每隔若干step保存
            with torch.no_grad():
                # [B, C, H, W]
                feat = source_in_feature[0].detach().cpu()  # 取第一个样本
                # 选取前3个通道可视化成RGB
                feat_rgb = feat[:3, :, :]  # [3,H,W]
                # 归一化到[0,1]
                feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-5)
                # 保存图片
                vutils.save_image(feat_rgb, os.path.join(save_dir, f"source_freqfusion.png"))

                feat2 = ref_in_feature[0].detach().cpu()
                feat2_rgb = feat2[:3, :, :]
                feat2_rgb = (feat2_rgb - feat2_rgb.min()) / (feat2_rgb.max() - feat2_rgb.min() + 1e-5)
                vutils.save_image(feat2_rgb, os.path.join(save_dir, f"ref_freqfusion.png"))

        ######################################### 深度语义映射模块#############################################################
        #原
        #combined = torch.cat([source_in_feature, ref_in_feature], 1)    #([270, 512, 26, 20])
        # #print(f"[After trans_conv1] combined shape: {combined.shape}")
        #img_para = self.trans_conv(combined)     # ([270, 128, 2, 2])
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


        #空间级注入
        img_para = self.adain_block(content=source_in_feature, style=ref_in_feature)
        #print(f"[img_para] img_para shape: {img_para.shape}")
        #深层空间编码
        img_para = self.trans_conv1(img_para)
        #print(f"[img_para_trans_conv] img_para shape: {img_para.shape}")
        #空间→向量
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)  # 四维变二维  # [280,128]
        #print(f"[After trans_conv_global] img_para shape: {img_para.shape}")


        # 原audio
        audio_para = self.audio_encoder(audio_feature)
        #print(f"[After audio_encoder] audio_para shape: {audio_para.shape}")
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        #print(f"[After audio_encoder_global] audio_para shape: {audio_para.shape}")


        # 原
        #trans_para = torch.cat([img_para, audio_para], 1)   # [280,256]
        #print(f"[After concat] trans_para shape: {trans_para.shape}")

        #向量级对齐
        # # 替换原来的拼接操作为交叉注意力融合
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
    # def forward(self, source_img, ref_img, audio_feature, visualize=True, save_dir="D:/Python/team/DINet2/eval/tu", batch_idx=0):
    #     """
    #     visualize: 是否可视化并保存
    #     save_dir: 保存目录
    #     batch_idx: 选哪个 batch 的样本进行可视化
    #     """
    #     import os
    #     import matplotlib.pyplot as plt
    #
    #
    #
    #     if visualize:
    #         os.makedirs(save_dir, exist_ok=True)
    #
    #     def to_np(x):
    #         return x.detach().cpu().numpy()
    #
    #     def norm01(x):
    #         x = x - x.min()
    #         if x.max() > 0:
    #             x = x / (x.max() + 1e-8)
    #         return x
    #
    #     def save_rgb(tensor, name):
    #         # tensor: [B,3,H,W]
    #         img = to_np(tensor[batch_idx])
    #         img = img.transpose(1, 2, 0)  # H,W,3
    #         img = norm01(img)
    #         path = os.path.join(save_dir, f"{name}.png")
    #         plt.imsave(path, img)
    #         if visualize:
    #             plt.figure();
    #             plt.imshow(img);
    #             plt.title(name);
    #             plt.axis("off");
    #             plt.show()
    #
    #     def save_depth(tensor, name, upsample_to=None):
    #         # tensor: [B,1,H,W] or [B,H,W] etc.
    #         arr = to_np(tensor)
    #         # handle shapes
    #         if arr.ndim == 4:
    #             d = arr[batch_idx, 0]
    #         elif arr.ndim == 3:
    #             # [B, C, H, W] with C>1? or [B, H, W]
    #             if arr.shape[1] == 1:
    #                 d = arr[batch_idx, 0]
    #             else:
    #                 # if channels >1 treat first channel
    #                 d = arr[batch_idx, 0]
    #         elif arr.ndim == 2:
    #             d = arr
    #         else:
    #             raise ValueError("Unsupported depth tensor shape: " + str(arr.shape))
    #
    #         # optional upsample to specific size using cv2
    #         if upsample_to is not None:
    #             d = cv2.resize(d, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_LINEAR)
    #
    #         d = norm01(d)
    #         depth_uint8 = (d * 255).astype("uint8")
    #         path = os.path.join(save_dir, f"{name}.png")
    #         cv2.imwrite(path, depth_uint8)
    #         if visualize:
    #             plt.figure();
    #             plt.imshow(d, cmap="gray");
    #             plt.title(name);
    #             plt.axis("off");
    #             plt.show()
    #
    #     def save_feature(tensor, name, channel_reduce="mean", upsample_to=None):
    #         # tensor: [B, C, H, W] -> convert to single HxW via mean or select channel
    #         arr = to_np(tensor)
    #         B, C, H, W = arr.shape
    #         if channel_reduce == "mean":
    #             feat = arr[batch_idx].mean(0)
    #         elif isinstance(channel_reduce, int):
    #             c = channel_reduce if channel_reduce < C else 0
    #             feat = arr[batch_idx, c]
    #         else:
    #             feat = arr[batch_idx].mean(0)
    #
    #         if upsample_to is not None:
    #             feat = cv2.resize(feat, (upsample_to[1], upsample_to[0]), interpolation=cv2.INTER_LINEAR)
    #
    #         feat = norm01(feat)
    #         path = os.path.join(save_dir, f"{name}.png")
    #         plt.imsave(path, feat, cmap="viridis")
    #         if visualize:
    #             plt.figure();
    #             plt.imshow(feat, cmap="viridis");
    #             plt.title(name);
    #             plt.axis("off");
    #             plt.show()
    #
    #     ############################ main forward (你的原始实现，只在需要位置插入可视化) ############################
    #     source_img = source_img.float()
    #     ref_img = ref_img.float()
    #
    #     # 调整 ref (假设 [B,15,H,W] -> 5 frames of 3 channels)
    #     batch_size, _, H, W = ref_img.shape
    #     ref_imgs = ref_img.view(batch_size, 5, 3, H, W)  # [B,5,3,H,W]
    #
    #     # 如果需要可视化：5 帧原始参考图像（RGB）
    #     if visualize:
    #         for i in range(5):
    #             save_rgb(ref_imgs[:, i], name=f"ref_frame_{i + 1}")
    #
    #     # depth image encoder: 对每帧预测 depth (disp)
    #     depth_features = []
    #     for i in range(5):
    #         feat_encoder = self.depth_encoder(ref_imgs[:, i])  # encoder output
    #         feat_decoder = self.depth_decoder(feat_encoder)
    #         feat_features = feat_decoder[("disp", 0)]  # [B,1,h',w']
    #         depth_features.append(feat_features)
    #
    #         # 可视化每一帧的单通道视差（上采样到原始 ref 大小以便观察）
    #         if visualize:
    #             save_depth(feat_features, name=f"ref_depth_raw_{i + 1}", upsample_to=(H, W))
    #
    #     # 拼接成 [B, 5, h', w']（你的 depth_ref_conv 接受5通道）
    #     combined_feat = torch.cat(depth_features, dim=1)  # [B,5,H',W']
    #     depth_ref = combined_feat  # 保留五通道深度堆叠
    #
    #     # 可视化：五通道堆叠的第 3 帧（示例），以及把五帧平均作为 fused depth（可选）
    #     if visualize:
    #         # 第3帧
    #         single3 = combined_feat[:, 2:3, :, :]  # [B,1,h',w']
    #         save_depth(single3, name="ref_depth_frame3", upsample_to=(H, W))
    #         # 平均融合并展示（fused 参考深度）
    #         fused_ref = torch.mean(combined_feat, dim=1, keepdim=True)  # [B,1,h',w']
    #         save_depth(fused_ref, name="ref_depth_fused", upsample_to=(H, W))
    #
    #     # source depth
    #     outputs = self.depth_decoder(self.depth_encoder(source_img))
    #     depth_source = outputs[("disp", 0)]  # [B,1,h_s,w_s]
    #
    #     # 可视化 depth_source（上采样到 source_img 尺寸以便观察）
    #     if visualize:
    #         save_depth(depth_source, name="depth_source_raw", upsample_to=(source_img.shape[2], source_img.shape[3]))
    #
    #     # 将 depth_source 上采样到 source 原始分辨率（你的原代码）
    #     depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]),
    #                                  mode='bilinear', align_corners=True)
    #
    #     # source image encoder
    #     source_in_feature = self.source_in_conv(source_img)  # [B,256, h_f, w_f]
    #
    #     # 可视化：source_in_feature（通道平均热图）
    #     if visualize:
    #         save_feature(source_in_feature, name="source_in_feature_before_fusion",
    #                      upsample_to=(source_img.shape[2], source_img.shape[3]))
    #
    #     # 将深度经 conv 转换为网络特征
    #     source_depth_feature = self.depth_source_conv(depth_source)  # [B,256, h_f, w_f]
    #
    #     # 可视化：source_depth_feature（通道平均）
    #     if visualize:
    #         save_feature(source_depth_feature, name="source_depth_feature",
    #                      upsample_to=(source_img.shape[2], source_img.shape[3]))
    #
    #     # source fusion (FreqFusion)
    #     source_in_feature = self.freq_fusion(source_in_feature, source_depth_feature)  # [B,256,h_f,w_f]
    #
    #     # 可视化：source_in_feature 融合后
    #     if visualize:
    #         save_feature(source_in_feature, name="source_in_feature_after_freqfusion",
    #                      upsample_to=(source_img.shape[2], source_img.shape[3]))
    #
    #     # reference image encoder (注意 ref_img 这里是原始 [B,15,H,W]，你的 ref_in_conv 设计为接收 ref_channel=15)
    #     ref_in_feature = self.ref_in_conv(ref_img)  # [B,256,h_f,w_f]
    #
    #     # 可视化：ref_in_feature（通道平均）
    #     if visualize:
    #         save_feature(ref_in_feature, name="ref_in_feature_before_fusion",
    #                      upsample_to=(ref_img.shape[2], ref_img.shape[3]))
    #
    #     # ref_depth_feature: 注意 depth_ref 是 [B,5,h',w']，depth_ref_conv 期望 in_channels=5
    #     ref_depth_feature = self.depth_ref_conv(depth_ref)  # [B,256,h_f,w_f]
    #
    #     # 可视化：ref_depth_feature（通道平均）
    #     if visualize:
    #         save_feature(ref_depth_feature, name="ref_depth_feature", upsample_to=(ref_img.shape[2], ref_img.shape[3]))
    #
    #     # reference fusion
    #     ref_in_feature = self.freq_fusion(ref_in_feature, ref_depth_feature)  # [B,256,h_f,w_f]
    #
    #     # 可视化：ref_in_feature 融合后
    #     if visualize:
    #         save_feature(ref_in_feature, name="ref_in_feature_after_freqfusion",
    #                      upsample_to=(ref_img.shape[2], ref_img.shape[3]))
    #
    #     # 下面保持你现有的 alignment / adain / audio / adaAT 流程
    #     img_para = self.adain_block(content=source_in_feature, style=ref_in_feature)
    #     img_para = self.trans_conv1(img_para)
    #     img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
    #
    #     audio_para = self.audio_encoder(audio_feature)
    #     audio_para = self.global_avg1d(audio_para).squeeze(2)
    #
    #     # cross attention fusion (和你原来的)
    #     img_seq = img_para.unsqueeze(1)
    #     audio_seq = audio_para.unsqueeze(1)
    #     attended_img, _ = self.cross_attention_img2audio(query=img_seq, key=audio_seq, value=audio_seq)
    #     attended_img = attended_img.squeeze(1)
    #     attended_audio, _ = self.cross_attention_audio2img(query=audio_seq, key=img_seq, value=img_seq)
    #     attended_audio = attended_audio.squeeze(1)
    #     trans_para = torch.cat([attended_img, attended_audio], dim=1)  # [B,256]
    #
    #     # AdaAT 变形
    #     ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
    #     ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
    #     ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)  # <-- 你指定要可视化的这个
    #
    #     # 可视化：AdaAT 后的 ref_trans_feature（通道平均）
    #     if visualize:
    #         save_feature(ref_trans_feature, name="ref_trans_feature_after_appearance_conv1",
    #                      upsample_to=(ref_img.shape[2], ref_img.shape[3]))
    #
    #     # decoder / out
    #     merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
    #     out = self.out_conv(merge_feature)
    #
    #     return out

# import depth
# depth_encoder = depth.ResnetEncoder(18, False)
# depth_encoder1 = depth.ResnetEncoder(18, False,5)
# depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
# loaded_dict_enc = torch.load("../asserts/encoder.pth")
# loaded_dict_dec = torch.load("../asserts/depth.pth")
# filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
# depth_encoder.load_state_dict(filtered_dict_enc)
# depth_decoder.load_state_dict(loaded_dict_dec)
# depth_encoder = depth_encoder.eval()
# depth_encoder1 = depth_encoder1.eval()
# depth_decoder = depth_decoder.eval()
#
# # 假设输入通道的配置（具体值需根据你的任务定义）
# source_channel = 3    # 输入图像的通道数（如RGB为3）
# ref_channel = 15        # 参考图像的通道数
# audio_channel = 29    # 音频特征的通道数
#
#
# # 实例化模型
# model = DINet(
#     source_channel=source_channel,
#     ref_channel=ref_channel,
#     audio_channel=audio_channel,
#     depth_encoder=depth_encoder,
#     depth_decoder=depth_decoder
# )
#
# # 打印模型结构
# print(model)

