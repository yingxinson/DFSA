import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
#import AdaIN
from models.FreqFusion import FreqFusion
from models.AdaIN import calc_mean_std
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
        super(DINet, self).__init__()
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
        self.freq_fusion = FreqFusion(256)

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
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        #self.out_conv = DINetDecoder()
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
        alpha = 0.1
        # print(f"[Before source_in_conv]")
        source_in_feature = self.source_in_conv(source_img)  # [280,256,26,20]
        #print("source_in_feature", source_in_feature.shape)
        source_depth_feature = self.depth_source_conv(depth_source)  # [280,256,26,20]
        #print("source_depth_feature", source_depth_feature.shape)
        #source_in_feature = source_in_feature + alpha * source_depth_feature  # 加权融合而不是拼接
        #source_in_feature = self.fusion_source(source_in_feature, source_depth_feature)  # 动态注意力融合
        source_in_feature = self.freq_fusion(source_in_feature, source_depth_feature)   # [280,256,26,20]
        #print(" source_in_feature_fusion",  source_in_feature.shape)
        #source_in_feature = self.fusion_source[1](source_in_feature, source_depth_feature)

        # print(f"[After source_in_conv] shape: {source_in_feature.shape}")
        ###########################################reference image encoder######################################################
        # print(f"[Before ref_in_conv] ")
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_depth_feature = self.depth_ref_conv(depth_ref)
        # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
        #ref_in_feature = ref_in_feature + alpha * ref_depth_feature  # 加权融合而不是拼接
        #ref_in_feature = self.fusion_ref(ref_in_feature, ref_depth_feature)  # 动态注意力融合
        ref_in_feature = self.freq_fusion(ref_in_feature, ref_depth_feature)   # [280,256,26,20]
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
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)

        #merge_feature = self.spatial_gate(source_in_feature, ref_trans_feature)

        out = self.out_conv(merge_feature)

        return out