import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
import AdaIN
from models.DeepFusion import DeepFusion
import os


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

        self.style_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(1),  # [B, C]
            nn.Linear(channels, channels * 2),
        )

    def calc_mean_std(self, feat):

        B, C, H, W = feat.size()
        feat_var = feat.view(B, C, -1).var(dim=2, unbiased=False) + self.eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content, style):

        style_stats = self.style_fc(style)  # [B, 2C]
        gamma, beta = style_stats.chunk(2, dim=1)  # [B, C] x2
        gamma = gamma.view(-1, content.size(1), 1, 1)
        beta = beta.view(-1, content.size(1), 1, 1)


        content_mean, content_std = self.calc_mean_std(content)
        normalized_content = (content - content_mean) / content_std


        return normalized_content * gamma + beta

class DFSA(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel, depth_encoder, depth_decoder):
        super(DFSA, self).__init__()

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

        self.adain_block = ParametricAdaIN(channels=256)

        self.trans_conv1 = nn.Sequential(

            SameBlock2d(256, 128, kernel_size=3, padding=1),

            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),

            DownBlock2d(128, 128, kernel_size=3, padding=1),


            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),


            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),


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


        self.cross_attention_img2audio = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.cross_attention_audio2img = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)


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


        batch_size, _, H, W = ref_img.shape
        ref_imgs = ref_img.view(batch_size, 5, 3, H, W)

        depth_features = []
        for i in range(5):
            feat_encoder = self.depth_encoder(ref_imgs[:, i])  # [B, C, H', W']
            feat_decoder = self.depth_decoder(feat_encoder)
            feat_features = feat_decoder[("disp", 0)]
            depth_features.append(feat_features)
        combined_feat = torch.cat(depth_features, dim=1)  # [B, 5C, H', W']
        depth_ref = combined_feat
        depth_ref = F.interpolate(depth_ref, size=(ref_img.shape[2], ref_img.shape[3]), mode="bilinear",align_corners=True)
        #print("depth_ref_inter",depth_ref.shape)


        # print("Before source depth_encoder")
        outputs = self.depth_decoder(self.depth_encoder(source_img))
        depth_source = outputs[("disp", 0)]
        depth_source = F.interpolate(depth_source, size=(source_img.shape[2], source_img.shape[3]), mode='bilinear',align_corners=True)
        #print("depth_ref_inter", depth_ref.shape)

        ################################################### source  image encoder  #################################################
        alpha = 1
        # print(f"[Before source_in_conv]")
        source_in_feature = self.source_in_conv(source_img)  # [280,256,26,20]
        #print("source_in_feature", source_in_feature.shape)
        source_depth_feature = self.depth_source_conv(depth_source)  # [280,256,26,20]
        #print("source_depth_feature", source_depth_feature.shape)
        #source_in_feature = source_in_feature + alpha * source_depth_feature
        ################################################### source image fusion#################################################
        source_in_feature = self.freq_fusion(source_in_feature, source_depth_feature)   # [280,256,26,20]
        #print(" source_in_feature_fusion",  source_in_feature.shape)
        # print(f"[After source_in_conv] shape: {source_in_feature.shape}")


        ###########################################reference image encoder######################################################
        # print(f"[Before ref_in_conv] ")
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_depth_feature = self.depth_ref_conv(depth_ref)
        # print(f"[After ref_in_conv] shape: {ref_in_feature.shape}")
        #ref_in_feature = ref_in_feature + alpha * ref_depth_feature
        ###########################################reference image fusion######################################################
        ref_in_feature = self.freq_fusion(ref_in_feature, ref_depth_feature)   # [280,256,26,20]
        #print(" ref_in_feature_fusion", ref_in_feature.shape)


        ######################################### audio#############################################################

        #combined = torch.cat([source_in_feature, ref_in_feature], 1)    #([270, 512, 26, 20])
        # #print(f"[After trans_conv1] combined shape: {combined.shape}")
        #img_para = self.trans_conv(combined)     # ([270, 128, 2, 2])
        # print(f"[After trans_conv1] img_para shape: {img_para.shape}")
        #

        # #combined = torch.cat([source_in_feature, ref_in_feature], 1)  # [B,512,H,W]
        # #print(f"[combined] combined shape: {combined.shape}")
        # img_para = self.adain_conv[0](combined)
        # print(f"[img_para] img_para shape: {img_para.shape}")
        # img_para = adaptive_instance_normalization(img_para, ref_in_feature)  #  # [280,256,26,20]
        # img_para = self.trans_conv1(img_para)   # [280,128,2,2]
        # print(f"[img_para_trans_conv] img_para shape: {img_para.shape}")


        img_para = self.adain_block(content=source_in_feature, style=ref_in_feature)
        img_para = self.trans_conv1(img_para)
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        #print(f"[After trans_conv_global] img_para shape: {img_para.shape}")



        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)



        img_para = img_para  # [B, 128]
        audio_para = audio_para  # [B, 128]

        #  [B, 1, 128]
        img_seq = img_para.unsqueeze(1)
        audio_seq = audio_para.unsqueeze(1)


        attended_img, _ = self.cross_attention_img2audio(query=img_seq,key=audio_seq,value=audio_seq)
        attended_img = attended_img.squeeze(1)  # [B, 128]


        attended_audio, _ = self.cross_attention_audio2img(query=audio_seq, key=img_seq,value=img_seq)
        attended_audio = attended_audio.squeeze(1)  # [B, 128]


        trans_para = torch.cat([attended_img, attended_audio], dim=1)  # [B, 256]

        ############################### use AdaAT do spatial deformation on reference feature maps#############################
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)   # feature_map, para_code
        #ref_trans_feature = self.adaAT(ref_trans_feature, trans_para,ref_in_feature)
        # print(f"[After AdaAT] ref_trans_feature shape: {ref_trans_feature.shape}")
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)


        ################################################### feature decoder#####################################################
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)

        #merge_feature = self.spatial_gate(source_in_feature, ref_trans_feature)

        out = self.out_conv(merge_feature)

        return out


