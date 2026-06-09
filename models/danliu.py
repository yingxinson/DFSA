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
class CrossModalBlock(nn.Module):
    """
    img_tokens:  [B, N, C]
    audio_tokens:[B, T, C]
    """
    def __init__(self, d_model=128, nhead=4, dim_ff=512):
        super(CrossModalBlock, self).__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, img_tokens, audio_tokens):
        x = img_tokens
        x_sa, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x_sa)

        x_ca, _ = self.cross_attn(x, audio_tokens, audio_tokens)
        x = self.norm2(x + x_ca)

        x_ffn = self.ffn(x)
        x = self.norm3(x + x_ffn)
        return x


class FiLM2d(nn.Module):
    """
    x:    [B, C, H, W]
    cond: [B, cond_dim]
    """
    def __init__(self, cond_dim, num_features):
        super(FiLM2d, self).__init__()
        self.fc = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        gamma, beta = self.fc(cond).chunk(2, dim=1)  # [B,C], [B,C]
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta  = beta.view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta


class DFSA(nn.Module):
    """
    单流 + Cross-Attn + 多尺度 FiLM（d_model=128版）
    """
    def __init__(self, source_channel, ref_channel, audio_channel,
                 d_model=128, nhead=4, num_layers=1, dim_ff=512):
        super(DFSA, self).__init__()
        self.d_model = d_model

        # ========= 1) Image encoders =========
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )

        # ========= 2) Audio encoder =========
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )
        # 关键：映射到 d_model=128
        self.audio_proj = nn.Conv1d(128, d_model, kernel_size=1)

        # ========= 3) Image fuse -> d_model =========
        # 关键：输出通道是 d_model=128
        self.img_fuse_conv = nn.Conv2d(256 * 2, d_model, kernel_size=1)

        # ========= 4) Cross-Modal Blocks =========
        self.cross_blocks = nn.ModuleList([
            CrossModalBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff)
            for _ in range(num_layers)
        ])

        # ========= 5) Appearance conv + AdaAT =========
        appearance_conv_list = []
        for _ in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)

        # 关键：para_ch=d_model=128
        self.adaAT = AdaAT(para_ch=d_model, feature_ch=256)

        # ========= 6) Decoder =========
        self.dec_conv1 = SameBlock2d(512, 128, kernel_size=3, padding=1)
        self.dec_up1   = UpBlock2d(128, 128, kernel_size=3, padding=1)
        self.dec_res   = ResBlock2d(128, 128, 3, 1)
        self.dec_up2   = UpBlock2d(128, 128, kernel_size=3, padding=1)
        self.dec_res_high = ResBlock2d(128, 128, 3, 1)
        self.dec_out   = nn.Conv2d(128, 3, kernel_size=7, padding=3)

        # ========= 7) Multi-scale FiLM =========
        self.num_scales = 3
        # 关键：输入 d_model=128，输出 3*d_model=384
        self.cond_proj = nn.Linear(d_model, d_model * self.num_scales)

        self.film_low  = FiLM2d(cond_dim=d_model, num_features=128)
        self.film_mid  = FiLM2d(cond_dim=d_model, num_features=128)
        self.film_high = FiLM2d(cond_dim=d_model, num_features=128)

    def forward(self, source_img, ref_img, audio_feature):
        # ===== 1) image encode =====
        source_in_feature = self.source_in_conv(source_img)  # [B,256,H',W']
        ref_in_feature    = self.ref_in_conv(ref_img)        # [B,256,H',W']

        # ===== 2) img fuse -> d_model tokens =====
        img_fused = torch.cat([source_in_feature, ref_in_feature], dim=1)  # [B,512,H',W']
        img_feat  = self.img_fuse_conv(img_fused)                          # [B,d_model,H',W']
        B, C, H, W = img_feat.shape
        img_tokens = img_feat.view(B, C, H * W).permute(0, 2, 1)           # [B,N,d_model]

        # ===== 3) audio encode -> d_model tokens =====
        audio_feat   = self.audio_encoder(audio_feature)                   # [B,128,T']
        audio_feat   = self.audio_proj(audio_feat)                         # [B,d_model,T']
        audio_tokens = audio_feat.permute(0, 2, 1)                         # [B,T',d_model]

        # ===== 4) cross-modal alignment =====
        for blk in self.cross_blocks:
            img_tokens = blk(img_tokens, audio_tokens)                     # [B,N,d_model]

        img_para   = img_tokens.mean(dim=1)                                # [B,d_model]
        audio_para = audio_tokens.mean(dim=1)                              # [B,d_model]
        trans_para = 0.5 * (img_para + audio_para)                         # [B,d_model]  ✅关键

        # ===== 5) AdaAT warp on ref feature =====
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)   # [B,256,H',W']
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)      # ✅ para_ch=d_model
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)

        # ===== 6) decode + multi-scale FiLM =====
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], dim=1)  # [B,512,H',W']

        cond_all = self.cond_proj(trans_para)                               # [B,3*d_model]=[B,384]
        cond_low, cond_mid, cond_high = cond_all.chunk(3, dim=1)            # each [B,128]

        x = self.dec_conv1(merge_feature)
        x = self.film_low(x, cond_low)

        x = self.dec_up1(x)
        x = self.dec_res(x)
        x = self.film_mid(x, cond_mid)

        x = self.dec_up2(x)
        x = self.dec_res_high(x)
        x = self.film_high(x, cond_high)

        out = torch.sigmoid(self.dec_out(x))
        return out


