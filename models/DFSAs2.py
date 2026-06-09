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
    img_tokens:  [B, N, C]   图像 token 序列 (N = H*W)
    audio_tokens:[B, T, C]   音频 token 序列 (T = 时间步)
    做一层: 图像 self-attn + 图像对音频的 cross-attn + FFN
    """
    def __init__(self, d_model=256, nhead=8, dim_ff=1024):
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
        # img_tokens: [B,N,C], audio_tokens: [B,T,C]

        # 1) 图像内部 self-attention
        x = img_tokens
        x_sa, _ = self.self_attn(x, x, x)               # [B,N,C]
        x = self.norm1(x + x_sa)

        # 2) 图像 query，音频 key/value 的 cross-attention
        x_ca, _ = self.cross_attn(x, audio_tokens, audio_tokens)
        x = self.norm2(x + x_ca)

        # 3) FFN
        x_ffn = self.ffn(x)
        x = self.norm3(x + x_ffn)

        return x


class FiLM2d(nn.Module):
    """
    简单的 FiLM: 用 cond 向量生成每个通道的 gamma / beta
    x:    [B, C, H, W]
    cond: [B, cond_dim]
    """
    def __init__(self, cond_dim, num_features):
        super(FiLM2d, self).__init__()
        self.fc = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        gamma, beta = self.fc(cond).chunk(2, dim=1)     # [B,C], [B,C]
        gamma = gamma.view(-1, x.size(1), 1, 1)         # [B,C,1,1]
        beta  = beta.view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta

class DFSA(nn.Module):
    """
    双流版 DFSA：
    - Global stream：全图特征 + audio 做 cross-attention
    - Mouth stream：嘴部区域特征 + audio 做另一条 cross-attention
    - 两条流得到的条件向量再与 audio 融合，作为 trans_para 去控制 AdaAT + FiLM
    """
    def __init__(self, source_channel, ref_channel, audio_channel,
                 d_model=128, nhead=4, num_layers=1, dim_ff=1024,
                 mouth_region_size=256):
        super(DFSA, self).__init__()

        # ====== mouth 相关参数，和 DFSADataset 保持一致 ======
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size // 2        # 和 Dataset 里一样
        self.radius_1_4 = self.radius // 4          # 和 Dataset 里一样
        # encoder：SameBlock(1) + Down(2) + Down(2) => 空间下采样因子 4
        self.down_factor = 4

        # ========== 1. 图像 encoder：与单流版保持一致 ==========
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

        # ========== 2. audio encoder：与单流版保持一致 ==========
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )
        # 把音频通道映射到 d_model，方便与图像 tokens 对齐
        self.audio_proj = nn.Conv1d(128, d_model, kernel_size=1)

        self.d_model = d_model

        # ========== 3. 图像融合 conv：全局流 + 嘴部流 各一套 ==========
        # 全局流：用整张 feature map
        self.img_fuse_conv_global = nn.Conv2d(256 * 2, d_model, kernel_size=1)
        # 嘴部流：只看 mouth patch 的 feature
        self.img_fuse_conv_mouth = nn.Conv2d(256 * 2, d_model, kernel_size=1)

        # ========== 4. 双流 Cross-Modal Transformer ==========
        self.cross_blocks_global = nn.ModuleList([
            CrossModalBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff)
            for _ in range(num_layers)
        ])
        self.cross_blocks_mouth = nn.ModuleList([
            CrossModalBlock(d_model=d_model, nhead=nhead, dim_ff=dim_ff)
            for _ in range(num_layers)
        ])

        # ========== 5. 外观卷积 + AdaAT：与单流版一样 ==========
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

        # 这里 para_ch = d_model
        self.adaAT = AdaAT(para_ch=d_model, feature_ch=256)

        # ========== 6. Decoder + FiLM：与单流版一致 ==========
        self.dec_conv1 = SameBlock2d(512, 128, kernel_size=3, padding=1)
        self.dec_up1   = UpBlock2d(128, 128, kernel_size=3, padding=1)
        self.dec_res   = ResBlock2d(128, 128, 3, 1)
        self.dec_up2   = UpBlock2d(128, 128, kernel_size=3, padding=1)
        self.dec_out   = nn.Conv2d(128, 3, kernel_size=7, padding=3)

        self.film1 = FiLM2d(cond_dim=d_model, num_features=128)
        self.film2 = FiLM2d(cond_dim=d_model, num_features=128)

        # ========== 7. 全局流 + 嘴部流 条件融合的小 MLP ==========
        # [img_global_para; img_mouth_para]  -> d_model
        self.trans_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

    # ====== 精确版嘴部裁剪：用 Dataset 同样的 mouth 区域（缩到特征图坐标） ======
    def _crop_mouth_feat(self, feat):
        """
        feat: [B, C, H', W']
        Dataset 中嘴部在原图坐标是:
            y ∈ [radius, radius + mouth_region_size)
            x ∈ [radius_1_4, radius_1_4 + mouth_region_size)
        encoder 下采样因子为 down_factor (=4)，
        所以在特征图坐标系中相应除以 down_factor。
        """
        B, C, H, W = feat.shape

        y0 = self.radius // self.down_factor
        x0 = self.radius_1_4 // self.down_factor
        h  = self.mouth_region_size // self.down_factor
        w  = self.mouth_region_size // self.down_factor

        y1 = y0 + h
        x1 = x0 + w

        # 防止越界，做一下 clamp
        y0 = max(0, min(y0, H - 1))
        x0 = max(0, min(x0, W - 1))
        y1 = max(y0 + 1, min(y1, H))
        x1 = max(x0 + 1, min(x1, W))

        return feat[:, :, y0:y1, x0:x1]

    def forward(self, source_img, ref_img, audio_feature):
        """
        source_img:   [B, source_channel, H, W]
        ref_img:      [B, ref_channel,   H, W]
        audio_feature:[B, audio_channel, T]
        """

        # ============ 1. 图像编码 ============
        source_in_feature = self.source_in_conv(source_img)   # [B,256,H',W']
        ref_in_feature    = self.ref_in_conv(ref_img)         # [B,256,H',W']

        # ============ 2. 音频编码 → token ============
        audio_feat   = self.audio_encoder(audio_feature)      # [B,128,T']
        audio_feat   = self.audio_proj(audio_feat)            # [B,d_model,T']
        audio_tokens = audio_feat.permute(0, 2, 1)            # [B,T',d_model]

        # ============ 3. 全局流：全图 cross-attention ============
        img_fused_global = torch.cat([source_in_feature, ref_in_feature], dim=1)   # [B,512,H',W']
        img_feat_global  = self.img_fuse_conv_global(img_fused_global)            # [B,d_model,H',W']

        # 再下采样一倍，减少 token 数，防止 OOM
        img_feat_global  = F.avg_pool2d(img_feat_global, kernel_size=2, stride=2)  # [B,d_model,Hg,Wg]

        B, Cg, Hg, Wg = img_feat_global.shape
        img_tokens_global = img_feat_global.view(B, Cg, Hg * Wg).permute(0, 2, 1)  # [B,Ng,d_model]

        for blk in self.cross_blocks_global:
            img_tokens_global = blk(img_tokens_global, audio_tokens)               # [B,Ng,d_model]

        img_para_global = img_tokens_global.mean(dim=1)                            # [B,d_model]

        # ============ 4. 嘴部流：mouth patch cross-attention ============
        source_mouth = self._crop_mouth_feat(source_in_feature)                    # [B,256,Hm,Wm]
        ref_mouth    = self._crop_mouth_feat(ref_in_feature)                       # [B,256,Hm,Wm]
        img_fused_mouth = torch.cat([source_mouth, ref_mouth], dim=1)              # [B,512,Hm,Wm]
        img_feat_mouth  = self.img_fuse_conv_mouth(img_fused_mouth)                # [B,d_model,Hm,Wm]

        img_feat_mouth  = F.avg_pool2d(img_feat_mouth, kernel_size=2, stride=2)    # [B,d_model,Hm',Wm']

        B, Cm, Hm, Wm = img_feat_mouth.shape
        img_tokens_mouth = img_feat_mouth.view(B, Cm, Hm * Wm).permute(0, 2, 1)    # [B,Nm,d_model]

        for blk in self.cross_blocks_mouth:
            img_tokens_mouth = blk(img_tokens_mouth, audio_tokens)                 # [B,Nm,d_model]

        img_para_mouth = img_tokens_mouth.mean(dim=1)                              # [B,d_model]

        # ============ 5. 全局 + 嘴部 + 音频 条件融合 ============
        audio_para = audio_tokens.mean(dim=1)                                      # [B,d_model]

        gm_cat  = torch.cat([img_para_global, img_para_mouth], dim=1)             # [B,2*d_model]
        gm_para = self.trans_fuse(gm_cat)                                         # [B,d_model]

        trans_para = 0.5 * (gm_para + audio_para)                                 # [B,d_model]

        # ============ 6. AdaAT：用 trans_para 形变参考特征 ============
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)          # [B,256,H',W']
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)             # [B,256,H',W']
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)

        # ============ 7. Decoder + FiLM 调制 ============
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], dim=1)  # [B,512,H',W']

        x = self.dec_conv1(merge_feature)                                         # [B,128,H',W']
        x = self.film1(x, trans_para)

        x = self.dec_up1(x)                                                       # [B,128,2H',2W']
        x = self.dec_res(x)
        x = self.film2(x, trans_para)

        x = self.dec_up2(x)                                                       # [B,128,4H',4W']
        out = torch.sigmoid(self.dec_out(x))                                      # [B,3,4H',4W']

        return out





