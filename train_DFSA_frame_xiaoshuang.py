from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.xiaoshuang import DFSA   # 这里的 DFSA 是你实现的“双流版 DFSA”

from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from torch.utils.data import DataLoader
from dataset.dataset_DFSA_frame import DFSADataset
from sync_batchnorm import convert_model
from config.config import DFSATrainingOptions

import random
import numpy as np
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from torch.cuda.amp import GradScaler, autocast  # AMP

if __name__ == "__main__":
    """
    frame training code of DFSA (双流版，建议用于 256 阶段)
    使用 coarse-to-fine 策略时：
      - 128 阶段：用原来的单流脚本训练，得到 netG_model_epoch_x.pth
      - 256 阶段：用本脚本，--coarse2fine --coarse_model_path=128 那个权重
    """
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取配置
    opt = DFSATrainingOptions().parse_args()

    # 随机数种子
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)

    # ===================== 1. 数据 =====================
    train_data = DFSADataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=72,
        pin_memory=True,
        drop_last=True
    )
    train_data_length = len(training_data_loader)

    # ===================== 2. 初始化网络（双流 G） =====================
    # 这里我们默认这个脚本是用在 256 阶段的，所以直接用双流 + d_model=128, nhead=4
    print(">>> Init dual-stream DFSA, mouth_region_size =", opt.mouth_region_size)
    net_g = DFSA(
        opt.source_channel,
        opt.ref_channel,
        opt.audio_channel,
        d_model=128,

    )

    net_dI = Discriminator(opt.source_channel,
                           opt.D_block_expansion,
                           opt.D_num_blocks,
                           opt.D_max_features)
    net_vgg = Vgg19()

    # ===================== 3. coarse2fine：从 128 单流加载权重 =====================
    if opt.coarse2fine:
        print('loading checkpoint for coarse2fine training: {}'.format(opt.coarse_model_path))
        checkpoint = torch.load(opt.coarse_model_path, map_location=device)

        # 兼容两种保存方式：直接 state_dict 或 {'state_dict': {'net_g': ...}}
        if 'state_dict' in checkpoint and 'net_g' in checkpoint['state_dict']:
            old_sd = checkpoint['state_dict']['net_g']
        else:
            old_sd = checkpoint

        # 去掉 DataParallel 的 "module." 前缀
        old_sd = {k.replace("module.", ""): v for k, v in old_sd.items()}

        new_sd = {}
        for k, v in old_sd.items():
            # 1) 单流里的 img_fuse_conv.* → 复制到 img_fuse_conv_global & img_fuse_conv_mouth
            if k.startswith("img_fuse_conv."):
                k_global = k.replace("img_fuse_conv", "img_fuse_conv_global")
                k_mouth  = k.replace("img_fuse_conv", "img_fuse_conv_mouth")
                new_sd[k_global] = v
                new_sd[k_mouth]  = v

            # 2) 单流里的 cross_blocks.* → 复制到 cross_blocks_global & cross_blocks_mouth
            elif k.startswith("cross_blocks."):
                k_global = k.replace("cross_blocks", "cross_blocks_global")
                k_mouth  = k.replace("cross_blocks", "cross_blocks_mouth")
                new_sd[k_global] = v
                new_sd[k_mouth]  = v

            else:
                # 3) 其他层（encoder / decoder / audio_encoder / adaAT / film 等）名字相同，直接拷贝
                new_sd[k] = v

        # 用 strict=False 部分加载：嘴部流新增 & trans_fuse 等保持随机初始化
        missing, unexpected = net_g.load_state_dict(new_sd, strict=False)
        print(">>> coarse2fine weight loaded with strict=False")
        print("    missing keys   :", len(missing))
        print("    unexpected keys:", len(unexpected))
    else:
        print(">>> No coarse2fine, training from scratch.")

    # ===================== 4. 多卡 & SyncBatchNorm =====================
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net_g = nn.DataParallel(net_g).to(device)
        net_g = convert_model(net_g).to(device)
        net_dI = nn.DataParallel(net_dI).to(device)
        net_vgg = nn.DataParallel(net_vgg).to(device)
    else:
        net_g = net_g.to(device)
        net_g = convert_model(net_g).to(device)
        net_dI = net_dI.to(device)
        net_vgg = net_vgg.to(device)

    # ===================== 5. 优化器 & AMP =====================
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)

    scaler_g = GradScaler()
    scaler_dI = GradScaler()

    # ===================== 6. 损失 & 学习率策略 =====================
    criterionGAN = GANLoss().to(device)
    criterionL1  = nn.L1Loss().to(device)

    net_g_scheduler  = get_scheduler(optimizer_g,  opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)

    # ===================== 7. 训练循环 =====================
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        net_dI.train()
        net_vgg.eval()  # VGG 一般 eval 就行

        for iteration, data in enumerate(training_data_loader):
            # 取数据
            source_image_data, source_image_mask, reference_clip_data, deepspeech_feature = data
            source_image_data   = source_image_data.float().to(device)
            source_image_mask   = source_image_mask.float().to(device)
            reference_clip_data = reference_clip_data.float().to(device)
            deepspeech_feature  = deepspeech_feature.float().to(device)

            # =========================
            # (1) Update D network
            # =========================
            optimizer_dI.zero_grad()

            # D 阶段：用 no_grad 跑一次 G，仅供 D 判别
            with torch.no_grad():
                with autocast():
                    fake_out_for_d = net_g(source_image_mask,
                                           reference_clip_data,
                                           deepspeech_feature)

            with autocast():
                _, pred_fake_dI = net_dI(fake_out_for_d.detach())
                loss_dI_fake = criterionGAN(pred_fake_dI, False)

                _, pred_real_dI = net_dI(source_image_data)
                loss_dI_real = criterionGAN(pred_real_dI, True)

                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5

            scaler_dI.scale(loss_dI).backward()
            scaler_dI.step(optimizer_dI)
            scaler_dI.update()

            # =========================
            # (2) Update G network
            # =========================
            optimizer_g.zero_grad()

            with autocast():
                fake_out = net_g(source_image_mask,
                                 reference_clip_data,
                                 deepspeech_feature)

                # down sample output image and real image
                fake_out_half      = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
                target_tensor_half = F.interpolate(source_image_data,
                                                   scale_factor=0.5,
                                                   mode='bilinear')

                # gan dI loss (for G)
                _, pred_fake_dI = net_dI(fake_out)
                loss_g_dI = criterionGAN(pred_fake_dI, True)

                # VGG perception loss
                perception_real       = net_vgg(source_image_data)
                perception_fake       = net_vgg(fake_out)
                perception_real_half  = net_vgg(target_tensor_half)
                perception_fake_half  = net_vgg(fake_out_half)

                loss_g_perception = 0.0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i],       perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i],  perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception

                # combine perception loss and gan loss
                loss_g = loss_g_perception + loss_g_dI

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            print(
                "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} "
                "Loss_perception: {:.4f} lr_g = {:.7f} ".format(
                    epoch, iteration, len(training_data_loader),
                    float(loss_dI.item()),
                    float(loss_g_dI.item()),
                    float(loss_g_perception.item()),
                    optimizer_g.param_groups[0]['lr']
                )
            )

        # 更新学习率
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)

        # ===================== 8. 保存 checkpoint =====================
        if True:  # 每个 epoch 都存
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path,
                                          'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {
                    'net_g': net_g.state_dict(),
                    'net_dI': net_dI.state_dict()
                },
                'optimizer': {
                    'net_g': optimizer_g.state_dict(),
                    'net_dI': optimizer_dI.state_dict()
                }
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to epoch {}".format(epoch))
