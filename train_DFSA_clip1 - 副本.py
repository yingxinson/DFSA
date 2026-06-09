from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DFSAdepth import DFSA
from models.Syncnet import SyncNetPerception
from utils.training_utils import get_scheduler, update_learning_rate, GANLoss
from config.config import DFSATrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DFSA_clip import DFSADataset

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from itertools import chain  # 添加缺失的导入
from torch.cuda.amp import GradScaler, autocast  # === AMP: 新增 ===

if __name__ == "__main__":
    '''
        clip training code of DFSA
        in the resolution you want, using clip training code after frame training
    '''
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load config
    opt = DFSATrainingOptions().parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)

    # load training data
    train_data = DFSADataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=min(72, os.cpu_count()),
        pin_memory=True,
        drop_last=True
    )
    train_data_length = len(training_data_loader)

    # init network
    net_g = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel)
    net_dI = Discriminator(opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features)
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features)
    net_vgg = Vgg19()
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).to(device)

    # parallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net_g = nn.DataParallel(net_g).to(device)
        net_g = convert_model(net_g).to(device)
        net_dI = nn.DataParallel(net_dI).to(device)
        net_dV = nn.DataParallel(net_dV).to(device)
        net_vgg = nn.DataParallel(net_vgg).to(device)
    else:
        net_g = net_g.to(device)
        net_g = convert_model(net_g).to(device)
        net_dI = net_dI.to(device)
        net_dV = net_dV.to(device)
        net_vgg = net_vgg.to(device)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.Adam(net_dV.parameters(), lr=opt.lr_dI)

    # === AMP: 初始化 GradScaler ===
    scaler_g = GradScaler()
    scaler_dI = GradScaler()
    scaler_dV = GradScaler()

    # load frame trained DFSA weight
    print('loading frame trained DFSA weight from: {}'.format(opt.pretrained_frame_DFSA_path))
    checkpoint = torch.load(opt.pretrained_frame_DFSA_path, map_location=device)
    net_g.load_state_dict(checkpoint['state_dict']['net_g'])

    # set criterion
    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)

    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).to(device)

    # 一般感知网络 / syncnet 不更新，用 eval（如果你想完全冻结也可以把 requires_grad=False）
    net_vgg.eval()
    net_lipsync.eval()

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        net_dI.train()
        net_dV.train()

        for iteration, data in enumerate(training_data_loader):
            # forward: 先把 clip 展成 frame 级别
            source_clip, source_clip_mask, reference_clip, deep_speech_clip, deep_speech_full = data
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().to(device)
            source_clip_mask = torch.cat(torch.split(source_clip_mask, 1, dim=1), 0).squeeze(1).float().to(device)
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().to(device)
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().to(device)
            deep_speech_full = deep_speech_full.float().to(device)

            # =========================
            # (1) 用 no_grad 跑一次 G 给所有 D 用
            # =========================
            with torch.no_grad():
                with autocast():
                    fake_out_for_d = net_g(source_clip_mask, reference_clip, deep_speech_clip)

            # =========================
            # (2) Update D_I
            # =========================
            optimizer_dI.zero_grad()
            with autocast():
                _, pred_fake_dI = net_dI(fake_out_for_d.detach())
                loss_dI_fake = criterionGAN(pred_fake_dI, False)

                _, pred_real_dI = net_dI(source_clip)
                loss_dI_real = criterionGAN(pred_real_dI, True)

                loss_dI = (loss_dI_fake + loss_dI_real) * 0.5

            scaler_dI.scale(loss_dI).backward()
            scaler_dI.step(optimizer_dI)
            scaler_dI.update()

            # =========================
            # (3) Update D_V
            # =========================
            optimizer_dV.zero_grad()

            # 按 batch 重新拼回 clip 形式： (B*T, C, H, W) -> (B, T*C, H, W)
            with autocast():
                condition_fake_dV_for_d = torch.cat(torch.split(fake_out_for_d, opt.batch_size, dim=0), 1)
                _, pred_fake_dV = net_dV(condition_fake_dV_for_d.detach())
                loss_dV_fake = criterionGAN(pred_fake_dV, False)

                condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
                _, pred_real_dV = net_dV(condition_real_dV)
                loss_dV_real = criterionGAN(pred_real_dV, True)

                loss_dV = (loss_dV_fake + loss_dV_real) * 0.5

            scaler_dV.scale(loss_dV).backward()
            scaler_dV.step(optimizer_dV)
            scaler_dV.update()

            # =========================
            # (4) Update G (DFSA)
            # =========================
            optimizer_g.zero_grad()
            with autocast():
                # 重新前向一次 G（这次保留计算图，用于所有 G 的 loss）
                fake_out = net_g(source_clip_mask, reference_clip, deep_speech_clip)

                fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
                source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')

                # D_I 给 G 的 GAN loss
                _, pred_fake_dI = net_dI(fake_out)
                loss_g_dI = criterionGAN(pred_fake_dI, True)

                # D_V 给 G 的 GAN loss
                condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
                _, pred_fake_dV = net_dV(condition_fake_dV)
                loss_g_dV = criterionGAN(pred_fake_dV, True)

                # VGG perceptual loss
                perception_real = net_vgg(source_clip)
                perception_fake = net_vgg(fake_out)
                perception_real_half = net_vgg(source_clip_half)
                perception_fake_half = net_vgg(fake_out_half)

                loss_g_perception = 0
                for i in range(len(perception_real)):
                    loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                    loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
                loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception

                # SyncNet perception loss
                fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
                fake_out_clip_mouth = fake_out_clip[:, :,
                                      train_data.radius:train_data.radius + train_data.mouth_region_size,
                                      train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
                sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
                loss_sync = criterionMSE(sync_score, real_tensor.expand_as(sync_score)) * opt.lamb_syncnet_perception

                # combine all losses
                loss_g = loss_g_perception + loss_g_dI + loss_g_dV + loss_sync

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            print(
                "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} "
                "Loss_DV: {:.4f} Loss_GV: {:.4f} "
                "Loss_perception: {:.4f} Loss_sync: {:.4f} lr_g = {:.7f} ".format(
                    epoch, iteration, len(training_data_loader),
                    float(loss_dI.item()), float(loss_g_dI.item()),
                    float(loss_dV.item()), float(loss_g_dV.item()),
                    float(loss_g_perception.item()), float(loss_sync.item()),
                    optimizer_g.param_groups[0]['lr'])
            )

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)

        # checkpoint
        if epoch % opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {
                    'net_g': net_g.state_dict(),
                    'net_dI': net_dI.state_dict(),
                    'net_dV': net_dV.state_dict()
                },
                'optimizer': {
                    'net_g': optimizer_g.state_dict(),
                    'net_dI': optimizer_dI.state_dict(),
                    'net_dV': optimizer_dV.state_dict()
                }
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
