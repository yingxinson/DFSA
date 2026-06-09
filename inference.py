'''
from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DFSAInferenceOptions
from models.DFSA import DFSA

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
from collections import OrderedDict

def validate_input_size(H, W):
    min_size = 32
    if not (isinstance(H, int) and isinstance(W, int)):
        raise TypeError(f"尺寸必须为整数，实际类型 H:{type(H)}, W:{type(W)}")
    if H < min_size or W < min_size:
        raise ValueError(f"输入尺寸必须 ≥ {min_size}x{min_size}, 当前: {H}x{W}")
def extract_frames_from_video(video_path,save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width),int(frame_height))

if __name__ == '__main__':
    # load config
    opt = DFSAInferenceOptions().parse_args()
    if not os.path.exists(opt.source_video_path):
        raise ('wrong video path : {}'.format(opt.source_video_path))
    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path,video_frame_dir)
    ############################################## extract deep speech feature ##############################################
    print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
    if not os.path.exists(opt.deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    if not os.path.exists(opt.driving_audio_path):
        raise ('wrong audio path :{}'.format(opt.driving_audio_path))
    ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    ############################################## load facial landmark ##############################################
    print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    if not os.path.exists(opt.source_openface_landmark_path):
        raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
    #video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(np.int)
    video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(int)
    ############################################## align frame with driving audio ##############################################
    print('aligning frames with driving audio')
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
    if len(video_frame_path_list) != video_landmark_data.shape[0]:
        raise ('video frames are misaligned with detected landmarks')
    video_frame_path_list.sort()
    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
    if video_frame_path_list_cycle_length >= res_frame_length:
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                    + res_video_frame_path_list \
                                    + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    ############################################## randomly select 5 reference images ##############################################
    print('selecting five reference images')
    ref_img_list = []
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    
    
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    for ref_index in ref_index_list:
        crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
        ref_img_crop = ref_img[
                  ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                  ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
                  :]
        ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
        #ref_img_crop = ref_img_crop / 255.0
        ref_img_crop = ref_img_crop.astype(np.float32) / 255.0
        ref_img_list.append(ref_img_crop)
    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

    ############################################## load pretrained model weight ##############################################
    print('loading pretrained model from: {}'.format(opt.pretrained_clip_DFSA_path))
    model = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DFSA_path):
        raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DFSA_path))
    state_dict = torch.load(opt.pretrained_clip_DFSA_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    ############################################## inference frame by frame ##############################################
    if not os.path.exists(opt.res_video_dir):
        os.mkdir(opt.res_video_dir)
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    if os.path.exists(res_video_path):
        os.remove(res_video_path)
    res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
    if os.path.exists(res_face_path):
        os.remove(res_face_path)
    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (resize_w, resize_h))
    for clip_end_index in range(5, pad_length, 1):
        print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
        crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.05)
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
        crop_frame_data = frame_data[
                            frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                            frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
                            :]
        crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
        #crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
        #crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h)).astype(np.float32)
        #crop_frame_data = crop_frame_data / 255.0
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h)).astype(np.float32)
        crop_frame_data = np.clip(crop_frame_data / 255.0, 0, 1)  # 显式截断
        crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                        opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0
        validate_input_size(crop_frame_data.shape[0], crop_frame_data.shape[1])
                              
        print(f"type: {type(crop_frame_data)}")          # 应输出 <class 'numpy.ndarray'>
        print(f"dtype: {crop_frame_data.dtype}")         # 检查是否为 float32
        print(f"shape: {crop_frame_data.shape}")        # 确认形状是否符合模型输入要求
        
	
        #crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        #crop_frame_tensor = torch.from_numpy(crop_frame_data.astype(np.float64)).float().cuda().permute(2, 0, 1).unsqueeze(0)  # float64
        # 修改后（强制 float32）
        #crop_frame_tensor = (torch.from_numpy(crop_frame_data.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda())
        #crop_frame_tensor = (torch.from_numpy(crop_frame_data).permute(2, 0, 1).unsqueeze(0).cuda())
        crop_frame_tensor = torch.from_numpy(crop_frame_data).to(dtype=torch.float32, device="cuda").permute(2, 0, 1).unsqueeze(0)
        print(f"dtype: {crop_frame_data.dtype}") 
        print(f"张量形状验证: {crop_frame_tensor.shape}") 
    
    
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3,:,:]
        videowriter.write(frame_data[:, :, ::-1])
    videowriter.release()
    videowriter_face.release()
    video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')
    if os.path.exists(video_add_audio_path):
        os.remove(video_add_audio_path)
    cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
        res_video_path,
        opt.driving_audio_path,
        video_add_audio_path)
    subprocess.call(cmd, shell=True)


from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface, compute_crop_radius
from config.config import DFSAInferenceOptions
from models.DFSA import DFSA

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
from collections import OrderedDict

def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        if not ret:
            break  # !!! 处理视频读取中断
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width), int(frame_height))

if __name__ == '__main__':
    # ================================= 环境验证 =================================
    assert torch.__version__ >= "1.8.0", "Require PyTorch >= 1.8"
    assert np.__version__ >= "1.19.0", "Require NumPy >= 1.19"
    
    # ================================= 加载配置 =================================
    opt = DFSAInferenceOptions().parse_args()
    if not os.path.exists(opt.source_video_path):
        raise ValueError('wrong video path : {}'.format(opt.source_video_path))  # !!! 使用标准异常类型

    # ================================= 视频帧提取 =================================
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    os.makedirs(video_frame_dir, exist_ok=True)  # !!! 替换原有创建方式
    video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir)

    # ================================= 音频特征提取 =================================
    print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
    if not os.path.exists(opt.deepspeech_model_path):
        raise FileNotFoundError('pls download pretrained model of deepspeech')  # !!!
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')

    # ================================= 面部关键点加载 =================================
    print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(int)

    # ================================= 帧与音频对齐 =================================
    video_frame_path_list = sorted(glob.glob(os.path.join(video_frame_dir, '*.jpg')))  # !!! 直接排序
    if len(video_frame_path_list) != video_landmark_data.shape[0]:
        raise ValueError('video frames are misaligned with detected landmarks')

    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    
    # ... [保持原有对齐逻辑不变] ...
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
    if video_frame_path_list_cycle_length >= res_frame_length:
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                    + res_video_frame_path_list \
                                    + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    # ================================= 参考图像选择 =================================
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    
    # !!! 确保 resize_w 和 resize_h 为奇数
    resize_w = resize_w if resize_w % 2 == 1 else resize_w + 1
    resize_h = resize_h if resize_h % 2 == 1 else resize_h + 1
    
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    ref_img_list = []
    
    for ref_index in ref_index_list:
        crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
            raise RuntimeError('Large facial size change detected')

        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]  # BGR to RGB
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]

        # !!! 添加边界保护
        y_start = max(0, ref_landmark[29, 1] - crop_radius)
        y_end = min(ref_img.shape[0], ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4)
        x_start = max(0, ref_landmark[33, 0] - crop_radius - crop_radius_1_4)
        x_end = min(ref_img.shape[1], ref_landmark[33, 0] + crop_radius + crop_radius_1_4)
        
        ref_img_crop = ref_img[y_start:y_end, x_start:x_end, :]
        ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h)).astype(np.float32) / 255.0  # !!! 显式指定类型
        ref_img_list.append(ref_img_crop)

    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).to(
        dtype=torch.float32, device="cuda"  # !!! 显式指定设备和类型
    ).permute(2, 0, 1).unsqueeze(0)

    # ================================= 模型加载 =================================
    model = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    state_dict = torch.load(opt.pretrained_clip_DFSA_path)['state_dict']['net_g']
    new_state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())  # !!! 简化写法
    model.load_state_dict(new_state_dict)
    model.eval()

    # ================================= 推理循环 =================================
    os.makedirs(opt.res_video_dir, exist_ok=True)
    res_video_path = os.path.join(opt.res_video_dir, os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    
    # !!! 使用更可靠的视频编码器
    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    videowriter_face = cv2.VideoWriter(
        res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), 25, (resize_w, resize_h)
    )

    for clip_end_index in range(5, pad_length, 1):
        print(f'synthesizing {clip_end_index - 5}/{pad_length - 5} frame')
        
        # !!! 数据预处理增强
        crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :], random_scale=1.05)
        if not crop_flag:
            raise RuntimeError('Large facial size change detected during inference')

        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]  # BGR to RGB
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]

        # !!! 边界保护 + 尺寸验证
        y_start = max(0, frame_landmark[29, 1] - crop_radius)
        y_end = min(frame_data.shape[0], frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4)
        x_start = max(0, frame_landmark[33, 0] - crop_radius - crop_radius_1_4)
        x_end = min(frame_data.shape[1], frame_landmark[33, 0] + crop_radius + crop_radius_1_4)
        
        crop_frame_data = frame_data[y_start:y_end, x_start:x_end, :]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h)).astype(np.float32) / 255.0
        np.clip(crop_frame_data, 0.0, 1.0, out=crop_frame_data)  # !!! 显式截断
        
        # !!! 输入形状断言
        assert crop_frame_data.shape == (resize_h, resize_w, 3), \
            f"Bad input shape: {crop_frame_data.shape}, expected ({resize_h}, {resize_w}, 3)"

        # !!! 张量转换优化
        crop_frame_tensor = torch.as_tensor(crop_frame_data, device="cuda", dtype=torch.float32)
        crop_frame_tensor = crop_frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # !!! 模型前向传播保护
        try:
            with torch.no_grad():
                pre_frame = model(crop_frame_tensor, ref_img_tensor, 
                                torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :])
                                .permute(1, 0).unsqueeze(0).float().cuda())
        except RuntimeError as e:
            print(f"Error at frame {clip_end_index}: {str(e)}")
            break

        # !!! 后处理优化
        pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        pre_frame = np.clip(pre_frame * 255, 0, 255).astype(np.uint8)  # 显式截断
        pre_frame_bgr = cv2.cvtColor(pre_frame, cv2.COLOR_RGB2BGR)  # 正确转换颜色空间
        
        videowriter_face.write(pre_frame_bgr)
        
        # 后续合成逻辑保持不变...
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3,:,:]
        videowriter.write(frame_data[:, :, ::-1])
        
    # ================================= 收尾工作 =================================
    videowriter.release()
    videowriter_face.release()
    
    # 添加音频时使用更安全的subprocess调用
    cmd = [
        'ffmpeg', '-y',  # !!! 自动覆盖输出文件
        '-i', res_video_path,
        '-i', opt.driving_audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-map', '0:v:0',
        '-map', '1:a:0',
        res_video_path.replace('.mp4', '_add_audio.mp4')
    ]
    subprocess.run(cmd, check=True)  # !!! 使用更安全的run方法

'''
from utils.deep_speech import DeepSpeech
from utils.data_processing import load_landmark_openface,compute_crop_radius
from config.config import DFSAInferenceOptions
from models.DFSA import DFSA

import numpy as np
import glob
import os
import cv2
import torch
import subprocess
import random
from collections import OrderedDict
import depth
import dlib
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 日志
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 修复 OpenMP 冲突

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("D:/Python/team/DFSA2/asserts/shape_predictor_68_face_landmarks.dat")

def extract_frames_from_video(video_path,save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)
    return (int(frame_width),int(frame_height))

def convert_audio_to_wav(audio_path):
    output_path = os.path.splitext(audio_path)[0] + '.wav'
    if not audio_path.lower().endswith('.wav'):
        command = f'ffmpeg -i "{audio_path}" -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}"'
        subprocess.run(command, shell=True, check=True)
    return output_path

def load_landmark_dlib(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError("No faces found in the image.")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

if __name__ == '__main__':
    # load config
    opt = DFSAInferenceOptions().parse_args()

    opt.driving_audio_path = convert_audio_to_wav(opt.driving_audio_path)  # 1111111111111111111111111111  opt.driving_audio_path
    if not os.path.exists(opt.source_video_path):
        #raise ('wrong video path : {}'.format(opt.source_video_path))
        raise FileNotFoundError(f"Wrong video path: {opt.source_video_path}")

    ############################################## extract frames from source video ##############################################
    print('extracting frames from video: {}'.format(opt.source_video_path))
    video_frame_dir = opt.source_video_path.replace('.mp4', '')
    if not os.path.exists(video_frame_dir):
        os.mkdir(video_frame_dir)
    video_size = extract_frames_from_video(opt.source_video_path,video_frame_dir)

    ############################################## extract deep speech feature ##############################################
    print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
    if not os.path.exists(opt.deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    if not os.path.exists(opt.driving_audio_path):
        raise ('wrong audio path :{}'.format(opt.driving_audio_path))
    ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
    ############################################## load facial landmark ##############################################
    # print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
    # if not os.path.exists(opt.source_openface_landmark_path):
    #     raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
    # video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(int)  #11111111111111111111111111111111111111111111

    print('Tracking Face')
    video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
    video_frame_path_list.sort()
    video_landmark_data = np.array([load_landmark_dlib(frame) for frame in video_frame_path_list])

    ############################################## align frame with driving audio ##############################################
    print('aligning frames with driving audio')
    # video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
    # if len(video_frame_path_list) != video_landmark_data.shape[0]:
    #     raise ('video frames are misaligned with detected landmarks')
    video_frame_path_list.sort()
    video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
    if video_frame_path_list_cycle_length >= res_frame_length:
        res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frame_path_list_cycle_length
        remainder = res_frame_length % video_frame_path_list_cycle_length
        res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
        res_video_landmark_data = np.concatenate([video_landmark_data_cycle]* divisor + [video_landmark_data_cycle[:remainder, :, :]],0)
    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
                                    + res_video_frame_path_list \
                                    + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    ############################################## randomly select 5 reference images ##############################################
    print('selecting five reference images')
    ref_img_list = []
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    for ref_index in ref_index_list:
        crop_flag,crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index- 3])[:, :, ::-1]
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
        ref_img_crop = ref_img[
                  ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                  ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[33, 0] + crop_radius +crop_radius_1_4,
                  :]
        ref_img_crop = cv2.resize(ref_img_crop,(resize_w,resize_h))
        ref_img_crop = ref_img_crop / 255.0
        ref_img_list.append(ref_img_crop)
    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

    ############################################## randomly select 1 reference images ##############################################
    # # 修改点1：选择1个参考帧索引（原为5个）
    # ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 1)  # 数量参数从5改为1
    # # 修改点2：加载单个参考帧（移除循环）
    # ref_img_list = []
    # resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    # resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    # for ref_index in ref_index_list:  # 现在只循环1次
    #     crop_flag, crop_radius = compute_crop_radius(video_size,
    #                                                  res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
    #     if not crop_flag:
    #         raise ValueError('Our method cannot handle videos with large facial size changes')
    #
    #     crop_radius_1_4 = crop_radius // 4
    #     ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]  # BGR转RGB
    #     ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
    #
    #     # 裁剪参考帧
    #     ref_img_crop = ref_img[
    #                    ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
    #                    ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[
    #                                                                            33, 0] + crop_radius + crop_radius_1_4,
    #                    :
    #                    ]
    #     ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h)) / 255.0
    #     ref_img_list.append(ref_img_crop)
    # # 修改点3：直接使用单帧，无需拼接通道维度
    # ref_video_frame = ref_img_list[0]  # 取第一个（也是唯一一个）参考帧
    # ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()  # 形状 [1, 3, H, W]

    ############################################## load pretrained model weight ##############################################
    # 深度网络初始化（设备管理优化）
    # depth_encoder = depth.ResnetEncoder(18, False)
    # depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
    # loaded_dict_enc = torch.load(opt.depth_encoder_model)
    # loaded_dict_dec = torch.load(opt.depth_model)
    # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    # depth_encoder.load_state_dict(filtered_dict_enc)
    # depth_decoder.load_state_dict(loaded_dict_dec)
    # depth_encoder = depth_encoder.eval()
    # depth_decoder = depth_decoder.eval()

    print('loading pretrained model from: {}'.format(opt.pretrained_clip_DFSA_path))
    model = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DFSA_path):
        raise ValueError('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DFSA_path))
        #raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DFSA_path))
    state_dict = torch.load(opt.pretrained_clip_DFSA_path)['state_dict']['net_g']
    new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove module.
    #     new_state_dict[name] = v
    def fix_key(key):
        key = key.replace("module.","")
        return key

    new_state_dict = {fix_key(k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    ############################################## inference frame by frame ##############################################
    if not os.path.exists(opt.res_video_dir):
        os.mkdir(opt.res_video_dir)
    res_video_path = os.path.join(opt.res_video_dir,os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4')
    if os.path.exists(res_video_path):
        os.remove(res_video_path)
    res_face_path = res_video_path.replace('_facial_dubbing.mp4', '_synthetic_face.mp4')
    if os.path.exists(res_face_path):
        os.remove(res_face_path)
    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, video_size)
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (resize_w, resize_h))
    for clip_end_index in range(5, pad_length, 1):
        print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
        crop_flag, crop_radius = compute_crop_radius(video_size,res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],random_scale = 1.05)
        if not crop_flag:
            raise ('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
        crop_frame_data = frame_data[
                            frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                            frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[33, 0] + crop_radius +crop_radius_1_4,
                            :]
        crop_frame_h,crop_frame_w = crop_frame_data.shape[0],crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w,resize_h))  # [32:224, 32:224, :]
        crop_frame_data = crop_frame_data / 255.0
        crop_frame_data[opt.mouth_region_size//2:opt.mouth_region_size//2 + opt.mouth_region_size,
                        opt.mouth_region_size//8:opt.mouth_region_size//8 + opt.mouth_region_size, :] = 0

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w,crop_frame_h))
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3,:,:]
        videowriter.write(frame_data[:, :, ::-1])
    videowriter.release()
    videowriter_face.release()
    video_add_audio_path = res_video_path.replace('.mp4', '_add_audio.mp4')
    if os.path.exists(video_add_audio_path):
        os.remove(video_add_audio_path)
    cmd = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}'.format(
        res_video_path,
        opt.driving_audio_path,
        video_add_audio_path)
    subprocess.call(cmd, shell=True)


# from utils.deep_speech import DeepSpeech
# from utils.data_processing import load_landmark_openface, compute_crop_radius
# from config.config import DFSAInferenceOptions
# from models.DFSAdepth1 import DFSA
#
# import numpy as np
# import glob
# import os
# import cv2
# import torch
# import subprocess
# import random
# from collections import OrderedDict
# import depth
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 日志
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 修复 OpenMP 冲突
#
#
# def get_versioned_filename(filepath):
#     base, ext = os.path.splitext(filepath)
#     counter = 1
#     while os.path.exists(filepath):
#         filepath = f"{base}({counter}){ext}"
#         counter += 1
#     return filepath
#
#
# def extract_frames_from_video(video_path, save_dir):
#     videoCapture = cv2.VideoCapture(video_path)
#     fps = videoCapture.get(cv2.CAP_PROP_FPS)
#     if int(fps) != 25:
#         print('warning: the input video is not 25 fps, it would be better to trans it to 25 fps!')
#     frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
#     frame_height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     frame_width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
#     for i in range(int(frames)):
#         ret, frame = videoCapture.read()
#         result_path = os.path.join(save_dir, str(i).zfill(6) + '.jpg')
#         cv2.imwrite(result_path, frame)
#     return (int(frame_width), int(frame_height))
#
#
# if __name__ == '__main__':
#     # load config
#     opt = DFSAInferenceOptions().parse_args()
#     if not os.path.exists(opt.source_video_path):
#         raise ('wrong video path : {}'.format(opt.source_video_path))
#
#     ############################################## extract frames from source video ##############################################
#     print('extracting frames from video: {}'.format(opt.source_video_path))
#     video_frame_dir = opt.source_video_path.replace('.mp4', '')
#     if not os.path.exists(video_frame_dir):
#         os.mkdir(video_frame_dir)
#     video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir)
#
#     ############################################## extract deep speech feature ##############################################
#     print('extracting deepspeech feature from : {}'.format(opt.driving_audio_path))
#     if not os.path.exists(opt.deepspeech_model_path):
#         raise ('pls download pretrained model of deepspeech')
#     DSModel = DeepSpeech(opt.deepspeech_model_path)
#     if not os.path.exists(opt.driving_audio_path):
#         raise ('wrong audio path :{}'.format(opt.driving_audio_path))
#     ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
#     res_frame_length = ds_feature.shape[0]
#     ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
#
#     ############################################## load facial landmark ##############################################
#     print('loading facial landmarks from : {}'.format(opt.source_openface_landmark_path))
#     if not os.path.exists(opt.source_openface_landmark_path):
#         raise ('wrong facial landmark path :{}'.format(opt.source_openface_landmark_path))
#     video_landmark_data = load_landmark_openface(opt.source_openface_landmark_path).astype(
#         int)  # 11111111111111111111111111111111111111111111
#
#     ############################################## align frame with driving audio ##############################################
#     print('aligning frames with driving audio')
#     video_frame_path_list = glob.glob(os.path.join(video_frame_dir, '*.jpg'))
#     if len(video_frame_path_list) != video_landmark_data.shape[0]:
#         raise ('video frames are misaligned with detected landmarks')
#     video_frame_path_list.sort()
#     video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
#     video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
#     video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
#     if video_frame_path_list_cycle_length >= res_frame_length:
#         res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
#         res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
#     else:
#         divisor = res_frame_length // video_frame_path_list_cycle_length
#         remainder = res_frame_length % video_frame_path_list_cycle_length
#         res_video_frame_path_list = video_frame_path_list_cycle * divisor + video_frame_path_list_cycle[:remainder]
#         res_video_landmark_data = np.concatenate(
#             [video_landmark_data_cycle] * divisor + [video_landmark_data_cycle[:remainder, :, :]], 0)
#     res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 \
#                                     + res_video_frame_path_list \
#                                     + [video_frame_path_list_cycle[-1]] * 2
#     res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
#     assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
#     pad_length = ds_feature_padding.shape[0]
#
#     ############################################## randomly select 5 reference images ##############################################
#     print('selecting five reference images')
#     ref_img_list = []
#     resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
#     resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
#     ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
#     for ref_index in ref_index_list:
#         crop_flag, crop_radius = compute_crop_radius(video_size,
#                                                      res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
#         if not crop_flag:
#             raise ('our method can not handle videos with large change of facial size!!')
#         crop_radius_1_4 = crop_radius // 4
#         ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]
#         ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
#         ref_img_crop = ref_img[
#                        ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
#                        ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[
#                                                                                33, 0] + crop_radius + crop_radius_1_4,
#                        :]
#         ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
#         ref_img_crop = ref_img_crop / 255.0
#         ref_img_list.append(ref_img_crop)
#     ref_video_frame = np.concatenate(ref_img_list, 2)
#     ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
#
#     ############################################## randomly select 1 reference images ##############################################
#     # # 修改点1：选择1个参考帧索引（原为5个）
#     # ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 1)  # 数量参数从5改为1
#     # # 修改点2：加载单个参考帧（移除循环）
#     # ref_img_list = []
#     # resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
#     # resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
#     # for ref_index in ref_index_list:  # 现在只循环1次
#     #     crop_flag, crop_radius = compute_crop_radius(video_size,
#     #                                                  res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
#     #     if not crop_flag:
#     #         raise ValueError('Our method cannot handle videos with large facial size changes')
#     #
#     #     crop_radius_1_4 = crop_radius // 4
#     #     ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]  # BGR转RGB
#     #     ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
#     #
#     #     # 裁剪参考帧
#     #     ref_img_crop = ref_img[
#     #                    ref_landmark[29, 1] - crop_radius:ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
#     #                    ref_landmark[33, 0] - crop_radius - crop_radius_1_4:ref_landmark[
#     #                                                                            33, 0] + crop_radius + crop_radius_1_4,
#     #                    :
#     #                    ]
#     #     ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h)) / 255.0
#     #     ref_img_list.append(ref_img_crop)
#     # # 修改点3：直接使用单帧，无需拼接通道维度
#     # ref_video_frame = ref_img_list[0]  # 取第一个（也是唯一一个）参考帧
#     # ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()  # 形状 [1, 3, H, W]
#
#     ############################################## load pretrained model weight ##############################################
#     # 深度网络初始化（设备管理优化）
#     depth_encoder = depth.ResnetEncoder(18, False)
#     depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
#     loaded_dict_enc = torch.load(opt.depth_encoder_model)
#     loaded_dict_dec = torch.load(opt.depth_model)
#     filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
#     depth_encoder.load_state_dict(filtered_dict_enc)
#     depth_decoder.load_state_dict(loaded_dict_dec)
#     depth_encoder = depth_encoder.eval()
#     depth_decoder = depth_decoder.eval()
#
#     print('loading pretrained model from: {}'.format(opt.pretrained_clip_DFSA_path))
#     model = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel, depth_encoder, depth_decoder).cuda()
#     if not os.path.exists(opt.pretrained_clip_DFSA_path):
#         raise ValueError('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DFSA_path))
#         # raise ('wrong path of pretrained model weight: {}'.format(opt.pretrained_clip_DFSA_path))
#     state_dict = torch.load(opt.pretrained_clip_DFSA_path)['state_dict']['net_g']
#     new_state_dict = OrderedDict()
#
#
#     # for k, v in state_dict.items():
#     #     name = k[7:]  # remove module.
#     #     new_state_dict[name] = v
#     def fix_key(key):
#         key = key.replace("module.", "")
#         return key
#
#
#     new_state_dict = {fix_key(k): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model.eval()
#
#     ############################################## inference frame by frame ##############################################
#     res_video_name = os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing.mp4'
#     res_video_path = os.path.join(opt.res_video_dir, res_video_name)
#     res_video_path = get_versioned_filename(res_video_path)  # Ensure unique filename
#
#     videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
#
#     if opt.auto_mask:
#         samelength_video_name = 'samelength.mp4'
#         samelength_video_path = os.path.join(opt.res_video_dir, samelength_video_name)
#         samelength_video_path = get_versioned_filename(samelength_video_path)  # Ensure unique filename
#         videowriter_samelength = cv2.VideoWriter(samelength_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
#
#     res_face_name = os.path.basename(opt.source_video_path)[:-4] + '_facial_dubbing_face.mp4'
#     res_face_path = os.path.join(opt.res_video_dir, res_face_name)
#     res_face_path = get_versioned_filename(res_face_path)  # Ensure unique filename
#
#     videowriter_face = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (resize_w, resize_h))
#
#     for clip_end_index in range(5, pad_length, 1):
#         print('synthesizing {}/{} frame'.format(clip_end_index - 5, pad_length - 5))
#         crop_flag, crop_radius = compute_crop_radius(video_size,
#                                                      res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :,
#                                                      :], random_scale=1.10)
#         if not crop_flag:
#             raise ('our method can not handle videos with large change of facial size!!')
#         crop_radius_1_4 = crop_radius // 4
#         frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]
#         frame_data_samelength = frame_data.copy()
#         if opt.auto_mask:
#             videowriter_samelength.write(frame_data_samelength[:, :, ::-1])
#
#         frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
#         crop_frame_data = frame_data[
#                           frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
#                           frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[
#                                                                                     33, 0] + crop_radius + crop_radius_1_4,
#                           :]
#         crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
#         crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h))  # [32:224, 32:224, :]
#         crop_frame_data = crop_frame_data / 255.0
#         crop_frame_data[opt.mouth_region_size // 2:opt.mouth_region_size // 2 + opt.mouth_region_size,
#         opt.mouth_region_size // 8:opt.mouth_region_size // 8 + opt.mouth_region_size, :] = 0
#
#         crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
#         deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1,0).unsqueeze(
#             0).float().cuda()
#
#         with torch.no_grad():
#             pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
#             pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
#         videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
#         pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
#         frame_data[
#         frame_landmark[29, 1] - crop_radius:
#         frame_landmark[29, 1] + crop_radius * 2,
#         frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
#         frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
#         :] = pre_frame_resize[:crop_radius * 3, :, :]
#         videowriter.write(frame_data[:, :, ::-1])
#     videowriter.release()
#     if opt.auto_mask:
#         videowriter_samelength.release()
#     videowriter_face.release()
#
#     if opt.auto_mask:
#         video_add_audio_path = os.path.join(opt.res_video_dir, 'pre_blend.mp4')
#     else:
#         video_add_audio_path = os.path.join(opt.res_video_dir, os.path.basename(opt.source_video_path)[:-4] + '_LIPSICK.mp4')
#
#     video_add_audio_path = get_versioned_filename(video_add_audio_path)  # Ensure unique filename
#
#     cmd = f'ffmpeg -r 25 -i "{res_video_path}" -i "{opt.driving_audio_path}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{video_add_audio_path}"'
#     subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress FFmpeg logs
#     os.remove(res_video_path)  # Clean up intermediate files
#     os.remove(res_face_path)  # Clean up intermediate files
#
#     if opt.auto_mask:
#         print('Auto Mask stage')
#         samelength_video_path = os.path.join(opt.res_video_dir, 'samelength.mp4')
#         pre_blend_video_path = os.path.join(opt.res_video_dir, 'pre_blend.mp4')
#
#         # Call blend.py for blending and masking
#         cmd = [
#             'python', 'utils/blend.py',
#             '--samelength_video_path', samelength_video_path,
#             '--pre_blend_video_path', pre_blend_video_path
#         ]
#         subprocess.call(cmd, shell=True)








