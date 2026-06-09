'''
import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset


def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
    
    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0],random_anchor[1:]
        ## load source image
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.sample(range(2, 7), 1)[0]
        source_image_data = cv2.imread(source_image_path_list[source_random_index])[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h))/ 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size,self.radius_1_4:self.radius_1_4 +self.mouth_region_size ,:] = 0

        ## load deep speech feature
        deepspeech_feature = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_random_index - 2:source_random_index + 3])     
        
        
        
        
        ## load reference images
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor]['frame_path_list']
            reference_random_index = random.sample(range(9), 1)[0]
            reference_frame_path = reference_frame_path_list[reference_random_index]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h))/ 255.0
            reference_frame_data_list.append(reference_frame_data)
        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        # display the source image and reference images
        # display_img = np.concatenate([source_image_data,source_image_mask]+reference_frame_data_list,1)
        # cv2.imshow('image display',(display_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)

        # # to tensor
        source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1,0)
        
        
        
        
        return source_image_data,source_image_mask, reference_clip_data,deepspeech_feature
    
    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0], random_anchor[1:]

        # 加载源图像
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_random_index = random.sample(range(2, 7), 1)[0]  # 保持原范围但需确保有效性
        source_image_data = cv2.imread(source_image_path_list[source_random_index])[:, :, ::-1]
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size, self.radius_1_4:self.radius_1_4+self.mouth_region_size, :] = 0

        # 修复音频特征切片
        deep_speech_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list']
    
        # 动态计算有效索引范围
        valid_start = 2  # 保证source_random_index-2 >= 0
        valid_end = len(deep_speech_list) - 3  # 保证source_random_index+3 <= len(list)
        if valid_end < valid_start:  # 处理异常情况
            valid_start = 0
            valid_end = len(deep_speech_list) - 5
    
        # 生成有效随机索引
        source_random_index = random.randint(valid_start, valid_end)
    
        # 确保切片包含5帧
        deepspeech_feature = np.array(deep_speech_list[source_random_index-2:source_random_index+3])

        # 加载参考图像（保持原有逻辑）
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor]['frame_path_list']
            # 添加路径长度校验（可选）
            valid_ref_index_max = len(reference_frame_path_list) - 1
            reference_random_index = random.randint(0, valid_ref_index_max)
            reference_frame_path = reference_frame_path_list[reference_random_index]
            reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
            reference_frame_data_list.append(reference_frame_data)
        reference_clip_data = np.concatenate(reference_frame_data_list, 2)

        # 转换为张量（保持原有逻辑）
        source_image_data = torch.from_numpy(source_image_data).float().permute(2,0,1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2,0,1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2,0,1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1,0)
        
        return source_image_data, source_image_mask, reference_clip_data, deepspeech_feature
        

    def __len__(self):
        return self.length
'''
import torch
import numpy as np
import json
import random
import cv2
from torch.utils.data import Dataset

'''
def get_data(json_name, augment_num):
    print('start loading data')
    with open(json_name, 'r') as f:
        data_dic = json.load(f)
    
    data_dic_name_list = []
    for _ in range(augment_num):
        data_dic_name_list.extend(data_dic.keys())
    
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list, data_dic
'''
def get_data(json_name, augment_num):
    print('start loading data')
    with open(json_name, 'r') as f:
        data_dic = json.load(f)
    
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name, video_data in data_dic.items():
            if len(video_data['clip_data_list']) >= 6:  # 只保留合格的视频
                data_dic_name_list.append(video_name)

    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list, data_dic

'''
def get_data(json_name, augment_num):
    print('Start loading data')

    with open(json_name, 'r') as f:
        data_dic = json.load(f)

    data_dic_name_list = []
    valid_video_count = 0

    # 遍历所有视频
    for video_name, video_data in data_dic.items():
        # 条件1：至少包含6个clip
        if len(video_data['clip_data_list']) < 6:
            continue

        # 条件2：每个clip必须包含足够长度的deep_speech_list
        valid_clips = []
        for clip in video_data['clip_data_list']:
            # 检查是否存在deep_speech_list且长度足够
            if 'deep_speech_list' in clip and len(clip['deep_speech_list']) >= 10:
                valid_clips.append(clip)

        # 条件3：有效clip数量需要支持数据增强
        if len(valid_clips) >= 5:
            # 替换为验证后的clip列表
            data_dic[video_name]['clip_data_list'] = valid_clips
            # 根据增强次数添加
            for _ in range(augment_num):
                data_dic_name_list.append(video_name)
            valid_video_count += 1

    random.shuffle(data_dic_name_list)
    #print(f'Loaded {valid_video_count} valid videos')
    #print(f'Total samples after augmentation: {len(data_dic_name_list)}')
    #print('Finish loading')

    return data_dic_name_list, data_dic
'''
class DINetDataset(Dataset):
    def __init__(self, path_json, augment_num, mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list, self.data_dic = get_data(path_json, augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size // 2
        self.radius_1_4 = self.radius // 4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)
    
    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_list = self.data_dic[video_name]['clip_data_list']
        video_clip_num = len(video_clip_list)
        
        # 确保有足够的 clip
        if video_clip_num < 6:
            raise ValueError(f"Video {video_name} has insufficient clips: {video_clip_num}")
        
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0], random_anchor[1:]

        # 加载源图像
        source_image_path_list = video_clip_list[source_anchor]['frame_path_list']
        source_random_index = random.randint(2, min(6, len(source_image_path_list) - 1))

        source_image_data = cv2.imread(source_image_path_list[source_random_index])
        if source_image_data is None:
            raise ValueError(f"Failed to load image: {source_image_path_list[source_random_index]}")
        
        source_image_data = cv2.cvtColor(source_image_data, cv2.COLOR_BGR2RGB)
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA) / 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size, self.radius_1_4:self.radius_1_4+self.mouth_region_size, :] = 0

        # 处理音频特征
        deep_speech_list = video_clip_list[source_anchor]['deep_speech_list']
        valid_start = max(2, 0)
        valid_end = min(len(deep_speech_list) - 3, len(deep_speech_list) - 1)
        if valid_end < valid_start:
            raise ValueError(f"Invalid DeepSpeech range for video {video_name}")

        source_random_index = random.randint(valid_start, valid_end)
        deepspeech_feature = np.array(deep_speech_list[source_random_index - 2:source_random_index + 3])

        # # Whisper特征处理
        # whisper_feature_list = video_clip_list[source_anchor]['whisper_features']
        # # 动态窗口参数
        # window_size = 7
        # half_window = window_size // 2
        # max_idx = len(whisper_feature_list) - 1
        # # 安全索引计算
        # valid_start = max(half_window, 0)
        # valid_end = min(max_idx - half_window, max_idx)
        # if valid_end < valid_start:
        #     raise ValueError(f"Whisper特征长度不足: {len(whisper_feature_list)}")
        # selected_idx = random.randint(valid_start, valid_end)
        # whisper_window = np.array(whisper_feature_list[selected_idx - half_window: selected_idx + half_window + 1])
        # # 维度检查
        # if whisper_window.shape != (7, 384):
        #     raise RuntimeError(f"Whisper窗口维度错误: {whisper_window.shape}")


        # 加载参考图像
        reference_frame_data_list = []
        for reference_anchor in reference_anchor_list:
            reference_frame_path_list = video_clip_list[reference_anchor]['frame_path_list']
            if len(reference_frame_path_list) == 0:
                raise ValueError(f"Reference video {video_name} has no frames")
            reference_random_index = random.randint(0, len(reference_frame_path_list) - 1)

            reference_frame_data = cv2.imread(reference_frame_path_list[reference_random_index])
            if reference_frame_data is None:
                raise ValueError(f"Failed to load reference image: {reference_frame_path_list[reference_random_index]}")
            
            reference_frame_data = cv2.cvtColor(reference_frame_data, cv2.COLOR_BGR2RGB)
            reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA) / 255.0
            reference_frame_data_list.append(reference_frame_data)
        
        reference_clip_data = np.concatenate(reference_frame_data_list, axis=2)

        # 转换为张量
        source_image_data = torch.from_numpy(source_image_data).float().permute(2, 0, 1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2, 0, 1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2, 0, 1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1, 0)
        #whisper_window = torch.from_numpy(whisper_window).float().permute(1, 0)
        
        return source_image_data, source_image_mask, reference_clip_data, deepspeech_feature#whisper_window #deepspeech_feature

    def __len__(self):
        return self.length


