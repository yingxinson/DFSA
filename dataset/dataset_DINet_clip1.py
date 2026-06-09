import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset

'''
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
'''
def get_data(json_name, augment_num):
    print('start loading data')

    # 读取 JSON 文件
    with open(json_name, 'r') as f:
        data_dic = json.load(f)

    data_dic_name_list = []

    for augment_index in range(augment_num):
        for video_name, video_data in data_dic.items():
            # 确保视频数据包含 'clip_data_list'，并且片段数量 >= 5
            if 'clip_data_list' in video_data and len(video_data['clip_data_list']) >= 5:
                data_dic_name_list.append(video_name)

    # 打乱数据
    random.shuffle(data_dic_name_list)
    
    print(f'Filtered valid videos: {len(data_dic_name_list)}')
    print('Finish loading')

    return data_dic_name_list, data_dic

'''
def get_data(json_name, augment_num):
    print('start loading data')
    with open(json_name, 'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    valid_video_count = 0

    for video_name, video_data in data_dic.items():
        if 'clip_data_list' not in video_data:
            continue
        valid_clips = []
        # 检查每个clip的deep_speech_list长度 >=10
        for clip in video_data['clip_data_list']:
            if 'deep_speech_list' in clip and len(clip['deep_speech_list']) >= 10:
                valid_clips.append(clip)
        # 视频需包含至少5个有效clip
        if len(valid_clips) >= 5:
            valid_video_count += 1
            for _ in range(augment_num):
                data_dic_name_list.append(video_name)

    random.shuffle(data_dic_name_list)
    print(f'Valid videos after filtering: {valid_video_count}')
    print(f'Total samples with augmentation: {len(data_dic_name_list)}')
    return data_dic_name_list, data_dic
'''
'''
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
        video_clip_list = self.data_dic[video_name]['clip_data_list']
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        source_anchor = random.sample(range(video_clip_num), 1)[0]
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_clip_list = []
        source_clip_mask_list = []

        deep_speech_list = []
        reference_clip_list = []
        for source_frame_index in range(2, 2 + 5):
            ## load source clip
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_clip_list.append(source_image_data)
            source_image_mask = source_image_data.copy()
            source_image_mask[self.radius:self.radius + self.mouth_region_size,
            self.radius_1_4:self.radius_1_4 + self.mouth_region_size, :] = 0
            source_clip_mask_list.append(source_image_mask)

            # 确保切片在有效范围内
            deep_speech_list_ref = self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list']
            max_valid_index = len(deep_speech_list_ref) - 3
            if source_frame_index > max_valid_index:
                source_frame_index = max_valid_index



            ## load deep speech feature
            # deepspeech_array = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][source_frame_index - 2:source_frame_index + 3])
            # deep_speech_list.append(deepspeech_array)
            # 提取并检查特征
            deepspeech_slice = deep_speech_list_ref[source_frame_index - 2:source_frame_index + 3]
            deepspeech_array = np.array(deepspeech_slice)
            if deepspeech_array.ndim == 1:
                deepspeech_array = deepspeech_array.reshape(5, -1)  # 转为 (5, 1) 或 (5, dim)
            deep_speech_list.append(deepspeech_array)

            # deep_speech_list = video_clip_list[source_anchor]['deep_speech_list']
            # valid_start = max(2, 0)
            # valid_end = min(len(deep_speech_list) - 3, len(deep_speech_list) - 1)
            # if valid_end < valid_start:
            #     raise ValueError(f"Invalid DeepSpeech range for video {video_name}")

            # source_random_index = random.randint(valid_start, valid_end)
            # deepspeech_feature = np.array(deep_speech_list[source_random_index - 2:source_random_index + 3])

            ## ## load reference images
            reference_frame_list = []
            reference_anchor_list = random.sample(range(video_clip_num), 5)
            for reference_anchor in reference_anchor_list:
                reference_frame_path_list = self.data_dic[video_name]['clip_data_list'][reference_anchor][
                    'frame_path_list']
                reference_random_index = random.sample(range(9), 1)[0]
                reference_frame_path = reference_frame_path_list[reference_random_index]
                reference_frame_data = cv2.imread(reference_frame_path)[:, :, ::-1]
                reference_frame_data = cv2.resize(reference_frame_data, (self.img_w, self.img_h)) / 255.0
                reference_frame_list.append(reference_frame_data)
            reference_clip_list.append(np.concatenate(reference_frame_list, 2))

        source_clip = np.stack(source_clip_list, 0)
        source_clip_mask = np.stack(source_clip_mask_list, 0)
        deep_speech_clip = np.stack(deep_speech_list, 0)
        reference_clip = np.stack(reference_clip_list, 0)
        deep_speech_full = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'])

        # # display data
        # display_source = np.concatenate(source_clip_list,1)
        # display_source_mask = np.concatenate(source_clip_mask_list,1)
        # display_reference0 = np.concatenate([reference_clip_list[0][:,:,:3],reference_clip_list[0][:,:,3:6],reference_clip_list[0][:,:,6:9],
        #                                 reference_clip_list[0][:,:,9:12],reference_clip_list[0][:,:,12:15]],1)
        # display_reference1 = np.concatenate([reference_clip_list[1][:, :, :3], reference_clip_list[1][:, :, 3:6],
        #                                 reference_clip_list[1][:, :, 6:9],
        #                                 reference_clip_list[1][:, :, 9:12], reference_clip_list[1][:, :, 12:15]],1)
        # display_reference2 = np.concatenate([reference_clip_list[2][:, :, :3], reference_clip_list[2][:, :, 3:6],
        #                                 reference_clip_list[2][:, :, 6:9],
        #                                 reference_clip_list[2][:, :, 9:12], reference_clip_list[2][:, :, 12:15]],1)
        # display_reference3 = np.concatenate([reference_clip_list[3][:, :, :3], reference_clip_list[3][:, :, 3:6],
        #                                 reference_clip_list[3][:, :, 6:9],
        #                                 reference_clip_list[3][:, :, 9:12], reference_clip_list[3][:, :, 12:15]],1)
        # display_reference4 = np.concatenate([reference_clip_list[4][:, :, :3], reference_clip_list[4][:, :, 3:6],
        #                                 reference_clip_list[4][:, :, 6:9],
        #                                 reference_clip_list[4][:, :, 9:12], reference_clip_list[4][:, :, 12:15]],1)
        # merge_img = np.concatenate([display_source,display_source_mask,
        #                             display_reference0,display_reference1,display_reference2,display_reference3,
        #                             display_reference4],0)
        # cv2.imshow('test',(merge_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)



        # # 2 tensor
        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        source_clip_mask = torch.from_numpy(source_clip_mask).float().permute(0, 3, 1, 2)
        reference_clip = torch.from_numpy(reference_clip).float().permute(0, 3, 1, 2)
        deep_speech_clip = torch.from_numpy(deep_speech_clip).float().permute(0, 2, 1)
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0)
        return source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full

    def __len__(self):
        return self.length
'''
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
        # Determine desired length for deep_speech_full (adjust as needed)
        self.desired_length = 10  # Set based on your dataset's requirements

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_list = self.data_dic[video_name]['clip_data_list']
        video_clip_num = len(video_clip_list)
        source_anchor = random.sample(range(video_clip_num), 1)[0]
        source_image_path_list = video_clip_list[source_anchor]['frame_path_list']
        source_clip_list = []
        source_clip_mask_list = []
        deep_speech_list = []
        reference_clip_list = []

        for source_frame_index in range(2, 7):  # 2 to 6 inclusive (5 iterations)
            # Load source clip
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_clip_list.append(source_image_data)
            # Create mask
            source_image_mask = source_image_data.copy()
            mask_region = (slice(self.radius, self.radius + self.mouth_region_size),
                           slice(self.radius_1_4, self.radius_1_4 + self.mouth_region_size))
            source_image_mask[mask_region[0], mask_region[1], :] = 0
            source_clip_mask_list.append(source_image_mask)

            # Handle DeepSpeech feature
            deep_speech_list_ref = video_clip_list[source_anchor]['deep_speech_list']
            max_valid_index = len(deep_speech_list_ref) - 3
            if source_frame_index > max_valid_index:
                source_frame_index = max_valid_index
            deepspeech_slice = deep_speech_list_ref[source_frame_index - 2:source_frame_index + 3]
            deepspeech_array = np.array(deepspeech_slice)
            if deepspeech_array.ndim == 1:
                deepspeech_array = deepspeech_array.reshape(5, -1)
            deep_speech_list.append(deepspeech_array)

            # Load reference images
            reference_anchor_list = random.sample(range(video_clip_num), 5)
            reference_frame_list = []
            for ref_anchor in reference_anchor_list:
                ref_frame_paths = video_clip_list[ref_anchor]['frame_path_list']
                ref_idx = random.choice(range(9))
                ref_frame = cv2.imread(ref_frame_paths[ref_idx])[:, :, ::-1]
                ref_frame = cv2.resize(ref_frame, (self.img_w, self.img_h)) / 255.0
                reference_frame_list.append(ref_frame)
            reference_clip_list.append(np.concatenate(reference_frame_list, axis=2))

        # Post-process deep_speech_full to fixed length
        deep_speech_full = np.array(video_clip_list[source_anchor]['deep_speech_list'])
        current_length = deep_speech_full.shape[0]
        if current_length < self.desired_length:
            pad = np.zeros((self.desired_length - current_length, deep_speech_full.shape[1]))
            deep_speech_full = np.vstack((deep_speech_full, pad))
        elif current_length > self.desired_length:
            deep_speech_full = deep_speech_full[:self.desired_length, :]

        # Convert to tensors
        source_clip = torch.from_numpy(np.stack(source_clip_list, 0)).float().permute(0, 3, 1, 2)
        source_clip_mask = torch.from_numpy(np.stack(source_clip_mask_list, 0)).float().permute(0, 3, 1, 2)
        reference_clip = torch.from_numpy(np.stack(reference_clip_list, 0)).float().permute(0, 3, 1, 2)
        deep_speech_clip = torch.from_numpy(np.stack(deep_speech_list, 0)).float().permute(0, 2, 1)
        deep_speech_full = torch.from_numpy(deep_speech_full).float().permute(1, 0)

        return source_clip, source_clip_mask, reference_clip, deep_speech_clip, deep_speech_full

    def __len__(self):
        return self.length
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
        self.desired_length = 10  # 可根据实际需求调整

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_list = self.data_dic[video_name]['clip_data_list']
        video_clip_num = len(video_clip_list)

        # 随机选择源片段
        source_anchor = random.randint(0, video_clip_num - 1)
        source_image_path_list = video_clip_list[source_anchor]['frame_path_list']

        # 初始化数据容器
        source_clip, source_clip_mask = [], []
        deep_speech_list = []
        reference_clip_list = []

        # 处理5个连续帧 (source_frame_index 2-6)
        for source_frame_index in range(2, 7):
            # 加载源帧
            img = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]  # BGR转RGB
            img = cv2.resize(img, (self.img_w, self.img_h)) / 255.0
            source_clip.append(img)

            # 创建掩码
            mask = img.copy()
            y_slice = slice(self.radius, self.radius + self.mouth_region_size)
            x_slice = slice(self.radius_1_4, self.radius_1_4 + self.mouth_region_size)
            mask[y_slice, x_slice, :] = 0
            source_clip_mask.append(mask)

            # 处理DeepSpeech特征
            ds_list = video_clip_list[source_anchor]['deep_speech_list']
            valid_idx = min(source_frame_index, len(ds_list) - 3)  # 确保切片有效
            ds_feat = np.array(ds_list[valid_idx - 2:valid_idx + 3])
            if ds_feat.ndim == 1:
                ds_feat = ds_feat.reshape(5, -1)
            deep_speech_list.append(ds_feat)

            # 修改点1：只加载1个参考帧
            # 随机选择参考片段
            ref_anchor = random.randint(0, video_clip_num - 1)
            # 从参考片段中随机选1帧
            ref_paths = video_clip_list[ref_anchor]['frame_path_list']
            ref_img = cv2.imread(ref_paths[random.randint(0, 8)])[:, :, ::-1]
            ref_img = cv2.resize(ref_img, (self.img_w, self.img_h)) / 255.0
            # 修改点2：直接使用单帧，无需拼接
            reference_clip_list.append(ref_img)  # 形状 [H, W, 3]

        # 处理完整DeepSpeech序列
        deep_speech_full = np.array(video_clip_list[source_anchor]['deep_speech_list'])
        # 填充或截断序列
        if len(deep_speech_full) < self.desired_length:
            deep_speech_full = np.pad(deep_speech_full,
                                      ((0, self.desired_length - len(deep_speech_full)), (0, 0)),
                                      mode='constant')
        else:
            deep_speech_full = deep_speech_full[:self.desired_length]

        # 转换为张量
        source_clip = torch.from_numpy(np.stack(source_clip)).float().permute(0, 3, 1, 2)  # [5, 3, H, W]
        source_clip_mask = torch.from_numpy(np.stack(source_clip_mask)).float().permute(0, 3, 1, 2)
        # 修改点3：调整参考帧维度
        reference_clip = torch.from_numpy(np.stack(reference_clip_list)).float().permute(0, 3, 1, 2)  # [5, 3, H, W]
        deep_speech_clip = torch.from_numpy(np.stack(deep_speech_list)).float().permute(0, 2, 1)  # [5, feat_dim, 5]
        deep_speech_full = torch.from_numpy(deep_speech_full).float().permute(1, 0)  # [feat_dim, 10]

        return source_clip, source_clip_mask, reference_clip, deep_speech_clip, deep_speech_full

    def __len__(self):
        return self.length