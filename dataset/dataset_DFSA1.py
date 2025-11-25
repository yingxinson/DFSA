
import torch
import numpy as np
import json
import random
import cv2
from torch.utils.data import Dataset


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


class DFSADataset(Dataset):
    def __init__(self, path_json, augment_num, mouth_region_size):
        super(DFSADataset, self).__init__()
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
        

        if video_clip_num < 6:
            raise ValueError(f"Video {video_name} has insufficient clips: {video_clip_num}")
        
        random_anchor = random.sample(range(video_clip_num), 6)
        source_anchor, reference_anchor_list = random_anchor[0], random_anchor[1:]


        source_image_path_list = video_clip_list[source_anchor]['frame_path_list']
        source_random_index = random.randint(2, min(6, len(source_image_path_list) - 1))

        source_image_data = cv2.imread(source_image_path_list[source_random_index])
        if source_image_data is None:
            raise ValueError(f"Failed to load image: {source_image_path_list[source_random_index]}")
        
        source_image_data = cv2.cvtColor(source_image_data, cv2.COLOR_BGR2RGB)
        source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA) / 255.0
        source_image_mask = source_image_data.copy()
        source_image_mask[self.radius:self.radius+self.mouth_region_size, self.radius_1_4:self.radius_1_4+self.mouth_region_size, :] = 0


        deep_speech_list = video_clip_list[source_anchor]['deep_speech_list']
        valid_start = max(2, 0)
        valid_end = min(len(deep_speech_list) - 3, len(deep_speech_list) - 1)
        if valid_end < valid_start:
            raise ValueError(f"Invalid DeepSpeech range for video {video_name}")

        source_random_index = random.randint(valid_start, valid_end)
        deepspeech_feature = np.array(deep_speech_list[source_random_index - 2:source_random_index + 3])




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

        source_image_data = torch.from_numpy(source_image_data).float().permute(2, 0, 1)
        source_image_mask = torch.from_numpy(source_image_mask).float().permute(2, 0, 1)
        reference_clip_data = torch.from_numpy(reference_clip_data).float().permute(2, 0, 1)
        deepspeech_feature = torch.from_numpy(deepspeech_feature).float().permute(1, 0)

        
        return source_image_data, source_image_mask, reference_clip_data, deepspeech_feature

    def __len__(self):
        return self.length


