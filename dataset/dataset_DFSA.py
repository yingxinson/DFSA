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

            if 'clip_data_list' in video_data and len(video_data['clip_data_list']) >= 5:
                data_dic_name_list.append(video_name)


    random.shuffle(data_dic_name_list)
    
    print(f'Filtered valid videos: {len(data_dic_name_list)}')
    print('Finish loading')

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
