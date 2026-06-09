import os
import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import Wav2Vec2Processor, HubertModel

class HuBERT():
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device="cuda:0"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(device)

    def extract_feature(self, audio_path):
        # 读取音频
        speech, sr = sf.read(audio_path)
        if len(speech.shape) > 1:  # 立体声转单声道
            speech = np.mean(speech, axis=1)
        # 采样率转换
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000, res_type='soxr_vhq')

        # 处理音频
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)

        # 提取 HuBERT 特征
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state

        # 调整形状
        hidden_states = hidden_states.squeeze(0).cpu().numpy()  # 变为 (T, 1024)
        return hidden_states

if __name__ == "__main__":
    audio_path = "./00168.wav"
    hubert = HuBERT()
    hubert_features = hubert.extract_feature(audio_path)
    print(f"HuBERT 特征形状: {hubert_features.shape}")  # (T, 1024)
