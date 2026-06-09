# inference_audio_noise.py
# -*- coding: utf-8 -*-

import os
import sys
import glob
import cv2
import torch
import subprocess
import random
import numpy as np
from collections import OrderedDict

import dlib

from utils.deep_speech import DeepSpeech
from utils.data_processing import compute_crop_radius
from config.config import DFSAInferenceOptions
from models.DFSA import DFSA


# =========================
# 0) Pre-parse custom args WITHOUT touching DFSAInferenceOptions
#    (remove --audio_snr_db from sys.argv to avoid "unrecognized arguments")
# =========================
def pop_arg_value(argv, key: str):
    """
    If key exists in argv, pop it and its value from argv, return the value.
    Supports: --key value
    """
    if key in argv:
        i = argv.index(key)
        if i + 1 >= len(argv):
            raise ValueError(f"Missing value for {key}")
        val = argv[i + 1]
        # remove both
        del argv[i:i + 2]
        return val
    return None


# =========================
# 1) Audio utilities: convert to wav + add AWGN at target SNR
# =========================
def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert any audio to 16kHz mono wav using ffmpeg.
    If already .wav, return as-is.
    """
    if audio_path.lower().endswith(".wav"):
        return audio_path

    output_path = os.path.splitext(audio_path)[0] + ".wav"
    command = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path


def _read_wav(path: str):
    """
    Read wav as float32 in [-1, 1]. Prefer soundfile; fallback to scipy.
    """
    try:
        import soundfile as sf
        x, sr = sf.read(path)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x.astype(np.float32)
        return x, sr
    except Exception:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        # int16 -> float32
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32)
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        return x, sr


def _write_wav(path: str, x: np.ndarray, sr: int):
    """
    Write wav float32 [-1,1]. Prefer soundfile; fallback to scipy.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        import soundfile as sf
        sf.write(path, x, sr)
    except Exception:
        from scipy.io import wavfile
        y = (x * 32767.0).astype(np.int16)
        wavfile.write(path, sr, y)


def add_awgn_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise to achieve target SNR(dB).
    """
    x = x.astype(np.float64)
    p_signal = np.mean(x ** 2)
    if p_signal < 1e-12:
        return x.astype(np.float32)

    noise = rng.standard_normal(size=x.shape).astype(np.float64)
    p_noise = np.mean(noise ** 2)

    p_noise_target = p_signal / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(p_noise_target / (p_noise + 1e-12))

    y = x + scale * noise
    return y.astype(np.float32)


def make_noisy_wav(in_wav_path: str, out_wav_path: str, snr_db: float, seed: int):
    x, sr = _read_wav(in_wav_path)
    rng = np.random.default_rng(seed)
    y = add_awgn_snr(x, snr_db=float(snr_db), rng=rng)
    _write_wav(out_wav_path, y, sr)


# =========================
# 2) Video utilities
# =========================
def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print("warning: the input video is not 25 fps, it would be better to trans it to 25 fps!")
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    os.makedirs(save_dir, exist_ok=True)

    for i in range(frames):
        ret, frame = videoCapture.read()
        if not ret:
            break
        result_path = os.path.join(save_dir, str(i).zfill(6) + ".jpg")
        cv2.imwrite(result_path, frame)

    videoCapture.release()
    return (frame_width, frame_height)


# =========================
# 3) Dlib landmarks
# =========================
face_detector = dlib.get_frontal_face_detector()

def load_landmark_dlib(image_path, landmark_predictor):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError(f"No faces found in: {image_path}")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
    return landmarks


# =========================
# 4) Main
# =========================
if __name__ == "__main__":
    # ---- (A) Parse custom arg: --audio_snr_db (and remove it from sys.argv) ----
    snr_str = pop_arg_value(sys.argv, "--audio_snr_db")
    audio_snr_db = float(snr_str) if snr_str is not None else None

    # ---- (B) Parse original project args (unchanged) ----
    opt = DFSAInferenceOptions().parse_args()

    if not os.path.exists(opt.source_video_path):
        raise FileNotFoundError(f"Wrong video path: {opt.source_video_path}")

    # output dir
    os.makedirs(opt.res_video_dir, exist_ok=True)

    # ---- (C) Convert audio to wav ----
    opt.driving_audio_path = convert_audio_to_wav(opt.driving_audio_path)
    if not os.path.exists(opt.driving_audio_path):
        raise FileNotFoundError(f"Wrong audio path: {opt.driving_audio_path}")

    # ---- (D) Add noise at test time (ONLY audio), if requested ----
    if audio_snr_db is not None:
        base = os.path.splitext(os.path.basename(opt.driving_audio_path))[0]
        noisy_dir = os.path.join(opt.res_video_dir, "noisy_audio")
        noisy_wav_path = os.path.join(noisy_dir, f"{base}_snr{int(audio_snr_db)}dB.wav")

        # reproducible per audio file + snr
        seed = (abs(hash(base)) + int(audio_snr_db * 10)) % (2**32)
        make_noisy_wav(opt.driving_audio_path, noisy_wav_path, snr_db=audio_snr_db, seed=seed)

        print(f"[AudioNoise] Using noisy audio: {noisy_wav_path} (SNR={audio_snr_db} dB, seed={seed})")
        opt.driving_audio_path = noisy_wav_path
    else:
        print("[AudioNoise] Disabled (clean audio).")

    # ---- (E) Extract frames from source video ----
    print(f"extracting frames from video: {opt.source_video_path}")
    video_frame_dir = opt.source_video_path.replace(".mp4", "")
    os.makedirs(video_frame_dir, exist_ok=True)
    video_size = extract_frames_from_video(opt.source_video_path, video_frame_dir)

    # ---- (F) Extract deepspeech feature from (possibly noisy) audio ----
    print(f"extracting deepspeech feature from: {opt.driving_audio_path}")
    if not os.path.exists(opt.deepspeech_model_path):
        raise FileNotFoundError("pls download pretrained model of deepspeech")
    DSModel = DeepSpeech(opt.deepspeech_model_path)
    ds_feature = DSModel.compute_audio_feature(opt.driving_audio_path)
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode="edge")

    # ---- (G) Face landmarks via dlib ----
    # You hard-coded predictor path; keep it but make it exist-check
    predictor_path = getattr(opt, "dlib_predictor_path", "D:/Python/team/DFSA2/asserts/shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Missing dlib predictor: {predictor_path}")
    landmark_predictor = dlib.shape_predictor(predictor_path)

    print("Tracking Face (dlib landmarks)")
    video_frame_path_list = sorted(glob.glob(os.path.join(video_frame_dir, "*.jpg")))
    if len(video_frame_path_list) == 0:
        raise RuntimeError(f"No frames found in {video_frame_dir}")

    video_landmark_data = np.array([load_landmark_dlib(frame, landmark_predictor) for frame in video_frame_path_list])
    # reshape to (T, 68, 2)
    video_landmark_data = video_landmark_data.reshape((-1, 68, 2))

    # ---- (H) Align frames with driving audio length (same logic as yours) ----
    print("aligning frames with driving audio")
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
        res_video_landmark_data = np.concatenate(
            [video_landmark_data_cycle] * divisor + [video_landmark_data_cycle[:remainder, :, :]],
            0
        )

    res_video_frame_path_list_pad = [video_frame_path_list_cycle[0]] * 2 + res_video_frame_path_list + [video_frame_path_list_cycle[-1]] * 2
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode="edge")

    assert ds_feature_padding.shape[0] == len(res_video_frame_path_list_pad) == res_video_landmark_data_pad.shape[0]
    pad_length = ds_feature_padding.shape[0]

    # ---- (I) Select 5 reference images ----
    print("selecting five reference images")
    ref_img_list = []
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)

    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    for ref_index in ref_index_list:
        crop_flag, crop_radius = compute_crop_radius(video_size, res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
            raise RuntimeError("our method can not handle videos with large change of facial size!!")

        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]  # BGR->RGB
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]

        ref_img_crop = ref_img[
            ref_landmark[29, 1] - crop_radius: ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
            ref_landmark[33, 0] - crop_radius - crop_radius_1_4: ref_landmark[33, 0] + crop_radius + crop_radius_1_4,
            :
        ]
        ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
        ref_img_crop = ref_img_crop.astype(np.float32) / 255.0
        ref_img_list.append(ref_img_crop)

    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()

    # ---- (J) Load model ----
    print(f"loading pretrained model from: {opt.pretrained_clip_DFSA_path}")
    if not os.path.exists(opt.pretrained_clip_DFSA_path):
        raise FileNotFoundError(f"wrong path of pretrained model weight: {opt.pretrained_clip_DFSA_path}")

    model = DFSA(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    state_dict = torch.load(opt.pretrained_clip_DFSA_path, map_location="cuda")["state_dict"]["net_g"]

    def fix_key(k: str):
        return k.replace("module.", "")

    new_state_dict = {fix_key(k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # ---- (K) Inference frame by frame ----
    os.makedirs(opt.res_video_dir, exist_ok=True)

    suffix = f"_snr{int(audio_snr_db)}dB" if audio_snr_db is not None else "_clean"
    res_video_path = os.path.join(
        opt.res_video_dir,
        os.path.basename(opt.source_video_path)[:-4] + f"_facial_dubbing{suffix}.mp4"
    )
    if os.path.exists(res_video_path):
        os.remove(res_video_path)

    res_face_path = res_video_path.replace(".mp4", "_synthetic_face.mp4")
    if os.path.exists(res_face_path):
        os.remove(res_face_path)

    videowriter = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*"XVID"), 25, video_size)
    videowriter_face = cv2.VideoWriter(res_face_path, cv2.VideoWriter_fourcc(*"XVID"), 25, (resize_w, resize_h))

    for clip_end_index in range(5, pad_length, 1):
        print(f"synthesizing {clip_end_index - 5}/{pad_length - 5} frame")

        crop_flag, crop_radius = compute_crop_radius(
            video_size,
            res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],
            random_scale=1.05
        )
        if not crop_flag:
            raise RuntimeError("our method can not handle videos with large change of facial size!!")

        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[:, :, ::-1]  # BGR->RGB
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]

        crop_frame_data = frame_data[
            frame_landmark[29, 1] - crop_radius: frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
            frame_landmark[33, 0] - crop_radius - crop_radius_1_4: frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
            :
        ]

        crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h))
        crop_frame_data = crop_frame_data.astype(np.float32) / 255.0

        # keep your mask behavior
        crop_frame_data[
            opt.mouth_region_size // 2: opt.mouth_region_size // 2 + opt.mouth_region_size,
            opt.mouth_region_size // 8: opt.mouth_region_size // 8 + opt.mouth_region_size,
            :
        ] = 0.0

        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0

        videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))  # RGB->BGR

        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
        frame_data[
            frame_landmark[29, 1] - crop_radius: frame_landmark[29, 1] + crop_radius * 2,
            frame_landmark[33, 0] - crop_radius - crop_radius_1_4: frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
            :
        ] = pre_frame_resize[:crop_radius * 3, :, :]

        videowriter.write(frame_data[:, :, ::-1])  # RGB->BGR

    videowriter.release()
    videowriter_face.release()

    # ---- (L) Add audio back (use the SAME audio path used for DeepSpeech feature) ----
    video_add_audio_path = res_video_path.replace(".mp4", "_add_audio.mp4")
    if os.path.exists(video_add_audio_path):
        os.remove(video_add_audio_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", res_video_path,
        "-i", opt.driving_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        "-map", "0:v:0",
        "-map", "1:a:0",
        video_add_audio_path
    ]
    subprocess.run(cmd, check=True)

    print(f"[Done] Video saved to: {video_add_audio_path}")
