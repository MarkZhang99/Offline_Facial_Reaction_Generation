import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import time
import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from multiprocessing import Pool
from scipy.io import loadmat
from functools import cmp_to_key
from scipy.io import wavfile
import python_speech_features as psf  # pip install python_speech_features


def _ensure_tensor(x, shape_tail, dtype=torch.float32):
    """
    如果 x 不是 Tensor，就返回一个全 0 的 Tensor，shape=(0, *shape_tail)；
    否则把 x 转到指定 dtype/device 后返回。
    """
    if not isinstance(x, torch.Tensor):
        return torch.zeros((0, *shape_tail), dtype=dtype)
    return x.to(dtype=dtype)

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames


def extract_audio_features(audio_path, fps, n_frames):
    # video_id = osp.basename(audio_path)[:-4]
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(-1)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    shifted_n_samples = 0
    curr_feats = []
    for i in range(n_frames):
        curr_samples = audio[i*frame_n_samples:shifted_n_samples + i*frame_n_samples + frame_n_samples]
        curr_mfcc = torchaudio.compliance.kaldi.mfcc(torch.from_numpy(curr_samples).float().view(1, -1), sample_frequency=sr, use_energy=True)
        curr_mfcc = curr_mfcc.transpose(0, 1) # (freq, time)
        curr_mfcc_d = torchaudio.functional.compute_deltas(curr_mfcc)
        curr_mfcc_dd = torchaudio.functional.compute_deltas(curr_mfcc_d)
        curr_mfccs = np.stack((curr_mfcc.numpy(), curr_mfcc_d.numpy(), curr_mfcc_dd.numpy())).reshape(-1)
        curr_feat = curr_mfccs
        # rms = librosa.feature.rms(curr_samples, sr).reshape(-1)
        # zcr = librosa.feature.zero_crossing_rate(curr_samples, sr).reshape(-1)
        # curr_feat = np.concatenate((curr_mfccs, rms, zcr))

        curr_feats.append(curr_feat)

    curr_feats = np.stack(curr_feats, axis=0)
    return curr_feats


class ReactionDataset(data.Dataset):
    def __init__(self,
                 root_path, split,
                 img_size=256, crop_size=224, clip_length=751, fps=25,
                 audio_dim=26, _3dmm_dim=58, emotion_dim=25,
                 load_audio=True, load_video_s=True, load_video_l=True,
                 load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,
                 repeat_mirrored=True):
        """
        root_path: Data root directory containing train/val folders
        split: "train"/"val"/"test"
        """
         # ---- Basic properties ----
        self._root_path = root_path
        self._split = split
        self._clip_length = clip_length
        self._fps = fps
        self.crop_size = crop_size
        self.audio_dim    = audio_dim
        self._3dmm_dim    = _3dmm_dim
        self._emotion_dim = emotion_dim

        # ---- Loading flags ----
        self.load_audio    = load_audio
        self.load_video_s  = load_video_s
        self.load_video_l  = load_video_l
        self.load_emotion_s= load_emotion_s
        self.load_emotion_l= load_emotion_l
        self.load_3dmm_s   = load_3dmm_s
        self.load_3dmm_l   = load_3dmm_l
        self.load_ref      = load_ref

        # ---- transform & loader ----
        self._img_loader = pil_loader
        self._transform = Transform(img_size, crop_size)
        self._transform_3dmm = transforms.Lambda(lambda e: (e - self.mean_face))

        # ---- Root directories for each file type ----
        base = os.path.join(self._root_path, self._split)
        self._video_root   = os.path.join(base, 'Video_frames')
        self._audio_root   = os.path.join(base, 'Audio_files')
        self._emotion_root = os.path.join(base, 'Emotion')
        self._3dmm_root    = os.path.join(base, '3D_FV_files')

        # ---- mean_face for 3dmm normalization ----
        mean = np.load('../external/FaceVerse/mean_face.npy').astype(np.float32)
        self.mean_face = torch.from_numpy(mean).view(1,1,-1)

        # ---- Read CSV to get speaker/listener parent directories ----
        df = pd.read_csv(os.path.join(self._root_path, self._split + '.csv'),
                         header=None, delimiter=',').drop(0)
        sp_list = df.values[:,1].tolist()
        lp_list = df.values[:,2].tolist()
        if self._split in ["val","test"] or repeat_mirrored:
            sp_list = sp_list + lp_list
            lp_list = lp_list + sp_list[:len(lp_list)]

        # ---- Build data_list, expand numeric subfolders ----
        
        self.data_list = []
        for sp, lp in zip(sp_list, lp_list):
            sp_dir = os.path.join(self._video_root, sp)
            lp_dir = os.path.join(self._video_root, lp)
            if not os.path.isdir(sp_dir) or not os.path.isdir(lp_dir):
                continue
        
            # Only do P1/P2 mapping for Emotion
            def label_folder(name):
                if name.endswith('Novice_video'):
                    return 'P1'
                elif name.endswith('Expert_video'):
                    return 'P2'
                else:
                    raise ValueError(f"Unknown video-folder: {name}")
        
            sp_label = label_folder(sp)  # 'P1' or 'P2'
            lp_label = label_folder(lp)
        
            # Iterate through each numeric subfolder idx
            for idx in sorted(os.listdir(sp_dir)):
                if not idx.isdigit():
                    continue
        
                # 1) Video frames & 3DMM & audio all use original sp/idx structure
                sub_path = os.path.join(sp, idx)  # e.g. "NoXI/.../Novice_video/1"
                lp_sub_path = os.path.join(lp, idx)
        
                # 2) Only Emotion uses label directory
                sp_em_sub = os.path.join(os.path.dirname(sp), sp_label, idx)
                lp_em_sub = os.path.join(os.path.dirname(lp), lp_label, idx)
        
                paths = {
                    # Video frames folder
                    'speaker_video_path':   os.path.join(self._video_root, sub_path),
                    'listener_video_path':  os.path.join(self._video_root, lp_sub_path),
                    # Audio .wav
                    'speaker_audio_path':   os.path.join(self._audio_root, sub_path + '.wav'),
                    'listener_audio_path':  os.path.join(self._audio_root, lp_sub_path + '.wav'),
                    # Emotion .csv (mapped to P1/P2)
                    'speaker_emotion_path': os.path.join(self._emotion_root, sp_em_sub + '.csv'),
                    'listener_emotion_path':os.path.join(self._emotion_root, lp_em_sub + '.csv'),
                    # 3DMM .npy (still using Novice_video/Expert_video)
                    'speaker_3dmm_path':    os.path.join(self._3dmm_root, sub_path + '.npy'),
                    'listener_3dmm_path':   os.path.join(self._3dmm_root, lp_sub_path + '.npy'),
                }
                # Filter out missing Emotion files
            if self.load_emotion_l:
                if not os.path.exists(paths['listener_emotion_path']):
                    print(f"⚠️ Skip missing listener emotion: {paths['listener_emotion_path']}")
                    continue
            if self.load_emotion_s:
                if not os.path.exists(paths['speaker_emotion_path']):
                    print(f"⚠️ Skip missing speaker emotion: {paths['speaker_emotion_path']}")
                    continue

            # Other paths can also be checked for existence as needed
            self.data_list.append(paths)
            
        self._len = len(self.data_list)

    def __len__(self):
        return self._len

       

    def __getitem__(self, index):
        """
        Returns one data pair:
        (speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm,
         listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm,
         listener_reference)
        """
        data = self.data_list[index]
    
        # --- 1) Data augmentation: randomly swap speaker/listener during training ---
        changed = 0
        if self._split == 'train':
            changed = random.randint(0, 1)
        sp_pref = 'speaker' if changed == 0 else 'listener'
        lp_pref = 'listener' if changed == 0 else 'speaker'
    
        # --- 2) Directly use the absolute paths already constructed in data_list ---
        sp_vid_dir = data[f'{sp_pref}_video_path']      # Don't join self._video_path again
        lp_vid_dir = data[f'{lp_pref}_video_path']
        sp_aud_p   = data[f'{sp_pref}_audio_path']
        lp_aud_p   = data[f'{lp_pref}_audio_path']
        sp_em_p    = data[f'{sp_pref}_emotion_path']
        lp_em_p    = data[f'{lp_pref}_emotion_path']
        sp_3d_p    = data[f'{sp_pref}_3dmm_path']
        lp_3d_p    = data[f'{lp_pref}_3dmm_path']
    
        # --- 3) Frame file collection (supports one-level or two-level directories) ---
        def collect_frames(folder):
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Frame-folder not found: {folder}")
            rels = []
            for name in sorted(os.listdir(folder)):
                p = os.path.join(folder, name)
                if os.path.isdir(p):
                    # Second-level subfolder
                    for f in os.listdir(p):
                        if f.endswith('.png') and f[:-4].isdigit():
                            rels.append(os.path.join(name, f))
                elif name.endswith('.png') and name[:-4].isdigit():
                    # First-level directory directly has png
                    rels.append(name)
            if not rels:
                raise RuntimeError(f"No .png frames under {folder}")
            # Sort by frame number
            rels.sort(key=lambda x: int(os.path.basename(x)[:-4]))
            return rels
    
        sp_frames = collect_frames(sp_vid_dir)
        lp_frames = collect_frames(lp_vid_dir)
    
        # --- 4) Extract continuous clip segment ---
        total = len(sp_frames)
        if self._clip_length < total:
            start = random.randint(0, total - self._clip_length)
        else:
            start = 0
        end = start + min(self._clip_length, total)
        sp_sel = sp_frames[start:end]
        lp_sel = lp_frames[start:end]
    
        # --- 5) Load speaker video clip ---
        speaker_video_clip = torch.zeros(0)
        if self.load_video_s:
            clips = []
            for rel in sp_sel:
                img = self._img_loader(os.path.join(sp_vid_dir, rel))
                img = self._transform(img)
                clips.append(img.unsqueeze(0))
            speaker_video_clip = torch.cat(clips, dim=0)
    
        # --- 6) Load listener video clip ---
        listener_video_clip = torch.zeros(0)
        if self.load_video_l:
            clips = []
            for rel in lp_sel:
                img = self._img_loader(os.path.join(lp_vid_dir, rel))
                img = self._transform(img)
                clips.append(img.unsqueeze(0))
            listener_video_clip = torch.cat(clips, dim=0)
    
        # --- 7) Load audio & compute MFCC (pure Python) ---
        speaker_audio_clip = None
        listener_audio_clip = None
        if self.load_audio:
            # 7.1 读 wav
            audio_path = data[f'{sp_pref}_audio_path']
            sr, wav_np = wavfile.read(audio_path)
            # 归一化到 [-1,1]
            if wav_np.dtype.kind in ('i', 'u'):
                wav_np = wav_np.astype(np.float32) / np.iinfo(wav_np.dtype).max
            else:
                wav_np = wav_np.astype(np.float32)
            # 多声道转单声道
            if wav_np.ndim > 1:
                wav_np = wav_np.mean(axis=1)

            # 7.2 计算 3×26 = 78 维 MFCC
            mfcc = psf.mfcc(
                signal     = wav_np,                   # ← 这里一定要用 wav_np！
                samplerate = sr,
                winlen     = 1.0 / self._fps,
                winstep    = 1.0 / self._fps,
                numcep     = 26,                       # 26 静态频谱
                nfft       = max(512, int(sr / self._fps))
            )
            d_mfcc  = psf.delta(mfcc, 2)             # 一阶差分
            dd_mfcc = psf.delta(d_mfcc, 2)           # 二阶差分
            mfcc78 = np.hstack([mfcc, d_mfcc, dd_mfcc])  # (T,78)

            # 7.3 截取这一段
            speaker_audio_clip = torch.from_numpy(mfcc78[start:end]).float()
            listener_audio_clip = torch.from_numpy(mfcc78[start:end]).float()


        # --- 8) Load emotion ---
        speaker_emotion, listener_emotion = 0, 0
        if self.load_emotion_s:
            em = pd.read_csv(sp_em_p, header=None).drop(0).values.astype(np.float32)
            speaker_emotion = torch.from_numpy(em)[start:end]
        if self.load_emotion_l:
            em = pd.read_csv(lp_em_p, header=None).drop(0).values.astype(np.float32)
            listener_emotion = torch.from_numpy(em)[start:end]
    
        # --- 9) Load 3DMM ---
        speaker_3dmm, listener_3dmm = 0, 0
        if self.load_3dmm_s:
            m = np.load(sp_3d_p).astype(np.float32)
            m = torch.from_numpy(m).squeeze()[start:end]
            speaker_3dmm = self._transform_3dmm(m)[0]
        if self.load_3dmm_l:
            m = np.load(lp_3d_p).astype(np.float32)
            m = torch.from_numpy(m).squeeze()[start:end]
            listener_3dmm = self._transform_3dmm(m)[0]
    
        # --- 10) Listener reference frame (first frame) ---
        listener_reference = 0
        if self.load_ref:
            ref = lp_frames[0]
            img = self._img_loader(os.path.join(lp_vid_dir, ref))
            listener_reference = self._transform(img)
        
        # —— 1) Video frames pad / truncate —— 
        L = self._clip_length
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1) 确保所有分支都返回 Tensor
        # Video: shape_tail=(C,H,W)
        speaker_video_clip  = _ensure_tensor(speaker_video_clip,  shape_tail=(3, self.crop_size, self.crop_size))
        listener_video_clip = _ensure_tensor(listener_video_clip, shape_tail=(3, self.crop_size, self.crop_size))
        # Audio MFCC: shape_tail=(audio_dim*3,)
        mfcc_dim = self.audio_dim * 3
        speaker_audio_clip  = _ensure_tensor(speaker_audio_clip,  shape_tail=(mfcc_dim,))
        listener_audio_clip = _ensure_tensor(listener_audio_clip, shape_tail=(mfcc_dim,))
        # 3DMM: shape_tail=(_3dmm_dim,)
        speaker_3dmm  = _ensure_tensor(speaker_3dmm,  shape_tail=(self._3dmm_dim,))
        listener_3dmm = _ensure_tensor(listener_3dmm, shape_tail=(self._3dmm_dim,))
        # Emotion: shape_tail=(emotion_dim,)
        speaker_emotion  = _ensure_tensor(speaker_emotion,  shape_tail=(self._emotion_dim,))
        listener_emotion = _ensure_tensor(listener_emotion, shape_tail=(self._emotion_dim,))

        # 2) 定义一个通用的 pad/truncate 函数
        def pad_trunc(x):
            T = x.shape[0]
            if T < L:
                # 如果序列太短，就用最后一帧（或最后一个向量）重复补足
                pad = x[-1:].repeat(L - T, *([1] * (x.ndim - 1)))
                return torch.cat([x, pad], dim=0)
            else:
                # 否则截断
                return x[:L]

        # 3) 对每个 modality 应用
        speaker_video_clip  = pad_trunc(speaker_video_clip)
        listener_video_clip = pad_trunc(listener_video_clip)
        speaker_audio_clip  = pad_trunc(speaker_audio_clip)
        listener_audio_clip = pad_trunc(listener_audio_clip)
        speaker_3dmm        = pad_trunc(speaker_3dmm)
        listener_3dmm       = pad_trunc(listener_3dmm)
        speaker_emotion     = pad_trunc(speaker_emotion)
        listener_emotion    = pad_trunc(listener_emotion)
            
        return (
            speaker_video_clip,
            speaker_audio_clip,
            speaker_emotion,
            speaker_3dmm,
            listener_video_clip,
            listener_audio_clip,
            listener_emotion,
            listener_3dmm,
            listener_reference
        )


    def __len__(self):
        return self._len


def get_dataloader(conf, split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=True):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    # print('==> Preparing data for {}...'.format(split) + '\n')
    dataset = ReactionDataset(
        conf.dataset_path, split,
        img_size     = conf.img_size,
        crop_size    = conf.crop_size,
        clip_length  = conf.clip_length,
        fps          = conf.fps if hasattr(conf, 'fps') else 25,
        audio_dim    = conf.audio_dim,
        _3dmm_dim    = conf._3dmm_dim,
        emotion_dim  = conf.emotion_dim,
        load_audio   = load_audio,
        load_video_s = load_video_s,
        load_video_l = load_video_l,
        load_emotion_s= load_emotion_s,
        load_emotion_l= load_emotion_l,
        load_3dmm_s  = load_3dmm_s,
        load_3dmm_l  = load_3dmm_l,
        load_ref     = load_ref,
        repeat_mirrored=repeat_mirrored
    )
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset=dataset, batch_size=conf.batch_size, shuffle=shuffle, num_workers=conf.num_workers, pin_memory=True)
    return dataloader
