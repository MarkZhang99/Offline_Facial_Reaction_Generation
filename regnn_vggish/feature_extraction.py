import torch
import torch.nn as nn
import torchvision
from models import SwinTransformer, VGGish
import pandas as pd
import os
from PIL import Image
import glob
from torch.utils import data
from torchvision import transforms

import numpy as np
import random
import pandas as pd
from PIL import Image

from decord import VideoReader
from decord import cpu

import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--data-dir', default="../data/react/cropped_face", type=str, help="dataset path")
    parser.add_argument('--save-dir', default="../data/react_clean", type=str, help="the dir to save features")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["train", "val", "test"], required=True)
    parser.add_argument('--type', type=str, help="type of features to extract", choices=["audio", "video"], required=True)
    
    args = parser.parse_args()
    return args

class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size
        
    def __call__(self, img):

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])

        img = transform(img)
        return img

def extract_audio_features(args):
    model = VGGish(preprocess=True)
    model = model.cuda()
    model.eval()

    _list_path = pd.read_csv(os.path.join(args.data_dir, args.split + '.csv'), header=None, delimiter=',')
    _list_path = _list_path.drop(0)

    all_path = [path for path in list(_list_path.values[:, 1])] + [path for path in list(_list_path.values[:, 2])]

    for path in all_path:
        ab_audio_path = os.path.join(args.data_dir, args.split, 'Audio_files', path+'.wav')

        with torch.no_grad():
            audio_features = model.forward(ab_audio_path, fs=25).cpu()
            
        site, group, pid, clip = path.split('/')
        if not os.path.exists(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid)):
            os.makedirs(os.path.join(args.save_dir, args.split, 'Audio_features', site, group, pid))

        torch.save(audio_features, os.path.join(args.save_dir, args.split, 'Audio_features', path+'.pth'))



def extract_video_features(args):

    # 初始化变换和模型
    _transform = Transform(img_size=256, crop_size=224)
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        num_classes=7
    )
    model.load_state_dict(torch.load(
        "./pretrained/swin_fer.pth",
        map_location='cpu'
    ))
    model = model.cuda().eval()

    # 读取 split CSV，获取所有相对路径（不含扩展名）
    df = pd.read_csv(
        os.path.join(args.data_dir, args.split + '.csv'),
        header=None,
        delimiter=','
    ).drop(0)
    all_paths = list(df.values[:, 1]) + list(df.values[:, 2])

    total_length = 751  # 固定帧长度
    chunk_size = 50     # 每批送入 GPU 的帧数

    for rel_path in all_paths:
        # 直接把路径后面加 .mp4
        ab_video_path = os.path.join(
            args.data_dir,
            args.split,
            'Video_files',
            rel_path + '.mp4'
        )
        if not os.path.isfile(ab_video_path):
            raise FileNotFoundError(f"No mp4 file found: {ab_video_path}")

        # 读取并 transform 帧
        frames = []
        with open(ab_video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
            for i in range(total_length):
                frame = vr[i]
                img = Image.fromarray(frame.asnumpy())
                img_t = _transform(img).unsqueeze(0)
                frames.append(img_t)

        # 分 chunk 推理，避免 OOM
        feature_chunks = []
        for start in range(0, total_length, chunk_size):
            end = min(start + chunk_size, total_length)
            sub_clip = torch.cat(frames[start:end], dim=0).cuda()
            with torch.no_grad():
                sub_feats = model.forward_features(sub_clip).cpu()
            feature_chunks.append(sub_feats)
            torch.cuda.empty_cache()

        video_features = torch.cat(feature_chunks, dim=0)

        # 保存特征: <save_dir>/<split>/Video_features/<rel_path>.pth
        save_dir = os.path.join(
            args.save_dir,
            args.split,
            'Video_features',
            *rel_path.split('/')
        )
        os.makedirs(save_dir, exist_ok=True)
        clip_name = rel_path.split('/')[-1]
        torch.save(
            video_features,
            os.path.join(save_dir, clip_name + '.pth')
        )




def main(args):
    torch.cuda.empty_cache()
    if args.type == 'video':
        extract_video_features(args)
    elif args.type == 'audio':
        extract_audio_features(args)

# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '32'
    main(args)
