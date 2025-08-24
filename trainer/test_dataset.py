# test_loader.py
import os, sys
from types import SimpleNamespace
from torch.utils.data import DataLoader

# 确保能 import 到你的 ReactionDataset
sys.path.append(os.path.abspath("trainer"))

from dataset import ReactionDataset  # 根据你实际文件名修改
import numpy as np

def main():
    conf = SimpleNamespace(
        dataset_path="../dataset/data",
        split="train",
        img_size=256,
        crop_size=224,
        clip_length=2,
        fps=25,
        load_audio=True,
        load_video_s=False,
        load_video_l=False,
        load_emotion_s=False,
        load_emotion_l=False,
        load_3dmm_s=False,
        load_3dmm_l=False,
        load_ref=False,
        repeat_mirrored=True
    )

    ds = ReactionDataset(
        root_path=conf.dataset_path,
        split=conf.split,
        img_size=conf.img_size,
        crop_size=conf.crop_size,
        clip_length=conf.clip_length,
        fps=conf.fps,
        load_audio=conf.load_audio,
        load_video_s=conf.load_video_s,
        load_video_l=conf.load_video_l,
        load_emotion_s=conf.load_emotion_s,
        load_emotion_l=conf.load_emotion_l,
        load_3dmm_s=conf.load_3dmm_s,
        load_3dmm_l=conf.load_3dmm_l,
        load_ref=conf.load_ref,
        repeat_mirrored=conf.repeat_mirrored
    )
    print("✅ Dataset size:", len(ds))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("✅ Batch loaded successfully!")
    for i, item in enumerate(batch):
        print(f"  Item {i:2d} →", type(item), getattr(item, "shape", item))

if __name__ == "__main__":
    main()
