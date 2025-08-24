import os, torch, numpy as np
import argparse
import sys
sys.path.append(os.path.abspath("..")) 
from torch.utils.data import DataLoader
from trainer.dataset import ReactionDataset
from model.TransformerVAE import TransformerVAE
from utils.render import Render
from utils.utils import torch_img_to_np2
# —— 配置 —— 
DATA_ROOT  = "../dataset/data"
CKPT       = "./results/train_offline/best_checkpoint.pth"
OUT_DIR    = "./results/one_render"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUT_DIR, exist_ok=True)

# —— 加载模型 & 数据 —— 
model = TransformerVAE(  # 同你的 train.py 参数
    img_size=256, audio_dim=78, output_3dmm_dim=58,
    output_emotion_dim=25, feature_dim=128,
    seq_len=751, online=False, window_size=8,
    device=str(DEVICE)
)
ckpt = torch.load(CKPT, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.to(DEVICE).eval()

dataset = ReactionDataset(
    root_path=DATA_ROOT,
    split="val",
    img_size=256,
    crop_size=224,
    clip_length=751,
    fps=25,
    load_audio=True,
    load_video_s=True,
    load_video_l=False,
    load_emotion_s=False,
    load_emotion_l=False,
    load_3dmm_s=False,
    load_3dmm_l=True,
    load_ref=True,
)
dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

batch = next(iter(dl))
# unpack并移到device
(sv, sa, se, s3, lv, la, le, l3, lref) = batch
# 只要 sv, sa, l3, lref
sv = sv.to(DEVICE)
sa = sa.to(DEVICE)

# —— 前向，拿到 listener_3dmm_out —— 
with torch.no_grad():
    l3_out, le_out, _ = model(sv, sa)

# 5) CPU 渲染
sv0   = sv[0].cpu()
l3_0  = l3_out[0].cpu()
ref0  = lref[0].cpu()
render = Render(device='cpu')
render.rendering_for_fid(
    OUT_DIR,
    "sample0",
    l3_0,
    sv0,
    ref0,
    sv0[:751]   # 或者 listener_video_clip 的前 751 帧
)
print("result saved at", OUT_DIR)