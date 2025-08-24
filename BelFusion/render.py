import os
import sys
import numpy as np
import torch
from torchvision import transforms
from skimage.io import imsave
import skvideo.io
from pathlib import Path
from tqdm import auto
import argparse
import cv2

from utils import torch_img_to_np, _fix_image, torch_img_to_np2
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator

# ==== FaceVerse 3DMM 反归一化所需均值/方差（numpy） ====
FV_MEAN = np.load('external/FaceVerse/mean_face.npy').astype(np.float32).reshape(1, 58)
FV_STD  = np.load('external/FaceVerse/std_face.npy').astype(np.float32).reshape(1, 58)


def obtain_seq_index(index, num_frames, semantic_radius = 13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim = 0)
    return semantic.transpose(1,2)



class Render(object):
    def __init__(self, device=None, enable_pirender=True):
        self.device = torch.device(device) if device is not None else (
            torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        )
        self.faceverse, _ = get_faceverse(device=str(self.device), img_size=224)
        self.faceverse.init_coeff_tensors()

        self.id_tensor = torch.from_numpy(
            np.load('external/FaceVerse/reference_full.npy')
        ).float().view(1,-1)[:,:150].to(self.device)

        self.pi_render = FaceGenerator().to(self.device)
        self.pi_render.eval()
        ckpt_path = 'external/PIRender/cur_model_fold.pth'
        print("PI-Render ckpt:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt.get('model', ckpt.get('net', ckpt)))
        self.pi_render.load_state_dict(sd, strict=False)

        self.mean_face = torch.from_numpy(
            np.load('external/FaceVerse/mean_face.npy').astype(np.float32)
        ).view(1,1,-1).to(self.device)
        self.std_face = torch.from_numpy(
            np.load('external/FaceVerse/std_face.npy').astype(np.float32)
        ).view(1,1,-1).to(self.device)

        self._reverse_transform_3dmm = transforms.Lambda(
            lambda e: e * self.std_face + self.mean_face
        )


    def rendering(self, path, ind, listener_vectors, speaker_video_clip, listener_reference):

        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:,:52].view(T,-1).to(self.device)
        self.faceverse.rot_tensor = listener_vectors[:,52:55].view(T, -1).to(self.device)
        self.faceverse.trans_tensor = listener_vectors[:,55:].view(T, -1).to(self.device)
        self.faceverse.id_tensor = self.id_tensor.view(1,150).repeat(T,1).view(T,150).to(self.device)


        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)


        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(self.device)
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).view(
            listener_vectors.shape[0], C, H, W)

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (672, 224))
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()



    def rendering_for_fid(self, path, ind, listener_vectors, speaker_video_clip, listener_reference, listener_video_clip):
        """
        只要 FID 对比需要的假/真帧；FaceVerse 逐帧跑，避免一次性把 T 帧吃进 GPU。
        """
        import gc
        T = listener_vectors.shape[0]
        # 反归一化
        lv = listener_vectors
        # 期望输入是 [T,58]；若是 [1,T,58] 先压成 [T,58]
        if lv.ndim == 3 and lv.shape[0] == 1:
            lv = lv[0]
        # 让反归一化在 [1,T,58] 上做广播更安全
        lv = lv.unsqueeze(0)             # [1,T,58]
        lv = self._reverse_transform_3dmm(lv)  # [1,T,58] -> 乘 std + 加 mean
        listener_vectors = lv.squeeze(0)  # [T,58]

        # 输出目录
        path_real = os.path.join(path, 'fid', 'real')
        path_fake = os.path.join(path, 'fid', 'fake')
        os.makedirs(path_real, exist_ok=True)
        os.makedirs(path_fake, exist_ok=True)

        # 真帧按需取（一次只拿一帧，省内存）
        # 注意：torch_img_to_np2 支持批；这里我们按帧切，不整批转
        # speaker_video_clip 不参与 FID，对比是 fake vs listener_video_clip
        stride = 1  # 原仓库是每 30 帧抽一张
        for t in range(T):
            # === FaceVerse：单帧 ===
            self.faceverse.batch_size = 1
            self.faceverse.init_coeff_tensors()
            v = listener_vectors[t].view(1, -1).to(self.device)   # [1,C]
            self.faceverse.exp_tensor   = v[:, :52]
            self.faceverse.rot_tensor   = v[:, 52:55]
            self.faceverse.trans_tensor = v[:, 55:]
            self.faceverse.id_tensor    = self.id_tensor.view(1, 150)

            with torch.inference_mode():
                pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
            img = pred_dict['rendered_img'][0].detach().cpu().numpy()  # H W 4 or 3
            img = np.clip(img[:, :, :3], 0, 255).astype(np.uint8)

            # 每 stride 才落一张（和原逻辑一致）
            if (t % stride) == 0:
                # 假帧（渲出来的听者）
                cv2.imwrite(os.path.join(path_fake, f"{ind}_{t+1}.png"), img)

                # 真帧（GT 听者）
                # 取第 t 帧 listener_video_clip -> numpy
                lvt = listener_video_clip[t:t+1]        # [1,3,H,W]
                real = torch_img_to_np2(lvt)[0]         # H W 3, uint8 (BGR)
                cv2.imwrite(os.path.join(path_real, f"{ind}_{t+1}.png"), real)

            # 立刻释放，防显存增长
            del v, pred_dict, img
            torch.cuda.empty_cache(); gc.collect()

        # 可选：拼接三联画视频（和你原逻辑一致），这里略；需要再加。

