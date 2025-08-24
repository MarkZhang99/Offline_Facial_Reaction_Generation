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
sys.path.append(os.path.abspath("..")) 

from utils.utils import torch_img_to_np2


sys.path.append(os.path.abspath("..")) 
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator


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
    """Computes and stores the average and current value"""

    def __init__(self, device = 'cuda'):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        # 降低光栅化负载，显著减小峰值显存
        try:
            rs = self.faceverse.renderer.rasterizer.raster_settings
            if hasattr(rs, "faces_per_pixel"):
                rs.faces_per_pixel = min(getattr(rs, "faces_per_pixel", 10), 5)
            if hasattr(rs, "bin_size"):          # 0 表示 naive rasterization（小图像常更省显存）
                rs.bin_size = 0
            if hasattr(rs, "max_faces_per_bin"):
                rs.max_faces_per_bin = 1000
        except Exception:
            pass

        self.id_tensor = torch.from_numpy(np.load('../external/FaceVerse/reference_full.npy')).float().view(1,-1)[:,:150]
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()        # 2) 初始化 PiRender 模型
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'external', 'PIRender', 'cur_model_fold.pth')
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.pi_render.load_state_dict(state_dict)
        #self.mean_face = torch.FloatTensor(
            #np.load('../external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, 1, -1).to(device)
        self.std_face = torch.FloatTensor(
            np.load('../external/FaceVerse/std_face.npy').astype(np.float32)).view(1, 1, -1).to(device)

        #self._reverse_transform_3dmm = transforms.Lambda(lambda e: e  + self.mean_face)
        # load mean_face once, but defer its device to match each input e
        mean_face = torch.FloatTensor(
            np.load('../external/FaceVerse/mean_face.npy').astype(np.float32)
        ).view(1, 1, -1)

        # device‑agnostic 3DMM reverse transform:
        self._reverse_transform_3dmm = transforms.Lambda(
            lambda e, mf=mean_face: e + mf.to(e.device)
        )

    def rendering(self, path, ind, listener_vectors, speaker_video_clip, listener_reference):

        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:,:52].view(T,-1).to(listener_vectors.device)
        self.faceverse.rot_tensor = listener_vectors[:,52:55].view(T, -1).to(listener_vectors.device)
        self.faceverse.trans_tensor = listener_vectors[:,55:].view(T, -1).to(listener_vectors.device)
        self.faceverse.id_tensor = self.id_tensor.view(1,150).repeat(T,1).view(T,150).to(listener_vectors.device)


        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)


        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.device)
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
        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:, :52].view(T, -1).to(listener_vectors.device)
        self.faceverse.rot_tensor = listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.device)
        self.faceverse.trans_tensor = listener_vectors[:, 55:].view(T, -1).to(listener_vectors.device)
        self.faceverse.id_tensor = self.id_tensor.view(1, 150).repeat(T, 1).view(T, 150).to(listener_vectors.device)

        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = rendered_img_r.detach().cpu().numpy()
        rendered_img_r = np.clip(rendered_img_r, 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)
        #rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        #rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # 2D video
        # listener_vectors = torch.cat((listener_exp.view(T,-1), listener_trans.view(T, -1), listener_rot.view(T, -1)))
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.device)
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

        if not os.path.exists(os.path.join(path, 'results_videos')):
            os.makedirs(os.path.join(path, 'results_videos'))
        out = cv2.VideoWriter(os.path.join(path, 'results_videos', ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (672, 224))
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

        listener_video_clip = torch_img_to_np2(listener_video_clip)

        path_real = os.path.join(path, 'fid', 'real')
        if not os.path.exists(path_real):
            os.makedirs(path_real)
        path_fake = os.path.join(path, 'fid', 'fake')
        if not os.path.exists(path_fake):
            os.makedirs(path_fake)

        for i in range(0, rendered_img_r.shape[0], 30):

            cv2.imwrite(os.path.join(path_fake, ind+'_'+str(i+1)+'.png'), listener_videos[i])
            cv2.imwrite(os.path.join(path_real, ind+'_'+str(i+1)+'.png'), listener_video_clip[i])



    def rendering_3d_only(self, path, ind, listener_vectors, fps=25, save_frames=False, chunk_size=48):
        """
        只渲 FaceVerse 的 3D 输出；分 chunk 渲防止 OOM。
        """
        import torch, os, cv2, numpy as np
        os.makedirs(path, exist_ok=True)

        # 统一 [B, T, C]
        if listener_vectors.dim() == 3:
            listener_vectors_ = listener_vectors
        elif listener_vectors.dim() == 2:
            listener_vectors_ = listener_vectors.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected listener_vectors shape: {tuple(listener_vectors.shape)}")

        # 反归一化 + 清理 NaN/Inf
        listener_vectors_ = self._reverse_transform_3dmm(listener_vectors_)
        ev = listener_vectors_[0]  # [T, C]
        ev = torch.where(torch.isfinite(ev), ev, torch.zeros_like(ev))

        T = ev.shape[0]
        dev = ev.device

        writer = None
        out_path = os.path.join(path, f"{ind}_3d.avi")

        with torch.no_grad():
            for s in range(0, T, chunk_size):
                e = min(s + chunk_size, T)
                cur = e - s
                chunk = ev[s:e]  # [cur, 58]

                # FaceVerse 系数就地设置
                self.faceverse.batch_size = cur
                self.faceverse.init_coeff_tensors()
                self.faceverse.exp_tensor   = chunk[:, :52].reshape(cur, -1).to(dev)
                self.faceverse.rot_tensor   = chunk[:, 52:55].reshape(cur, -1).to(dev)
                self.faceverse.trans_tensor = chunk[:, 55:].reshape(cur, -1).to(dev)
                self.faceverse.id_tensor    = self.id_tensor.view(1, 150).repeat(cur, 1).reshape(cur, 150).to(dev)

                # 渲染一小段
                pred = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
                imgs = pred['rendered_img']                     # [cur, H, W, 3/4]
                imgs = torch.clamp(imgs, 0, 255).detach().cpu().numpy().astype(np.uint8)
                imgs = imgs[:, :, :, :3]

                # 初始化视频写出（按渲出的原始分辨率）
                if writer is None:
                    h, w = imgs.shape[1], imgs.shape[2]
                    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

                for i in range(imgs.shape[0]):
                    writer.write(imgs[i])
                    if save_frames:
                        cv2.imwrite(os.path.join(path, f"{ind}_3d_{s+i:05d}.png"), imgs[i])

                # 及时释放显存碎片
                del pred, imgs, chunk
                torch.cuda.empty_cache()

        if writer is not None:
            writer.release()
        return out_path

