import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from dataset import ReactionDataset
from model import TransformerVAE
from utils import AverageMeter
try:
    # 懒加载/可选：渲染模块缺失也不阻塞评估
    from render import Render
except Exception:
    Render = None
from model.losses import VAELoss
from metric import *
from dataset import get_dataloader
from utils import load_config
import model as module_arch
import model.losses as module_loss
from functools import partial


FV_MEAN = np.load('external/FaceVerse/mean_face.npy').astype(np.float32).reshape(1, 58)

FV_STD  = np.load('external/FaceVerse/std_face.npy').astype(np.float32).reshape(1, 58)

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="../dataset/data", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--threads', default=8, type=int, help="num max of threads")
    parser.add_argument('--binarize', action='store_true', help='binarize AUs output from model')
    parser.add_argument('--no-render', action='store_true',
                        help='disable rendering (skip PIRender/FaceVerse)')
    parser.add_argument('--limit', type=int, default=0,
                        help='use only the first N samples for evaluation')
    parser.add_argument('--save-preds-dir', type=str, default='',
                        help='folder to save per-sample predictions (.npz)')
    parser.add_argument('--local-neighbors', action='store_true',
                    help='Build neighbors within the subset (no global matrix).')
    parser.add_argument('--save-metrics-cache', type=str, default='',
                    help='npz path to save {pred, gt} for metrics-only pass')
    parser.add_argument('--skip-neighbor-metrics', action='store_true',
                    help='skip neighbor-based metrics (FRC/FRD/FRVar)')


    args = parser.parse_args()
    return args
# ===== 放到文件顶部 import 后、val() 前 =====
def _build_local_neighbors(gt_t, k):
    import numpy as np
    gt = gt_t.detach().cpu().numpy()          # [N,T,C]
    N = gt.shape[0]; k = max(1, min(k, N-1))
    X = gt.mean(axis=1)                        # [N,C]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    idx_topk = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx_topk, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx_topk, order, axis=1)   # [N,k]

def compute_FRC_localKNN(pred_t, gt_t, k=10, p=0.5, split="val"):
    """
    用子集内部 KNN 做邻居，适配仓库的 metric/FRC._func：
    - k_neighbour_matrix: 需要 [N] 的 0/1 掩码
    - k_pred:             需要 [k, T, C]
    """
    import numpy as np, inspect
    from metric import FRC as FRCmod
    _frc_inner = getattr(FRCmod, "_func")

    # 1) 子集内部邻居索引 [N, k]
    neigh = _build_local_neighbors(gt_t, k=k)

    # 2) 准备数组，保证是 [N, T, C]
    em    = gt_t.detach().cpu().numpy()
    preds = pred_t.detach().cpu().numpy()
    if preds.ndim == 2: preds = preds[..., None]
    if em.ndim    == 2: em    = em[..., None]

    N = preds.shape[0]

    # 3) 自适应签名（有的版本没有 p/val_test）
    import inspect
    sig = inspect.signature(_frc_inner).parameters
    base_kwargs = {}
    if "em" in sig:       base_kwargs["em"] = em
    if "p" in sig:        base_kwargs["p"] = p
    if "val_test" in sig: base_kwargs["val_test"] = split

    # 4) 逐样本：邻居索引 -> 0/1 掩码；预测复制到 [k, T, C]
    vals = []
    for i in range(N):
        idx = neigh[i]                             # [k]
        k_i = idx.shape[0]
        mask = np.zeros((N,), dtype=np.int8)
        mask[idx] = 1                              # [N] 0/1

        k_pred = np.repeat(preds[i][None, ...], k_i, axis=0)  # [k, T, C]
        vals.append(_frc_inner(mask, k_pred, **base_kwargs))

    return float(np.mean(vals))

def compute_neighbor_metric_localKNN(mod_name, pred_t, gt_t, k=10, split="val", **extra):
    """
    用子集内部 KNN 近邻计算邻居型指标（FRC/FRD/FRVar/FRDvs），
    适配仓库 metric.<mod_name> 里的 `_func` 签名。
    """
    import numpy as np, inspect
    from importlib import import_module
    M = import_module(f"metric.{mod_name}")
    _inner = getattr(M, "_func")

    neigh = _build_local_neighbors(gt_t, k=k)     # [N,k]
    em    = gt_t.detach().cpu().numpy()           # [N,T,C]
    pred  = pred_t.detach().cpu().numpy()
    if pred.ndim == 2: pred = pred[..., None]
    if em.ndim   == 2: em   = em[..., None]

    # 自适配 _func 的参数
    sig  = inspect.signature(_inner).parameters
    base = {}
    if "em" in sig:       base["em"] = em
    if "val_test" in sig: base["val_test"] = split
    # 只传函数真的接受的 kwargs（比如 FRC 的 p）
    base.update({k:v for k,v in extra.items() if k in sig})

    vals = []
    N = pred.shape[0]
    for i in range(N):
        mask = np.zeros((N,), dtype=np.int8)  # [N] 0/1 掩码
        mask[neigh[i]] = 1
        k_pred = np.repeat(pred[i][None, ...], neigh.shape[1], axis=0)  # [k,T,C]
        vals.append(_inner(mask, k_pred, **base))
    return float(np.mean(vals))

def metrics_from_cache(cache_path, local_neighbors=False, k=10, p=0.5, split="val"):
    
    z = np.load(cache_path, allow_pickle=True)
    pred_t = torch.from_numpy(z["pred"])
    gt_t   = torch.from_numpy(z["gt"])
    #if local_neighbors:
        #frc = compute_FRC_localKNN(pred_t, gt_t, k=k, p=p, split=split)
    #else:
        # 若你后来有了全量邻居矩阵，可以调回官方的 compute_FRC_mp
        #frc = compute_FRC_mp(args_like, pred_t, gt_t, val_test=split, p=p)
    return compute_FRC_localKNN(pred_t, gt_t, k=k, p=p, split=split)


# Evaluating
def val(args, model, val_loader, criterion, render, binarize=False):

    T = 750

    losses = AverageMeter(); rec_losses = AverageMeter(); kld_losses = AverageMeter()
    model.eval()

    out_dir = os.path.join(args.outdir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # 用于“先存再算指标”
    all_pred = []   # listener emotion pred
    all_gt   = []   # listener emotion GT
    all_sp = []

    # 可选：逐样本落盘（3DMM/情感）
    if args.save_preds_dir:
        os.makedirs(args.save_preds_dir, exist_ok=True)

    pbar = tqdm(val_loader)
    for batch_idx, (speaker_video_clip, speaker_audio_clip, speaker_emotion, _,
                    listener_video_clip, _, listener_emotion, listener_3dmm, listener_references) in enumerate(pbar):

        # ------ to cuda & 统一裁 T ------
        if torch.cuda.is_available():
            if isinstance(speaker_video_clip, torch.Tensor) and speaker_video_clip.ndim >= 2:
                speaker_video_clip = speaker_video_clip[:, :T].cuda()
            if isinstance(speaker_audio_clip, torch.Tensor) and speaker_audio_clip.ndim >= 2:
                speaker_audio_clip = speaker_audio_clip[:, :T].cuda()
            else:
                speaker_audio_clip = None
            speaker_emotion   = speaker_emotion[:, :T].cuda()
            listener_emotion  = listener_emotion[:, :T].cuda()
            listener_3dmm     = listener_3dmm[:, :T].cuda()
            listener_references = listener_references.cuda()

        with torch.no_grad():
            prediction = model(speaker_video=speaker_video_clip,
                               speaker_audio=speaker_audio_clip,
                               speaker_emotion=speaker_emotion,
                               listener_emotion=listener_emotion)

            if isinstance(prediction, list):  # Trans-VAE
                listener_3dmm_out, listener_emotion_out, distribution = prediction
                loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm,
                                                     listener_emotion_out, listener_3dmm_out, distribution)
                losses.update(loss.item(), speaker_video_clip.size(0))
                rec_losses.update(rec_loss.item(), speaker_video_clip.size(0))
                kld_losses.update(kld_loss.item(), speaker_video_clip.size(0))
            else:  # BeLFusion
                listener_3dmm_out    = prediction["3dmm_coeff"]
                listener_emotion_out = prediction["prediction"]
                loss = criterion(**prediction)["loss"].item()
                losses.update(loss)

        # ------ 先攒指标需要的数据 ------
        all_pred.append(listener_emotion_out.detach().cpu())   # [B,T,C]
        all_gt.append(listener_emotion.detach().cpu())         # [B,T,C]
        all_sp.append(speaker_emotion.detach().cpu())

        # 可选：逐样本保存预测（3DMM + 情感）
        if args.save_preds_dir:
            B = listener_3dmm_out.shape[0]
            for bs in range(B):
                C_pred = listener_3dmm_out[bs].detach().cpu().numpy()   # [T,58] 预测（标准化空间）
                C_gt   = listener_3dmm[bs].detach().cpu().numpy()       # [T,58] GT（标准化空间）

                # 只加 mean（不要乘 std）
                C_pred_denorm = C_pred + FV_MEAN
                C_gt_denorm   = C_gt   + FV_MEAN

                rec = {
                    "listener_3dmm":    C_pred_denorm,   # 渲染用这个
                    "listener_3dmm_gt": C_gt_denorm,
                    "speaker_emotion":  speaker_emotion[bs].detach().cpu().numpy(),
                    "listener_emotion": listener_emotion[bs].detach().cpu().numpy(),
                    "meta": {"split": args.split, "batch": int(batch_idx), "index_in_batch": int(bs)},
                }
                np.savez_compressed(
                    os.path.join(args.save_preds_dir, f"{args.split}_b{batch_idx:05d}_i{bs:02d}.npz"),
                    **rec
                )


        # 可选渲染（你现在 --no-render 就不会走这里）
        if binarize:
            listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
        if (batch_idx % 25) == 0 and (not args.no_render) and (render is not None) and hasattr(render, 'rendering_for_fid'):
            B = listener_emotion_out.size(0)
            for bs in range(B):
                render.rendering_for_fid(out_dir,
                    f"{args.split}_b{batch_idx+1}_ind{bs+1}",
                    listener_3dmm_out[bs], speaker_video_clip[bs],
                    listener_references[bs], listener_video_clip[bs,:T])

    # ------ 把指标输入缓存落盘（一次性）------
    pred_t = torch.cat(all_pred, dim=0)   # [N,T,C]
    gt_t   = torch.cat(all_gt,   dim=0)   # [N,T,C]
    sp_t   = torch.cat(all_sp,   dim=0)
    if args.save_metrics_cache:
        cache_path = args.save_metrics_cache
        cache_dir  = os.path.dirname(cache_path)
        if cache_dir:  # 避免空字符串
            os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, pred=pred_t.numpy(), gt=gt_t.numpy(), sp=sp_t.numpy(), split=args.split)
        print(f"[cache] saved metrics arrays -> {cache_path}")

    # ------ 现在（同一进程）要不要算指标？ ------
    if args.skip_neighbor_metrics:
        FRC = FRD = FRDvs = FRVar = smse = TLCC = 0.0
    else:
        # 你想要的本地邻居 FRC（不依赖全量矩阵）
        k = 10
        if args.local_neighbors:
            print(f"[metric] FRC with local KNN on subset (k={k})")
            FRC = compute_FRC_localKNN(pred_t, gt_t, k=k, p=0.5, split=args.split)
            FRC   = compute_neighbor_metric_localKNN("FRC",   pred_t, gt_t, k=k, split=args.split, p=0.5)
            FRD   = compute_neighbor_metric_localKNN("FRD",   pred_t, gt_t, k=k, split=args.split)
            FRVar = compute_neighbor_metric_localKNN("FRVar", pred_t, gt_t, k=k, split=args.split)
            # 如果仓库里有 FRDvs.py
            # FRDvs = compute_neighbor_metric_localKNN("FRDvs", pred_t, gt_t, k=k, split=args.split)

        else:
            # （如果将来你拿到了官方邻居矩阵，再切回原函数）
            from metric.FRC import compute_FRC_mp
            FRC = compute_FRC_mp(args, pred_t, gt_t, val_test=args.split, p=0.5)
        # 其它指标（如果有）在这里继续；没有的话给 0
        

    return losses.avg, rec_losses.avg, kld_losses.avg, FRC, FRD, FRDvs, FRVar, smse, TLCC




def main(args):
    checkpoint_path = args.resume
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if not os.path.exists(config_path): # args-based loading --> Trans-VAE by default
        val_loader = get_dataloader(args, args.split, load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_ref=True)
        model = TransformerVAE(img_size = args.img_size, audio_dim = args.audio_dim, output_emotion_dim = args.emotion_dim, output_3dmm_dim = args._3dmm_dim, feature_dim = args.feature_dim, seq_len = args.max_seq_len, online = args.online, window_size = args.window_size, device = args.device)
        criterion = VAELoss(args.kl_p)
    else: # config-based loading --> BeLFusion
        cfg = load_config(config_path)
        dataset_cfg = cfg.validation_dataset if args.split == "val" else cfg.test_dataset
        dataset_cfg.dataset_path = args.dataset_path
        #val_loader = get_dataloader(dataset_cfg, args.split, load_audio=False, load_video_s=True, load_video_l=True, load_emotion_s=True,
        #                                            load_emotion_l=True, load_3dmm_s=False, load_3dmm_l=True, load_ref=True)
        need_video_s = True
        need_video_l = not args.no_render
        val_loader = get_dataloader(dataset_cfg, args.split,
                                    load_audio=False,
                                    load_video_s=need_video_s, load_video_l=need_video_l,
                                    load_emotion_s=True, load_emotion_l=True,
                                    load_3dmm_s=False, load_3dmm_l=True,
                                    )  # 先单线程，确保不再卡死
        
        from torch.utils.data import DataLoader, Subset
        base_ds = val_loader.dataset    # 原 loader 的 dataset

        # 可选：只跑前 N 条（--limit）
        if getattr(args, "limit", 0):
            n = min(args.limit, len(base_ds))
            base_ds = Subset(base_ds, list(range(n)))
            print(f"[limit] using first {n} samples")

        # 重新构建 DataLoader（这下 batch/worker 真生效了）
        val_loader = DataLoader(
            base_ds,
            batch_size=args.batch_size,         # <- 你想要的
            shuffle=False,
            num_workers=args.num_workers,       # <- 你想要的
            pin_memory=False,
            persistent_workers=False
        )
        print(f"[dataloader] batch_size={args.batch_size} num_workers={args.num_workers} "
            f"speaker_video={need_video_s} listener_video={need_video_l} "
            f"samples={len(base_ds)}")
        model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
        criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    if args.resume != '': #  resume from a checkpoint
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)



    if torch.cuda.is_available():
        model = model.cuda()
    # 渲染：按需启动；禁用或导入失败时设为 None
    if args.no_render or (Render is None):
        render = None
    else:
        render = Render('cuda' if torch.cuda.is_available() else 'cpu')

    val_loss, rec_loss, kld_loss, FRC, FRD, FRDvs, FRVar, smse, TLCC = val(args, model, val_loader, criterion, render, binarize=args.binarize)
    print("{}_loss: {:.5f}   {}_rec_loss: {:.5f}  {}_kld_loss: {:.5f} ".format(args.split, val_loss, args.split, rec_loss, args.split, kld_loss))
    print("Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
    print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))


# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)

