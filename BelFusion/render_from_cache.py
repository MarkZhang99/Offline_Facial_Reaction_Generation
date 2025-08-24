# render_from_cache.py（开头到 npz 选择这一段替换）
import os, glob, argparse, yaml, numpy as np, torch
from importlib import import_module

def build_conf(cfg_path, dataset_path=None):
    y = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    class Cfg: pass
    c = Cfg()
    # dataset 根：优先 CLI，再 YAML，再 ../dataset/data（你说一直用这个）
    c.dataset_path = dataset_path or y.get("dataset_path") or "../dataset/data"
    # ReactionDataset / get_dataloader 需要的字段
    c.img_size     = y.get("img_size", 224)
    c.crop_size    = y.get("crop_size", 224)
    c.clip_length  = y.get("clip_length", 750)
    c.window_size  = y.get("window_size", 16)
    c.batch_size   = y.get("batch_size", 1)
    c.num_workers  = y.get("num_workers", 0)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="results/train_offline/config.yaml")
    ap.add_argument("--dataset-path", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--cache-dir", type=str, default="results/preds_test")
    ap.add_argument("--outdir", type=str, default="results/render_from_cache")
    ap.add_argument("--offset", type=int, default=0)       # 只渲一小段：从第几条开始
    ap.add_argument("--limit",  type=int, default=None)    # 渲多少条
    ap.add_argument("--listener-only", action="store_true")
    ap.add_argument("--save-seq", action="store_true")
    ap.add_argument("--save-mp4", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    # 选 npz（支持 offset/limit）
    all_npzs = sorted(glob.glob(os.path.join(args.cache_dir, f"{args.split}_b*_i*.npz")))
    start, end = args.offset, (None if args.limit is None else args.offset + args.limit)
    npzs = all_npzs[start:end]
    print(f"[render_from_cache] use npz[{start}:{end}] -> {len(npzs)} files")
    ...

    # 1) 准备 dataloader
    conf = build_conf(args.cfg, args.dataset_path)
    get_dataloader = import_module("dataset").get_dataloader
    dl = get_dataloader(
        conf, args.split,
        load_audio=False,
        load_video_s=True, load_video_l=True,   # 需要 real 帧
        load_emotion_s=False, load_emotion_l=False,
        load_3dmm_s=False,   load_3dmm_l=False,
        load_ref=True                               # 需要 listener reference
    )

    # 2) 渲染器
    Render = import_module("render").Render
    render = Render(enable_pirender=False)


    # 3) 输出目录
    split_root = os.path.join(args.outdir, args.split)
    fid_real = os.path.join(split_root, "fid", "real")
    fid_fake = os.path.join(split_root, "fid", "fake")
    os.makedirs(fid_real, exist_ok=True)
    os.makedirs(fid_fake, exist_ok=True)
    if args.listener_only:
        os.makedirs(os.path.join(split_root, "listener_only_seq"), exist_ok=True)
        os.makedirs(os.path.join(split_root, "listener_only_videos"), exist_ok=True)

    # 4) 缓存 npz
    npzs = sorted(glob.glob(os.path.join(args.cache-dir if hasattr(args,'cache-dir') else args.cache_dir,
                                         f"{args.split}_b*_i*.npz")))
    if args.limit:
        npzs = npzs[:args.limit]
    print(f"[render_from_cache] dataset_path={conf.dataset_path}  split={args.split}")
    print(f"[render_from_cache] npz files={len(npzs)}  outdir={split_root}")

    if not npzs:
        raise SystemExit("❌ 没找到缓存 npz（检查 --cache-dir 是否正确）")

    # 5) 跟 dataloader 对齐地迭代
    it = iter(dl)
    processed = 0
    for batch in it:
        # 解包（按你的 ReactionDataset 返回顺序）：0 spk_vid, 4 lst_vid, 7 ref
        spk_v = batch[0]   # 形状 [B,T,3,224,224] 或 [T,3,224,224]
        lst_v = batch[4]
        ref   = batch[7]
        if spk_v.ndim == 4:  # 兼容 batch_size=1
            spk_v = spk_v.unsqueeze(0); lst_v = lst_v.unsqueeze(0); ref = ref.unsqueeze(0)
        B = spk_v.shape[0]
        for b in range(B):
            if processed >= len(npzs): break
            npz_path = npzs[processed]
            tag = os.path.splitext(os.path.basename(npz_path))[0]
            z = np.load(npz_path, allow_pickle=True)
            if "listener_3dmm" not in z:
                raise ValueError(f"{npz_path} 缺少 listener_3dmm")
            coeff = torch.from_numpy(z["listener_3dmm"]).float().to(render.device)

            # 调用原有渲染函数：它会把 fake/real 帧落到 fid 下
            render.rendering_for_fid(
                path=split_root,
                ind=tag,
                listener_vectors=coeff,
                speaker_video_clip=spk_v[b].to(render.device),
                listener_reference=ref[b].to(render.device),
                listener_video_clip=lst_v[b].to(render.device),
            )

            # 如果只要听者帧，顺手搬到 listener_only_seq / 可选 mp4
            if args.listener_only:
                fake_frames = sorted(glob.glob(os.path.join(fid_fake, f"{tag}_*.png")))
                if fake_frames:
                    import cv2
                    seq_dir = os.path.join(split_root, "listener_only_seq", tag)
                    os.makedirs(seq_dir, exist_ok=True)
                    imgs = []
                    for i, p in enumerate(fake_frames):
                        img = cv2.imread(p); imgs.append(img)
                        if args.save_seq:
                            cv2.imwrite(os.path.join(seq_dir, f"{i:06d}.png"), img)
                    if args.save_mp4 and imgs:
                        h, w = imgs[0].shape[:2]
                        vpath = os.path.join(split_root, "listener_only_videos", f"{tag}.mp4")
                        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                        for img in imgs: vw.write(img)
                        vw.release()

            processed += 1

    print("✅ done ->", split_root)
    print("假帧目录：", fid_fake)
    print("真帧目录：", fid_real)
    print(f"\n可以算 FID：\npython -m pytorch_fid {fid_real} {fid_fake}")

if __name__ == "__main__":
    main()
