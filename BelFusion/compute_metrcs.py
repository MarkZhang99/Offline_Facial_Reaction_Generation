import numpy as np, torch
from importlib import import_module

def expand_k(pred_t, k=10):
    p = pred_t.detach().cpu().numpy()
    if p.ndim == 2: p = p[..., None]        # [N,T] -> [N,T,1]
    pk = np.repeat(p[:, None, ...], k, axis=1)  # [N,1,T,C] -> [N,k,T,C]
    return torch.from_numpy(pk)

def _build_local_neighbors(gt_t, k):
    gt = gt_t.detach().cpu().numpy()
    if gt.ndim == 2: gt = gt[..., None]
    X = gt.mean(axis=1)                                     # [N,C]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    k = max(1, min(k, gt.shape[0]-1))
    idx_topk = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx_topk, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx_topk, order, axis=1)      # [N,k]

def compute_neighbor_metric_localKNN(mod_name, pred_t, gt_t, k=10, split="val", **extra):
    """
    通用：FRC/FRD 走 _func(mask, k_pred, ...)，
         FRVar/FRDvs/S_MSE 走 compute_*([N,k,T,C])。
    """
    M = import_module(f"metric.{mod_name}")

    # 情况 A：模块有 _func（FRC/FRD）
    if hasattr(M, "_func"):
        inner = getattr(M, "_func")
        # 准备 em/p/val_test 这些可选参数
        import inspect
        sig  = inspect.signature(inner).parameters
        base = {}
        if "em" in sig:       base["em"] = gt_t.detach().cpu().numpy()
        if "val_test" in sig: base["val_test"] = split
        for kname, v in extra.items():
            if kname in sig: base[kname] = v

        # 本地 KNN -> 掩码；预测复制到 [k,T,C]
        neigh = _build_local_neighbors(gt_t, k)             # [N,k]
        pred  = pred_t.detach().cpu().numpy()
        if pred.ndim == 2: pred = pred[..., None]
        N, kN = pred.shape[0], neigh.shape[1]
        vals = []
        for i in range(N):
            mask = np.zeros((N,), dtype=np.int8)
            mask[neigh[i]] = 1
            k_pred = np.repeat(pred[i][None, ...], kN, axis=0)  # [k,T,C]
            vals.append(inner(mask, k_pred, **base))
        return float(np.mean(vals))

    # 情况 B：没有 _func，就找 compute_<ModName>（FRVar/FRDvs/S_MSE）
    fn_name = f"compute_{mod_name}"
    if hasattr(M, fn_name):
        fn = getattr(M, fn_name)
        pred_k = expand_k(pred_t, k)                         # [N,k,T,C]
        return float(fn(pred_k))

    # 实在没有就报错
    raise AttributeError(f"metric.{mod_name} has neither _func nor {fn_name}()")
import numpy as np, torch
from metric.TLCC import compute_TLCC_mp
from metric.S_MSE import compute_s_mse
# 读缓存
z    = np.load('results/cache/test_limit32.npz', allow_pickle=True)
pred = torch.from_numpy(z['pred'])   # [N,T,C]
gt   = torch.from_numpy(z['gt'])     # [N,T,C]
sp   = torch.from_numpy(z['sp'])     # [N,T,C]  <-- 确保你在前向阶段把 speaker 也存了

k = 10

print("FRC(local-KNN)  =", compute_neighbor_metric_localKNN("FRC",  pred, gt, k=k, split='test', p=0.5))
print("FRD(local-KNN)  =", compute_neighbor_metric_localKNN("FRD",  pred, gt, k=k, split='test'))
print("FRVar           =", compute_neighbor_metric_localKNN("FRVar",pred, gt, k=k, split='test'))
print("FRDvs           =", compute_neighbor_metric_localKNN("FRDvs",pred, gt, k=k, split='test'))
SMSE = float(compute_s_mse(expand_k(pred, k)))  # expand_k 输出 [N,k,T,C]
print("S-MSE           =", SMSE)

# TLCC 需要 [N,k,T,C] 和 speaker
from compute_metrcs import expand_k  # 或者把 expand_k 放本文件顶部
pred_k = expand_k(pred, k)           # [N,k,T,C]
print("TLCC            =", float(compute_TLCC_mp(pred_k, sp, p=8)))
