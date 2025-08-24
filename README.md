# REACT 2024 Facial Reaction Generation

> **TranVAE (Learning2Listen), BeLFusion (REACT baseline), REGNN (REACT baseline)**

A lightweight, reproducible codebase for conversational facial reaction generation on the **REACT 2024** dataset. We provide scripts/configs for three representative structures and a unified evaluation pipeline.

---

## TL;DR

* **Dataset:** REACT 2024 (NoXI + RECOLA), subject-independent splits
* **FPS:** 25; clips standardized to **750 frames ≈ 30 s**
* **Outputs:**

  * **TranVAE** → 3DMM (expression/pose)
  * **BeLFusion** → 3DMM (expression/pose)
  * **REGNN** → Action Units (AU; GraphAU 25-D)
* **Official metrics:** FRCorr (↑), FRDist (↓), FRVar (↑), FRDvs (↑), TLCC (↓)

---

## Provenance & Targets

* **TranVAE** follows the **TransVAE** configuration from *Learning2Listen*; native output: **3DMM**.
* **BeLFusion** reproduces the **official REACT 2024 baseline** adaptation; native output: **3DMM**.
* **REGNN** reproduces the **official REACT 2024 baseline**; native output: **AU** (GraphAU).

---

## Repo structure

```
.
├── configs/
│   ├── transvae.yaml
│   ├── belfusion_vae.yaml           # stage 1
│   ├── belfusion_ldm.yaml           # stage 2
│   ├── regnn_vggish.yaml
│   └── regnn_wav2vec2.yaml
├── data/                            # your features here (see below)
├── scripts/
│   ├── train_transvae.py
│   ├── train_belfusion.py           # stage 1 & 2 via config
│   ├── train_regnn.py
│   └── evaluate.py
├── results/
└── README.md
```

> If your repo uses different script names/paths, adjust the commands below accordingly.

---

## Environment

* **Python** 3.9
* **PyTorch** 2.0.1 + CUDA 11.8

```bash
conda create -n react24 python=3.9 -y
conda activate react24
pip install -r requirements.txt
```

---

## Data & Features

Follow the REACT 2024 protocol. Expected layout (customize as needed):

```
data/
├── noxi/ ...
├── recola/ ...
└── features/
    ├── graphau/            # 25-D AU embeddings
    ├── faceverse_3dmm/     # 58-D (52 expr + 3 rot + 3 trans)
    ├── vggish/             # 128-D audio
    └── wav2vec2/           # 768-D audio (optional)
```

All streams are aligned to **25 fps** and normalized per session (z-score). We use subject-independent splits with no speaker overlap.

---

## Quickstart

### 1) TranVAE (3DMM)

```bash
python scripts/train_transvae.py \
  --config configs/transvae.yaml \
  --data_root ./data \
  --save_dir ./results/transvae
```

### 2) BeLFusion (3DMM; two-stage)

```bash
# Stage 1: VAE
python scripts/train_belfusion.py \
  --config configs/belfusion_vae.yaml \
  --name All_VAEv2_W50

# Stage 2: LDM (latent diffusion)
python scripts/train_belfusion.py \
  --config configs/belfusion_ldm.yaml \
  --name Belfusion --arch.args.k 10
```

### 3) REGNN (AU)

```bash
# VGGish audio
python scripts/train_regnn.py \
  --config configs/regnn_vggish.yaml \
  --save_dir ./results/regnn_vggish

# Wav2Vec 2.0 audio (optional)
python scripts/train_regnn.py \
  --config configs/regnn_wav2vec2.yaml \
  --save_dir ./results/regnn_w2v2
```

### 4) Evaluation (unified)

```bash
python scripts/evaluate.py \
  --resume ./results/<exp>/best_checkpoint.pth \
  --split test --no-render
```

By default we report **FRCorr / FRDist / FRVar / FRDvs** in the main text; **TLCC** is available in logs (frames & seconds @25 fps).

---

## Results (REACT 2024, test)

| Model            |  FRCorr ↑ |  FRDist ↓ |   FRVar ↑   |   FRDvs ↑   | TLCC ↓ (frames, sec) |
| ---------------- | :-------: | :-------: | :---------: | :---------: | :------------------: |
| **TranVAE**      |   0.159   |   121.06  |   0.01202   | **0.09871** |     47.04 (1.88)     |
| **BeLFusion**    |   0.137   | **75.00** |   0.00685   |   0.00851   |     49.00 (1.96)     |
| **REGNN–VGGish** | **0.196** |   84.12   |   0.00602   |   0.03359   |   **41.46 (1.66)**   |
| **REGNN–W2V2**   |   0.080   |   137.80  | **0.02144** |   0.02151   |     44.29 (1.77)     |

> Sampling budget & seeds are aligned across methods (5 samples / clip; mean over 3 seeds), unless noted.

---

## Reproducibility Notes

* Fixed **10** diffusion steps for BeLFusion as in the official baseline.
* Sliding-window processing (50 frames) for long sequences (750 frames ≈ 30 s).
* Session-level z-score normalization.
* Same sampling budget and seeds across models by default.

---

## Citations

If you use this repo, please cite the baselines and the dataset.

```bibtex
@inproceedings{learning2listen,
  title={Learning to Listen ...},
  author={...},
  booktitle={...},
  year={...}
}
@inproceedings{react2024,
  title={REACT 2024 Challenge ...},
  author={...},
  booktitle={...},
  year={2024}
}
```

---

## License

TBD (MIT/Apache-2.0). Replace with your project’s license.

---

## Acknowledgments

We build on TransVAE from *Learning2Listen* and the official REACT 2024 baselines for BeLFusion and REGNN. Thanks to the dataset contributors (NoXI, RECOLA) and the community.
