# SiPaKMeD Binary-Classifier (PyTorch + Docker)

Train several CNN backbones on the **SiPaKMeD** cervical-cell dataset with a
single command.

---

## 1. Requirements (host)

* Docker ≥ 20.10
* NVIDIA driver  (for GPU training)  
* NVIDIA Container Toolkit (enables `docker run --gpus`)  
  *(skip the last two bullets if you will run on CPU only)*

---

## 2. Build the image (one-time)

```bash
docker build -t sipak-trainer .
```

---

## 3. Run

### GPU machine (recommended)

```bash
mkdir -p data metrics                     # persists across runs

docker run --rm --gpus all \
  -v $PWD/data:/workspace/data \
  -v $PWD/metrics:/workspace/metrics \
  --shm-size=1g \
  sipak-trainer
```

#### CPU-only fallback

```bash
docker run --rm \
  -e CUDA_VISIBLE_DEVICES= \              # hides GPUs
  -v $PWD/data:/workspace/data \
  -v $PWD/metrics:/workspace/metrics \
  sipak-trainer
```

---

## 4. Outputs

```
metrics/
└── timestamp/
    ├── epoch_logs.csv # per-epoch metrics (train / val)
    ├── summary.csv # best epoch per backbone
    └── checkpoints/
        └── efficientnet_b0_best.pt
        └── ...
```

*The first run downloads and unpacks **SiPaKMeD** automatically; subsequent runs
reuse the cached copy in `./data`.*
