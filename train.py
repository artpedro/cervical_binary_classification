# misc
from typing import Optional, Tuple, OrderedDict

# Data management 
import pathlib
import shutil
import os, random, csv, warnings, datetime
import kagglehub
import zipfile
import urllib.request as urllib_request
import itertools, json, math, time, copy, textwrap
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image

# Models
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as T
import torchvision.models as tvm
import timm

# Metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import gc
from pathlib import Path
from collections import defaultdict

# Reproducibility / device
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Folders path
DATA_DIR    = Path(os.getenv("DATA_DIR",    "/workspace/data"))
METRICS_DIR = Path(os.getenv("METRICS_DIR", "/workspace/metrics"))
RUNS_DIR    = Path(os.getenv("RUNS_DIR",    "/workspace/runs"))

# Models to use
TODO = {
    "SqueezeNet 1.1":"tv_squeezenet1_1",
    "MobileNet V2 1.0x":"mobilenetv2_100",
    "ShuffleNet V2 1.0x":"tv_shufflenet_v2_x1_0",
    "GhostNet 1.0x":"ghostnet_100",
    **{f"EfficientNet-B{i}":f"efficientnet_b{i}" for i in range(9)}
}

# Dataset class
class SipakBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = Image.open(self.df.path[idx]).convert("RGB")
        return self.tf(img), int(self.df.binary_idx[idx])

# Class labeling
NORMAL   = {"Superficial-Intermediate","Parabasal"}
ABNORMAL = {"Koilocytotic","Dyskeratotic","Metaplastic"}

# Transforms on dataset
train_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.RandomRotation(180), T.RandomHorizontalFlip(0.5),
    T.ColorJitter(0.3, 0.5, 0.3, 0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Adapt model for binary classification
def _adapt_head(model: nn.Module, num_classes: int = 2) -> int:
    """
    Replace the model's final classification layer(s) with a fresh head that
    outputs `num_classes` logits.

    Parameters
    ----------
    model : nn.Module
        A CNN backbone (timm, torchvision, or hub-based).
    num_classes : int, default=2
        Desired number of output classes.

    Returns
    -------
    int
        The incoming feature dimension of the new head (i.e. `in_features`).

    Raises
    ------
    RuntimeError
        If no Linear or Conv2d head could be located.
    """
    # --- Fast-path for timm models ----------------------------------------
    if hasattr(model, "reset_classifier"):
        old_head = model.get_classifier()
        in_feats = old_head.in_features if hasattr(old_head, "in_features") \
                else old_head.in_channels
        model.reset_classifier(num_classes)
        return in_feats

    # --- TorchVision or custom models -------------------------------------
    for attr in ("classifier", "fc", "_fc"):
        if not hasattr(model, attr):
            continue

        head = getattr(model, attr)

        # Simple Linear head (e.g. ResNet, DenseNet) -----------------------
        if isinstance(head, nn.Linear):
            in_feats = head.in_features
            setattr(model, attr, nn.Linear(in_feats, num_classes))
            return in_feats

        # Sequential head (e.g. AlexNet, VGG) -----------------------------
        if isinstance(head, nn.Sequential):
            layers = list(head.children())

            # Walk backwards to find the first Linear or Conv2d layer
            for idx in range(len(layers) - 1, -1, -1):
                layer = layers[idx]

                if isinstance(layer, nn.Linear):
                    in_feats = layer.in_features
                    layers[idx] = nn.Linear(in_feats, num_classes)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_feats

                if isinstance(layer, nn.Conv2d):
                    in_ch = layer.in_channels
                    layers[idx] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
                    setattr(model, attr, nn.Sequential(*layers))
                    return in_ch

    raise RuntimeError("Could not find a Linear/Conv2d classification head.")

# Load any of the models expected
def load_any(name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Load a backbone by name from timm, torchvision or PyTorch Hub and adapt it
    for `num_classes` outputs.
    """
    # timm
    try:
        model = timm.create_model(name, pretrained=pretrained)
        origin = f"timm:{name}"
    except (ValueError, RuntimeError):
        model, origin = None, None

    # torchvision
    if model is None:
        tv_registry = {
            "tv_squeezenet1_1":      tvm.squeezenet1_1,
            "tv_shufflenet_v2_x1_0": tvm.shufflenet_v2_x1_0,
            "tv_mobilenet_v2":       tvm.mobilenet_v2,
            "mobilenetv2_100":       tvm.mobilenet_v2,
            **{f"efficientnet_b{i}": getattr(tvm, f"efficientnet_b{i}")
               for i in range(9)}
        }
        tv_ctor = tv_registry.get(name)
        if tv_ctor:
            weights = (tv_ctor.Weights.DEFAULT             # torchvision ≥0.15
                       if pretrained and hasattr(tv_ctor, "Weights")
                       else None)
            model = tv_ctor(weights=weights)
            origin = f"torchvision:{name}"

    # PyTorch
    if model is None and name.startswith("ghostnet"):
        model = torch.hub.load("pytorch/vision",
                               "ghostnet_1x",
                               pretrained=pretrained)
        origin = "hub:ghostnet"

    if model is None:
        raise ValueError(f"Unknown backbone: {name}")

    in_features = _adapt_head(model, num_classes)
    return model, in_features, origin

# Metrics

EPS = 1e-9

def _confusion_parts(y_true, y_pred) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp

def metrics_binary(y_true, y_pred) -> dict[str, float]:
    """Return all requested binary-classification metrics in one dict."""
    tn, fp, fn, tp = _confusion_parts(y_true, y_pred)

    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred),           # PPV
        "rec":  recall_score(y_true, y_pred),              # Sensitivity
        "spec": tn / (tn + fp + EPS),
        "f1":   f1_score(y_true, y_pred),
        "ppv":  tp / (tp + fp + EPS),                      # same as precision
        "npv":  tn / (tn + fn + EPS),
    }
    
# Epoch runner
def _run_epoch(dataloader: DataLoader,
              model: nn.Module,
              criterion: nn.Module,
              optimiser: Optional[torch.optim.Optimizer] = None
              ) -> Tuple[float, float, float, float, float]:
    """
    Execute a single epoch.
    - If optimiser is provided -> training mode, gradients ON.
    - If optimiser is None     -> evaluation mode, gradients OFF.
    
    Returns
    -------
    (avg_loss, accuracy, f1, recall, specificity)
    """
    training = optimiser is not None
    model.train(training)

    total_loss, preds, trues = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * labels.size(0)
            preds.append(logits.detach().argmax(1).cpu())
            trues.append(labels.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(trues)
    n      = len(dataloader.dataset)

    metrics = metrics_binary(y_true, y_pred)
    metrics["loss"] = total_loss / n
    return metrics
    
def main():
    # Download dataset from Kaggle
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading dataset...")
    slug = "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
    raw_path = kagglehub.dataset_download(slug)  # downloads + unzips under ~/.cache/kagglehub/…
    print("Downloaded to :", raw_path)

    # Move/rename
    final_path = DATA_DIR / "sipakmed"
    
    if final_path.exists():
        shutil.rmtree(final_path)
    shutil.copytree(raw_path, final_path)
    print("Dataset ready at:", final_path)

    DATA_ROOT = DATA_DIR / "sipakmed"  
    assert DATA_ROOT.exists(), f"Missing folder: {DATA_ROOT}"

    # Crawl: one class = one top-level im_* folder
    records = []      # (path, label) rows
    cls_counts = defaultdict(int)
    for class_dir in sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()]):
        label = class_dir.name.replace("im_", "")
        imgs = list(class_dir.glob("**/CROPPED/*.bmp"))
        cls_counts[label] = len(imgs)
        records += [(str(p), label) for p in imgs]

    df_counts = pd.Series(cls_counts, name="#images").sort_index().to_frame()
    print(f"Total images found: {len(records)}")
    
    print("Preparing for training...")
    
    # Check device
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(" GPU used:", torch.cuda.get_device_name())
    else:
        print(" CPU mode - CUDA not visible")
        
    # Create a timestamped run-folder and the CSV logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir   = METRICS_DIR / timestamp
    checkpt_dir = run_dir / "checkpoints"
    checkpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "epoch_logs.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["model", "origin", "epoch", "split",
        "loss", "acc", "prec", "rec", "spec", "f1", "ppv", "npv",
        "lr", "seconds"]
        )
    log_file.flush()

    # Scan the SiPaKMeD dataset and build a fold-annotated DataFrame
    dataset_root = DATA_DIR / "sipakmed"
    
    if not dataset_root.exists():
        raise FileNotFoundError(f"{dataset_root} not found - did the download succeed?")

    records: list[tuple[str, str]] = []          # [(image_path, full_label), …]
    class_counts: dict[str, int] = defaultdict(int)

    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        full_label = class_dir.name.replace("im_", "")
        image_paths = class_dir.glob("**/CROPPED/*.bmp")
        for img_path in image_paths:
            records.append((str(img_path), full_label))
        class_counts[full_label] += len(list(image_paths))

    df = pd.DataFrame(records, columns=["path", "label_full"])
    
    df["binary_label"] = df["label_full"].apply(
        lambda lbl: "normal" if lbl in NORMAL else "abnormal"
    )
    df["binary_idx"] = df["binary_label"].map({"normal": 0, "abnormal": 1})

    # Assign fold indices (5-fold stratified split, seed = 42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["binary_idx"])):
        df.loc[val_idx, "fold"] = fold_idx
        
    # Build a class-weighted cross-entropy loss
    
    class_freq = df["binary_idx"].value_counts(normalize=True).sort_index()
    
    # Guard against missing classes
    if class_freq.size != 2:
        raise ValueError(f"Both classes required, found: {class_freq.to_dict()}")

    weights = torch.tensor([1 / class_freq[0], 1 / class_freq[1]],
                        dtype=torch.float32,
                        device=DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights)

    # train loop
    history_rows: list[OrderedDict] = []        # one row = one (epoch, split)
    summary_rows: list[list]        = []        # one row per backbone
    summary_path = run_dir / "summary.csv"

    checkpt_dir = run_dir / "checkpoints"       # run_dir created earlier
    checkpt_dir.mkdir(parents=True, exist_ok=True)

    for friendly_name, backbone_id in TODO.items():
        print(f"\n> {friendly_name}")
        try:
            # DataLoaders
            train_df = df[df.fold != 0]
            val_df   = df[df.fold == 0]

            train_loader = DataLoader(
                SipakBinaryDataset(train_df, train_tf),
                batch_size=32,
                shuffle=True,
                num_workers=int(os.getenv("NUM_WORKERS", 2)),
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                SipakBinaryDataset(val_df, val_tf),
                batch_size=32,
                shuffle=False,
                num_workers=int(os.getenv("NUM_WORKERS", 2)),
                pin_memory=torch.cuda.is_available(),
            )

            # Model / optimiser / scheduler
            model, _, origin = load_any(backbone_id,
                                        num_classes=2,
                                        pretrained=True)
            model.to(DEVICE)

            optimiser = torch.optim.SGD(model.parameters(),
                                        lr=5e-4,
                                        momentum=0.9,
                                        weight_decay=5e-3)
            
            scheduler = MultiStepLR(optimiser, milestones=[10, 20], gamma=0.1)

            best_val = {"epoch": 0, "acc": 0.0, "prec": 0.0, "rec": 0.0,
                        "spec": 0.0, "f1": 0.0, "ppv": 0.0, "npv": 0.0}

            # Epoch loop
            for epoch in range(1, 26):
                t0 = time.time()

                train_m = _run_epoch(train_loader, model, criterion, optimiser)
                val_m   = _run_epoch(val_loader,   model, criterion)

                scheduler.step()
                lr_now   = scheduler.get_last_lr()[0]
                duration = time.time() - t0

                # store results in history_rows
                for split, m in [("train", train_m), ("val", val_m)]:
                    history_rows.append(OrderedDict(
                        model   = friendly_name,
                        origin  = origin,
                        epoch   = epoch,
                        split   = split,
                        loss    = m["loss"],
                        acc     = m["acc"],
                        prec    = m["prec"],
                        rec     = m["rec"],       # sensitivity
                        spec    = m["spec"],
                        f1      = m["f1"],
                        ppv     = m["ppv"],
                        npv     = m["npv"],
                        lr      = lr_now,
                        seconds = duration,
                    ))
                    log_writer.writerow([friendly_name,
                        origin,
                        epoch,
                        split,
                        m["loss"],
                        m["acc"],
                        m["prec"],
                        m["rec"],       # sensitivity
                        m["spec"],
                        m["f1"],
                        m["ppv"],
                        m["npv"],
                        lr_now,
                        duration])
                    log_file.flush()
                # checkpoint best by validation accuracy
                if val_m["acc"] > best_val["acc"]:
                    best_val.update(epoch = epoch,
                                    acc   = val_m["acc"],
                                    prec  = val_m["prec"],
                                    rec   = val_m["rec"],
                                    spec  = val_m["spec"],
                                    f1    = val_m["f1"],
                                    ppv   = val_m["ppv"],
                                    npv   = val_m["npv"])
                    torch.save(model.state_dict(),
                            checkpt_dir / f"{backbone_id}_best.pt")

                # console progress every 5 epochs
                if epoch == 1 or epoch % 5 == 0 or epoch == 25:
                    print(f"Ep{epoch:02d} "
                        f"loss {val_m['loss']:.4f} "
                        f"acc {val_m['acc']:.3f} "
                        f"prec {val_m['prec']:.3f} "
                        f"rec {val_m['rec']:.3f} "
                        f"spec {val_m['spec']:.3f} "
                        f"f1 {val_m['f1']:.3f} "
                        f"ppv {val_m['ppv']:.3f} "
                        f"npv {val_m['npv']:.3f}"
                        f"last epoch duration {duration:.1f}s ")
            
            row_summary = [
                friendly_name, best_val["epoch"], best_val["acc"],
                best_val["prec"], best_val["rec"], best_val["spec"],
                best_val["f1"],   best_val["ppv"], best_val["npv"],
                ]        
                
            summary_rows.append(row_summary)
            pd.DataFrame([row_summary]).to_csv(
                summary_path,
                index=False,
                mode="a",
                header=not summary_path.exists(),    # write header once!
                columns=["model", "best_epoch", "best_acc", "best_prec",
                "best_rec", "best_spec", "best_f1",
                "best_ppv", "best_npv"]
            )
            
        except Exception as exc:
            warnings.warn(f"{friendly_name} failed: {exc}")
            summary_rows.append([friendly_name] + [None] * 8)

        # hygiene: free GPU memory
        del model, optimiser, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSVs
    pd.DataFrame(history_rows).to_csv(
        run_dir / "epoch_logs.csv",
        index=False,
        columns=["model", "origin", "epoch", "split",
                "loss", "acc", "prec", "rec", "spec",
                "f1", "ppv", "npv", "lr", "seconds"]
    )

    pd.DataFrame(summary_rows,
                columns=["model", "best_epoch", "best_acc", "best_prec",
                        "best_rec", "best_spec", "best_f1",
                        "best_ppv", "best_npv"]
                ).to_csv(run_dir / "summary.csv", index=False)

if __name__ == "__main__":
    main()