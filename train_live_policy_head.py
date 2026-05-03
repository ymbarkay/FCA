import argparse
import copy
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FeatureDataset(Dataset):
    def __init__(self, feature_file: str, y_file: str, num_angle_classes: int = 17):
        self.X = np.load(feature_file, mmap_mode="r")
        self.y = np.load(y_file).astype(np.float32)
        self.num_angle_classes = int(num_angle_classes)

        if len(self.X) != len(self.y):
            raise ValueError(f"Feature/label length mismatch: {len(self.X)} vs {len(self.y)}")

        if self.y.ndim != 2 or self.y.shape[1] < 2:
            raise ValueError("y file must be shape (N, 2+) with columns [angle_norm, speed]")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        angle_norm = float(np.clip(self.y[idx, 0], 0.0, 1.0))
        speed = float(np.clip(self.y[idx, 1], 0.0, 1.0))

        angle_class = int(round(angle_norm * (self.num_angle_classes - 1)))
        angle_class = max(0, min(self.num_angle_classes - 1, angle_class))

        return (
            x,
            torch.tensor(angle_class, dtype=torch.long),
            torch.tensor([speed], dtype=torch.float32),
            torch.tensor([angle_norm], dtype=torch.float32),
        )


class LivePolicyHead(nn.Module):
    def __init__(self, feature_dim=512, hidden1=256, hidden2=128, num_angle_classes=17, dropout=0.10):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.num_angle_classes = int(num_angle_classes)

        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

        self.angle_head = nn.Linear(hidden2, num_angle_classes)
        self.speed_head = nn.Linear(hidden2, 1)

    def forward(self, x):
        h = self.net(x)
        return self.angle_head(h), self.speed_head(h)


@torch.no_grad()
def angle_expected_value(angle_logits: torch.Tensor, num_angle_classes: int) -> torch.Tensor:
    probs = torch.softmax(angle_logits, dim=-1)
    values = torch.linspace(0.0, 1.0, num_angle_classes, device=angle_logits.device)
    return probs @ values


def build_indices(n_samples: int, val_ratio: float, split_mode: str, chunk_size: int, seed: int):
    idx = np.arange(n_samples)
    rng = np.random.default_rng(seed)

    if split_mode == "random":
        rng.shuffle(idx)
        n_val = max(1, int(round(n_samples * val_ratio)))
        val_idx = np.sort(idx[:n_val])
        train_idx = np.sort(idx[n_val:])
        return train_idx, val_idx

    # chunked split reduces temporal leakage from near-duplicate adjacent frames
    n_chunks = int(math.ceil(n_samples / chunk_size))
    chunk_ids = np.arange(n_chunks)
    rng.shuffle(chunk_ids)

    n_val_chunks = max(1, int(round(n_chunks * val_ratio)))
    val_chunks = set(chunk_ids[:n_val_chunks].tolist())

    train_list = []
    val_list = []
    for i in range(n_samples):
        c = i // chunk_size
        if c in val_chunks:
            val_list.append(i)
        else:
            train_list.append(i)

    train_idx = np.array(train_list, dtype=np.int64)
    val_idx = np.array(val_list, dtype=np.int64)
    return train_idx, val_idx


def build_speed_sampler(ds: FeatureDataset, train_idx: np.ndarray):
    speeds = ds.y[train_idx, 1]
    pos = float(np.sum(speeds >= 0.5))
    neg = float(np.sum(speeds < 0.5))

    if pos < 1 or neg < 1:
        return None

    w_pos = 0.5 / pos
    w_neg = 0.5 / neg

    weights = np.where(speeds >= 0.5, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(weights), replacement=True)


@dataclass
class Metrics:
    loss: float
    angle_mse: float
    angle_mae: float
    speed_acc: float
    speed_precision: float
    speed_recall: float
    speed_f1: float


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_angle_classes: int,
    ce_weight: float,
    angle_reg_weight: float,
    speed_weight: float,
    speed_pos_weight: torch.Tensor,
):
    model.eval()

    total_loss = 0.0
    total_angle_mse = 0.0
    total_angle_mae = 0.0
    total_n = 0

    tp = 0.0
    fp = 0.0
    fn = 0.0
    correct = 0.0

    with torch.no_grad():
        for x, angle_class, speed, angle_norm in loader:
            x = x.to(device)
            angle_class = angle_class.to(device)
            speed = speed.to(device)
            angle_norm = angle_norm.to(device)

            angle_logits, speed_logit = model(x)

            angle_ce = F.cross_entropy(angle_logits, angle_class, label_smoothing=0.05)
            pred_angle_norm = angle_expected_value(angle_logits, num_angle_classes).unsqueeze(1)
            angle_reg = F.mse_loss(pred_angle_norm, angle_norm)
            speed_bce = F.binary_cross_entropy_with_logits(speed_logit, speed, pos_weight=speed_pos_weight)
            loss = ce_weight * angle_ce + angle_reg_weight * angle_reg + speed_weight * speed_bce

            pred_speed = (torch.sigmoid(speed_logit) >= 0.5).float()

            batch_n = x.size(0)
            total_loss += loss.item() * batch_n
            total_angle_mse += torch.sum((pred_angle_norm - angle_norm) ** 2).item()
            total_angle_mae += torch.sum(torch.abs(pred_angle_norm - angle_norm)).item()
            correct += torch.sum(pred_speed == speed).item()
            total_n += batch_n

            tp += torch.sum((pred_speed == 1) & (speed == 1)).item()
            fp += torch.sum((pred_speed == 1) & (speed == 0)).item()
            fn += torch.sum((pred_speed == 0) & (speed == 1)).item()

    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)

    return Metrics(
        loss=total_loss / max(1, total_n),
        angle_mse=total_angle_mse / max(1, total_n),
        angle_mae=total_angle_mae / max(1, total_n),
        speed_acc=correct / max(1, total_n),
        speed_precision=precision,
        speed_recall=recall,
        speed_f1=f1,
    )


def main():
    parser = argparse.ArgumentParser(description="Train LivePolicyHead on precomputed 512-d features")
    parser.add_argument("--feature-file", default="X_features_512.npy")
    parser.add_argument("--y-file", default="y_train.npy")
    parser.add_argument("--out", default="live_policy_head_512.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-angle-classes", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", choices=["chunked", "random"], default="chunked")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--use-balanced-sampler", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=2.0)
    parser.add_argument("--angle-reg-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ds = FeatureDataset(args.feature_file, args.y_file, num_angle_classes=args.num_angle_classes)

    train_idx, val_idx = build_indices(
        n_samples=len(ds),
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    train_subset = Subset(ds, train_idx.tolist())
    val_subset = Subset(ds, val_idx.tolist())

    sampler = build_speed_sampler(ds, train_idx) if args.use_balanced_sampler else None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )

    model = LivePolicyHead(num_angle_classes=args.num_angle_classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        div_factor=20.0,
        final_div_factor=100.0,
    )

    train_speeds = ds.y[train_idx, 1]
    pos = float(np.sum(train_speeds >= 0.5))
    neg = float(np.sum(train_speeds < 0.5))
    if pos < 1:
        pos_weight_value = 1.0
    else:
        pos_weight_value = max(1.0, neg / pos)
    speed_pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in args.device))

    best_val = float("inf")
    best_state = None
    no_improve = 0

    print(
        f"samples={len(ds)} train={len(train_idx)} val={len(val_idx)} "
        f"split={args.split_mode} chunk={args.chunk_size} pos_weight={pos_weight_value:.3f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, angle_class, speed, angle_norm in train_loader:
            x = x.to(args.device)
            angle_class = angle_class.to(args.device)
            speed = speed.to(args.device)
            angle_norm = angle_norm.to(args.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=("cuda" in args.device)):
                angle_logits, speed_logit = model(x)

                angle_ce = F.cross_entropy(angle_logits, angle_class, label_smoothing=0.05)
                pred_angle_norm = angle_expected_value(angle_logits, args.num_angle_classes).unsqueeze(1)
                angle_reg = F.mse_loss(pred_angle_norm, angle_norm)
                speed_bce = F.binary_cross_entropy_with_logits(speed_logit, speed, pos_weight=speed_pos_weight)
                loss = args.ce_weight * angle_ce + args.angle_reg_weight * angle_reg + args.speed_weight * speed_bce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        train_loss = train_loss_sum / max(1, train_n)

        val = evaluate(
            model=model,
            loader=val_loader,
            device=args.device,
            num_angle_classes=args.num_angle_classes,
            ce_weight=args.ce_weight,
            angle_reg_weight=args.angle_reg_weight,
            speed_weight=args.speed_weight,
            speed_pos_weight=speed_pos_weight,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:03d} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | val_loss={val.loss:.4f} | "
            f"val_angle_mse={val.angle_mse:.6f} | val_angle_mae={val.angle_mae:.6f} | "
            f"val_speed_acc={val.speed_acc:.4f} | val_speed_f1={val.speed_f1:.4f}"
        )

        if val.loss < best_val:
            best_val = val.loss
            no_improve = 0

            best_state = {
                "model_state": copy.deepcopy(model.state_dict()),
                "feature_dim": 512,
                "num_angle_classes": args.num_angle_classes,
                "architecture": "LivePolicyHead-512-256-128",
                "split_mode": args.split_mode,
                "chunk_size": args.chunk_size,
                "val": {
                    "loss": val.loss,
                    "angle_mse": val.angle_mse,
                    "angle_mae": val.angle_mae,
                    "speed_acc": val.speed_acc,
                    "speed_precision": val.speed_precision,
                    "speed_recall": val.speed_recall,
                    "speed_f1": val.speed_f1,
                },
                "train_config": vars(args),
            }
            torch.save(best_state, args.out)
            print("  saved best")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print("Done. Best saved to:", args.out)


if __name__ == "__main__":
    main()
