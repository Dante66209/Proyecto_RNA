"""
Entrena un modelo CNN con Transfer Learning (MobileNetV3-Small) sobre las imágenes RGB de MSL-ABC,
con separación estricta POR PERSONA (sin fuga de datos):
  train : S1..S19 excepto {S5, S12, S17}  (muestreo configurable --stride para evitar frames redundantes)
  val   : S5, S12, S17
  test  : S18, S20 (evaluación COMPLETA en A, B y C)

Ejecuta el reemplazo de fondo croma verde y los aumentos visuales en GPU por lotes para máxima velocidad.

Uso:
  python lsm/train_static_cnn.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import io as tvio
from torchvision import models
from torchvision.transforms import v2 as T2

VAL_PERSONS = {"S5", "S12", "S17"}


def collect_samples(root: Path, split: str, stride: int = 1):
    items = []
    for sub in ("A", "B", "C"):
        sp_dir = root / f"lsm-abc-{sub}" / split
        if not sp_dir.exists():
            continue
        for cls_dir in sorted(sp_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            files = sorted(cls_dir.glob("*.jpg"))
            for k, p in enumerate(files):
                if k % stride != 0:
                    continue
                m = re.match(r"(S\d+)-", p.name)
                person = m.group(1) if m else ""
                items.append((str(p), cls_dir.name, person, sub))
    return items


class FastImageDataset(Dataset):
    """Solo decodifica JPEG en C++ y redimensiona a 224x224; los aumentos corren en GPU."""
    def __init__(self, items, cls_to_idx):
        self.items = items
        self.cls_to_idx = cls_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab, _, sub = self.items[i]
        img = tvio.decode_image(path, mode=tvio.ImageReadMode.RGB)  # uint8 (3, H, W)
        if img.shape[1] != 224 or img.shape[2] != 224:
            img = F.interpolate(img.unsqueeze(0).float(), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).byte()
        return img, self.cls_to_idx[lab], sub


def gpu_chroma_augment(x_u8: torch.Tensor, chroma_aug: bool, mean: torch.Tensor, std: torch.Tensor, gpu_aug: nn.Module | None) -> torch.Tensor:
    """Convierte (B, 3, H, W) uint8 -> float normalizado en GPU, con reemplazo opcional de croma verde."""
    x = x_u8.float() / 255.0
    if chroma_aug:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        mask = (g > 0.27) & ((g - r) > 0.08) & ((g - b) > 0.07)
        apply = (torch.rand(x.shape[0], 1, 1, device=x.device) < 0.7) & (mask.float().mean(dim=(1, 2), keepdim=True) > 0.05)
        mask = (mask & apply).unsqueeze(1).expand_as(x)
        c1 = torch.rand(x.shape[0], 3, 1, 1, device=x.device) * 0.85 + 0.05
        c2 = torch.rand(x.shape[0], 3, 1, 1, device=x.device) * 0.85 + 0.05
        alpha = torch.linspace(0, 1, x.shape[2], device=x.device).view(1, 1, -1, 1)
        bg = c1 * (1 - alpha) + c2 * alpha
        bg = bg + torch.randn_like(x) * 0.03
        x = torch.where(mask, bg.clamp(0.0, 1.0), x)
    if gpu_aug is not None:
        x = gpu_aug(x)
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/MSL-ABC")
    ap.add_argument("--stride", type=int, default=8, help="tomar 1 de cada N frames en train/val")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--chroma-aug", action="store_true", default=True)
    ap.add_argument("--no-chroma-aug", dest="chroma_aug", action="store_false")
    ap.add_argument("--out", default="outputs/static_cnn")
    a = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(a.root)
    tr_all = collect_samples(root, "train", stride=a.stride)
    te_all = collect_samples(root, "test", stride=1)

    classes = sorted({str(x[1]) for x in tr_all})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    tr_items = [x for x in tr_all if x[2] not in VAL_PERSONS]
    va_items = [x for x in tr_all if x[2] in VAL_PERSONS]
    print(f"Device: {dev} | train: {len(tr_items)} (stride={a.stride}) | val: {len(va_items)} | test: {len(te_all)}", flush=True)

    nw = 6
    dl_tr = DataLoader(FastImageDataset(tr_items, cls_to_idx), batch_size=a.batch, shuffle=True,
                       num_workers=nw, pin_memory=(dev == "cuda"), persistent_workers=True)
    dl_va = DataLoader(FastImageDataset(va_items, cls_to_idx), batch_size=a.batch * 2, shuffle=False,
                       num_workers=nw, pin_memory=(dev == "cuda"), persistent_workers=True)
    dl_te = DataLoader(FastImageDataset(te_all, cls_to_idx), batch_size=a.batch * 2, shuffle=False,
                       num_workers=nw, pin_memory=(dev == "cuda"))

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    gpu_aug = T2.Compose([
        T2.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.9, 1.1)),
        T2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    ]).to(dev)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(dev)
    print(f"MobileNetV3-Small params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl_tr))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_ep, bad, t0 = -1.0, 0, 0, time.time()
    ckpt = out / "mobilenetv3_best.pt"

    def run_eval(loader):
        model.eval()
        yt, yp, subs, total_loss, n = [], [], [], 0.0, 0
        with torch.no_grad():
            for xb, yb, sb in loader:
                xb = gpu_chroma_augment(xb.to(dev, non_blocking=True), False, mean, std, None)
                yb = yb.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    lo = model(xb)
                    loss = lossf(lo, yb)
                total_loss += loss.item() * len(yb)
                n += len(yb)
                yt.append(yb.cpu().numpy())
                yp.append(lo.argmax(1).cpu().numpy())
                subs.extend(sb)
        return total_loss / n, np.concatenate(yt), np.concatenate(yp), np.array(subs)

    for ep in range(1, a.epochs + 1):
        model.train()
        tl, tc, ns = 0.0, 0, 0
        for xb, yb, _ in dl_tr:
            xb = gpu_chroma_augment(xb.to(dev, non_blocking=True), a.chroma_aug, mean, std, gpu_aug)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                lo = model(xb)
                loss = lossf(lo, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(yb)
            tc += (lo.argmax(1) == yb).sum().item()
            ns += len(yb)

        vl, yt_v, yp_v, _ = run_eval(dl_va)
        vacc = float((yt_v == yp_v).mean())
        vf1 = float(f1_score(yt_v, yp_v, average="macro"))
        hist["train_loss"].append(tl / ns)
        hist["train_acc"].append(tc / ns)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        hist["val_f1"].append(vf1)
        print(f"ep {ep:2d} | train loss {tl/ns:.4f} acc {tc/ns:.4f} | val loss {vl:.4f} acc {vacc:.4f} f1 {vf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save({"state": model.state_dict(), "classes": classes}, ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping en época {ep} (mejor: {best_ep})", flush=True)
                break

    ck = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(ck["state"])
    tl_, yt, yp, subs_te = run_eval(dl_te)
    acc = float((yt == yp).mean())
    f1m = float(f1_score(yt, yp, average="macro"))
    rep = classification_report(yt, yp, target_names=classes, digits=4)
    rep_d = classification_report(yt, yp, target_names=classes, output_dict=True)
    cm = confusion_matrix(yt, yp)
    by_sub = {}
    for s in ("A", "B", "C"):
        ms = subs_te == s
        by_sub[s] = {"acc": float((yt[ms] == yp[ms]).mean()), "f1": float(f1_score(yt[ms], yp[ms], average="macro"))}
    print(f"\n=== TEST MobileNetV3-Small (S18, S20 no vistas) ===")
    print(f"Acc: {acc:.4f} | Macro-F1: {f1m:.4f} | A={by_sub['A']['acc']:.4f} B={by_sub['B']['acc']:.4f} C={by_sub['C']['acc']:.4f}")
    print(rep)
    (out / "reporte_cnn.txt").write_text(rep, encoding="utf-8")
    np.savetxt(out / "matriz_confusion_cnn.csv", cm, fmt="%d", delimiter=",")
    json.dump({
        "model": "mobilenet_v3_small",
        "params": sum(p.numel() for p in model.parameters()),
        "chroma_aug": a.chroma_aug,
        "stride": a.stride,
        "test_accuracy": acc,
        "test_precision_macro": float(precision_score(yt, yp, average="macro")),
        "test_recall_macro": float(recall_score(yt, yp, average="macro")),
        "test_f1_macro": f1m,
        "by_subset": by_sub,
        "val_f1_macro": best_f1,
        "best_epoch": best_ep,
        "training_seconds": time.time() - t0,
        "per_class_f1": {c: float(rep_d[c]["f1-score"]) for c in classes},
        "history": hist,
    }, open(out / "resultados.json", "w", encoding="utf-8"), indent=2)
    plot_cnn_results(out, classes, hist, best_ep, cm, acc)


def plot_cnn_results(out: Path, classes: list[str], hist: dict, best_ep: int, cm: np.ndarray, acc: float):
    """Genera las curvas entrenamiento/validación y la matriz de confusión normalizada de MobileNetV3-Small."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(hist["train_loss"], label="Entrenamiento")
    ax[0].plot(hist["val_loss"], "--", label="Validación")
    ax[0].axvline(best_ep - 1, color="gray", ls=":", label="Mejor ep.")
    ax[0].set_title("MobileNetV3-Small — Pérdida")
    ax[0].set_xlabel("Época")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    ax[1].plot(hist["train_acc"], label="Entrenamiento")
    ax[1].plot(hist["val_acc"], "--", label="Validación")
    ax[1].axvline(best_ep - 1, color="gray", ls=":", label="Mejor ep.")
    ax[1].set_title("MobileNetV3-Small — Exactitud")
    ax[1].set_xlabel("Época")
    ax[1].legend()
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "curvas_entrenamiento.png", dpi=150)
    plt.close(fig)

    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión normalizada — MobileNetV3-Small (Acc={acc:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cmn[i, j]
            if val >= 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.55 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion_cnn.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
