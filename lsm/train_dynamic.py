"""
Entrena y compara dos arquitecturas temporales (RNN) para las 6 señas dinámicas del alfabeto LSM
(Zenodo 14689869: J, K, Ñ, Q, X, Z), con separación estricta POR PERSONA:
  train : personas de train excepto {S5, S13, S17}
  val   : S5, S13, S17  -> early stopping por Macro-F1
  test  : S20, S21      -> personas nunca vistas (vistas frontal y perfil)

Modelos comparados:
  1. LSTM-Base         : LSTM unidireccional sobre secuencia cruda (63 coords + posición muñeca).
  2. DynamicBiGRUAttn  : Bi-GRU de 2 capas con Atención Temporal + Pooling sobre 214 features
                         espaciotemporales (forma 3D de la mano + trayectoria, velocidad, aceleración
                         de muñeca/puntas y velocidad angular de la palma) + aumento 3D/temporal.

Uso:
  python lsm/train_dynamic.py
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsm.features import FEATURE_DIM, PALM_ANCHORS, PALM_ORIENT_SLICE, compute_hand_features

VAL_PERSONS = {"S5", "S13", "S17"}
SEQ_LEN = 32
DYN_FEAT_DIM = FEATURE_DIM + 30  # 184 forma + 3 pos + 3 vel + 3 acel + 15 vel_puntas + 6 vel_orient = 214


def resample_clip(seq: np.ndarray, det: np.ndarray, target_len: int = SEQ_LEN) -> tuple[np.ndarray, float]:
    """Recorta al rango con mano detectada, interpola huecos breves y remuestrea a target_len cuadros."""
    if len(seq) == 0 or det.sum() < 5:
        return np.zeros((target_len, 21, 3), dtype=np.float32), 0.0
    idx = np.where(det)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    sub = seq[i0 : i1 + 1].copy()
    sub_det = det[i0 : i1 + 1]
    t_all = np.arange(len(sub))
    if not sub_det.all():
        t_ok = t_all[sub_det]
        for j in range(21):
            for c in range(3):
                sub[:, j, c] = np.interp(t_all, t_ok, sub[sub_det, j, c])
    t_new = np.linspace(0, len(sub) - 1, target_len)
    out = np.empty((target_len, 21, 3), dtype=np.float32)
    for j in range(21):
        for c in range(3):
            out[:, j, c] = np.interp(t_new, t_all, sub[:, j, c])
    return out, float(det.mean())


def normalize_clip_np(seq: np.ndarray) -> np.ndarray:
    """Centra la trayectoria en el centroide de la muñeca y escala por el tamaño mediano de la palma."""
    local = seq - seq[:, 0:1, :]
    palm_scales = np.linalg.norm(local[:, PALM_ANCHORS, :], axis=-1).mean(axis=-1)
    med = float(np.median(palm_scales))
    scale = med if med > 1e-6 else 1.0
    w_mean = seq[:, 0:1, :].mean(axis=0, keepdims=True)
    return ((seq - w_mean) / scale).astype(np.float32)


def augment_dynamic_3d(
    seq: torch.Tensor,
    max_yaw_deg: float = 20.0,
    max_pitch_deg: float = 15.0,
    max_roll_deg: float = 15.0,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Aumento espaciotemporal: rotación 3D coherente en todo el clip, estiramiento temporal y ruido."""
    b, T, _, _ = seq.shape
    dev, dt = seq.device, seq.dtype

    # 1. Estiramiento/recorte temporal suave (simula firmar más rápido o lento)
    out = seq.clone()
    for k in range(b):
        if torch.rand(1, device=dev).item() < 0.7:
            frac = 0.80 + 0.20 * torch.rand(1, device=dev).item()
            win = max(8, int(round(T * frac)))
            start = int(torch.randint(0, T - win + 1, (1,), device=dev).item())
            crop = seq[k, start : start + win].permute(1, 2, 0).reshape(63, 1, win)  # (63,1,win)
            res = F.interpolate(crop.unsqueeze(0), size=(1, T), mode="bilinear", align_corners=True)
            out[k] = res.squeeze(0).squeeze(1).permute(1, 0).reshape(T, 21, 3)

    # 2. Rotación 3D uniforme en toda la secuencia
    rx = (torch.rand(b, device=dev, dtype=dt) * 2 - 1) * np.deg2rad(max_pitch_deg)
    ry = (torch.rand(b, device=dev, dtype=dt) * 2 - 1) * np.deg2rad(max_yaw_deg)
    rz = (torch.rand(b, device=dev, dtype=dt) * 2 - 1) * np.deg2rad(max_roll_deg)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    z0, o1 = torch.zeros_like(cx), torch.ones_like(cx)
    Rx = torch.stack([o1, z0, z0, z0, cx, -sx, z0, sx, cx], dim=-1).view(b, 3, 3)
    Ry = torch.stack([cy, z0, sy, z0, o1, z0, -sy, z0, cy], dim=-1).view(b, 3, 3)
    Rz = torch.stack([cz, -sz, z0, sz, cz, z0, z0, z0, o1], dim=-1).view(b, 3, 3)
    R = (Rz @ Ry @ Rx).unsqueeze(1)  # (B, 1, 3, 3)

    out = torch.matmul(out, R.transpose(-1, -2))
    scale = 1.0 + (torch.rand(b, 1, 1, 3, device=dev, dtype=dt) * 2 - 1) * 0.08
    out = out * scale + torch.randn_like(out) * noise_std
    return out


def compute_dynamic_features(seq: torch.Tensor) -> torch.Tensor:
    """Convierte secuencia (B, T, 21, 3) normalizada en (B, T, 214)."""
    b, T, _, _ = seq.shape
    shape = compute_hand_features(seq.reshape(b * T, 21, 3)).reshape(b, T, FEATURE_DIM)
    wrist = seq[:, :, 0, :]  # (B, T, 3)
    w_vel = F.pad(wrist[:, 1:] - wrist[:, :-1], (0, 0, 1, 0))
    w_acc = F.pad(w_vel[:, 1:] - w_vel[:, :-1], (0, 0, 1, 0))
    tips = seq[:, :, [4, 8, 12, 16, 20], :].reshape(b, T, 15)
    t_vel = F.pad(tips[:, 1:] - tips[:, :-1], (0, 0, 1, 0))
    orient = shape[:, :, PALM_ORIENT_SLICE]
    o_vel = F.pad(orient[:, 1:] - orient[:, :-1], (0, 0, 1, 0))
    return torch.cat([shape, wrist, w_vel, w_acc, t_vel, o_vel], dim=-1)


class LSTMBase(nn.Module):
    """Baseline temporal: LSTM simple sobre coordenadas crudas (63D) por cuadro."""
    def __init__(self, n_cls: int, hidden: int = 128, p: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(63, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(hidden, n_cls))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.reshape(seq.shape[0], seq.shape[1], 63)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class DynamicBiGRUAttn(nn.Module):
    """Bi-GRU bidireccional de 2 capas con Atención Temporal + MaxPool sobre 214 features."""
    def __init__(self, n_cls: int, d_in: int = DYN_FEAT_DIM, d_model: int = 160, hidden: int = 128, p: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(p * 0.5),
        )
        self.gru = nn.GRU(d_model, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        d_bi = hidden * 2
        self.attn = nn.Sequential(nn.Linear(d_bi, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_bi * 2),
            nn.Dropout(p),
            nn.Linear(d_bi * 2, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, n_cls),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = compute_dynamic_features(seq)
        h = self.proj(x)
        h, _ = self.gru(h)  # (B, T, 2*hidden)
        w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        attn_pool = (h * w).sum(dim=1)
        max_pool = h.max(dim=1).values
        return self.head(torch.cat([attn_pool, max_pool], dim=-1))


def train_model(name, model, Xtr, ytr, Xva, yva, classes, out_dir: Path, epochs, batch_size, lr, patience, aug, ls):
    dev = Xtr.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    steps = epochs * ((len(Xtr) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=ls)
    ckpt = out_dir / f"{name}_best.pt"

    @torch.no_grad()
    def eval_s(X, y):
        model.eval()
        lo = model(X)
        return lossf(lo, y).item(), lo.argmax(1)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_ep, bad, t0 = -1.0, 0, 0, time.time()
    print(f"\n=== Entrenando {name} ({sum(p.numel() for p in model.parameters()):,} params) ===", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tl, tc, ns = 0.0, 0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            xb, yb = Xtr[b], ytr[b]
            if aug:
                xb = augment_dynamic_3d(xb)
            opt.zero_grad()
            lo = model(xb)
            loss = lossf(lo, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tc += (lo.argmax(1) == yb).sum().item()
            ns += len(b)

        vl, vp = eval_s(Xva, yva)
        vf1 = float(f1_score(yva.cpu().numpy(), vp.cpu().numpy(), average="macro"))
        vacc = float((vp == yva).float().mean().item())
        hist["train_loss"].append(tl / ns)
        hist["train_acc"].append(tc / ns)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        hist["val_f1"].append(vf1)
        if ep == 1 or ep % 10 == 0 or vf1 > best_f1:
            print(f"  ep {ep:3d} | loss {tl/ns:.4f} acc {tc/ns:.4f} | val loss {vl:.4f} acc {vacc:.4f} f1 {vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save({"state": model.state_dict(), "classes": classes, "seq_len": SEQ_LEN}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stopping en época {ep} (mejor: {best_ep}, val_f1={best_f1:.4f})")
                break

    ck = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(ck["state"])
    return model, hist, best_f1, best_ep, time.time() - t0


def load_dynamic_data(npz_path: str, parquet_dir: str = "data/hf_lsm_dataset/dynamic"):
    """Carga las secuencias dinámicas desde `.npz` local o con fallback a Parquet ZSTD."""
    p = Path(npz_path)
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return d["seqs"], d["detected"], d["splits"], d["persons"], d["labels"], d["views"]

    pq_root = Path(parquet_dir)
    if not pq_root.exists():
        raise FileNotFoundError(f"No se encontró {p} ni {pq_root}")

    import pyarrow.parquet as pq

    seqs, dets, splits, persons, labels, views = [], [], [], [], [], []
    for sp_name in ("train", "validation", "test"):
        tbl = pq.read_table(pq_root / f"{sp_name}.parquet")
        sp_tag = "test" if sp_name == "test" else "train"
        for s_flat, d_arr, per, lab, vw in zip(
            tbl["sequence"].to_numpy(zero_copy_only=False),
            tbl["detected"].to_numpy(zero_copy_only=False),
            tbl["person"].to_numpy(zero_copy_only=False).astype(str),
            tbl["label"].to_numpy(zero_copy_only=False).astype(str),
            tbl["view"].to_numpy(zero_copy_only=False).astype(str),
        ):
            seqs.append(np.asarray(s_flat, dtype=np.float32).reshape(-1, 21, 3))
            dets.append(np.asarray(d_arr, dtype=bool))
            splits.append(sp_tag)
            persons.append(per)
            labels.append(lab)
            views.append(vw)
    return (
        np.array(seqs, dtype=object),
        np.array(dets, dtype=object),
        np.array(splits),
        np.array(persons),
        np.array(labels),
        np.array(views),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/landmarks/dynamic_lsm.npz")
    ap.add_argument("--parquet-dir", default="data/hf_lsm_dataset/dynamic")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="outputs/dynamic")
    a = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    seqs, dets, splits, persons, labels, views = load_dynamic_data(a.data, a.parquet_dir)

    X_list, det_rates, keep = [], [], []
    for i in range(len(seqs)):
        r, rate = resample_clip(seqs[i], dets[i], SEQ_LEN)
        if rate > 0 and r.any():
            X_list.append(normalize_clip_np(r))
            det_rates.append(rate)
            keep.append(i)
    keep = np.array(keep)
    X_all = np.stack(X_list)
    splits, persons, labels, views = splits[keep], persons[keep], labels[keep], views[keep]

    classes = sorted({str(c) for c in labels})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_idx[v] for v in labels], dtype=np.int64)

    is_tr_split = splits == "train"
    is_val = is_tr_split & np.isin(persons, list(VAL_PERSONS))
    is_tr = is_tr_split & (~is_val)
    is_te = splits == "test"

    print(f"Clips válidos: {len(X_all)}/{len(seqs)} | Clases ({len(classes)}): {classes}")
    print(f"train: {is_tr.sum()} | val: {is_val.sum()} ({sorted(set(persons[is_val]))}) | test: {is_te.sum()} ({sorted(set(persons[is_te]))})")

    to_t = lambda arr: torch.tensor(arr, dtype=torch.float32, device=dev)
    to_y = lambda arr: torch.tensor(arr, dtype=torch.long, device=dev)
    Xtr_t, ytr_t = to_t(X_all[is_tr]), to_y(y_idx[is_tr])
    Xva_t, yva_t = to_t(X_all[is_val]), to_y(y_idx[is_val])
    Xte_t, yte_t = to_t(X_all[is_te]), to_y(y_idx[is_te])
    views_te = views[is_te]

    lstm_base = LSTMBase(len(classes)).to(dev)
    lstm_base, h_lstm, bf1_l, bep_l, t_l = train_model(
        "lstm_base", lstm_base, Xtr_t, ytr_t, Xva_t, yva_t, classes, out,
        epochs=a.epochs, batch_size=a.batch, lr=2e-3, patience=a.patience, aug=False, ls=0.0,
    )

    bigru = DynamicBiGRUAttn(len(classes)).to(dev)
    bigru, h_gru, bf1_g, bep_g, t_g = train_model(
        "bigru_attn", bigru, Xtr_t, ytr_t, Xva_t, yva_t, classes, out,
        epochs=a.epochs, batch_size=a.batch, lr=2e-3, patience=a.patience, aug=True, ls=0.05,
    )

    def eval_full(name, model, hist, bf1_val, bep, secs):
        model.eval()
        with torch.no_grad():
            yp = model(Xte_t).argmax(1).cpu().numpy()
        yt = yte_t.cpu().numpy()
        rep = classification_report(yt, yp, target_names=classes, digits=4)
        rep_d = classification_report(yt, yp, target_names=classes, output_dict=True)
        cm = confusion_matrix(yt, yp)
        acc = float((yt == yp).mean())
        f1m = float(f1_score(yt, yp, average="macro"))
        by_view = {}
        for vw in sorted(set(views_te)):
            mv = views_te == vw
            by_view[vw] = {"acc": float((yt[mv] == yp[mv]).mean()), "f1": float(f1_score(yt[mv], yp[mv], average="macro"))}
        print(f"\n=== TEST {name} (S20, S21 no vistas) ===")
        print(f"Acc: {acc:.4f} | Macro-F1: {f1m:.4f} | Vistas: {by_view}")
        print(rep)
        (out / f"reporte_{name}.txt").write_text(rep, encoding="utf-8")
        np.savetxt(out / f"matriz_confusion_{name}.csv", cm, fmt="%d", delimiter=",")
        return {
            "model": name,
            "params": sum(p.numel() for p in model.parameters()),
            "test_accuracy": acc,
            "test_precision_macro": float(precision_score(yt, yp, average="macro")),
            "test_recall_macro": float(recall_score(yt, yp, average="macro")),
            "test_f1_macro": f1m,
            "by_view": by_view,
            "val_f1_macro": bf1_val,
            "best_epoch": bep,
            "training_seconds": secs,
            "per_class_f1": {c: float(rep_d[c]["f1-score"]) for c in classes},
            "history": hist,
        }, cm

    r_l, _ = eval_full("lstm_base", lstm_base, h_lstm, bf1_l, bep_l, t_l)
    r_g, cm_g = eval_full("bigru_attn", bigru, h_gru, bf1_g, bep_g, t_g)

    with open(out / "resumen_metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "params", "test_acc", "f1_macro", "val_f1_macro", "mejor_epoca", "segundos"])
        for r in (r_l, r_g):
            w.writerow([r["model"], r["params"], f"{r['test_accuracy']:.4f}", f"{r['test_f1_macro']:.4f}",
                        f"{r['val_f1_macro']:.4f}", r["best_epoch"], f"{r['training_seconds']:.1f}"])
    json.dump({"classes": classes, "lstm_base": r_l, "bigru_attn": r_g},
              open(out / "resultados.json", "w", encoding="utf-8"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for col, (t, r) in enumerate([("LSTM-Base", r_l), ("DynamicBiGRUAttn", r_g)]):
        h = r["history"]
        ax[0, col].plot(h["train_loss"], label="Entrenamiento"); ax[0, col].plot(h["val_loss"], "--", label="Validación")
        ax[0, col].set_title(f"{t} — Pérdida"); ax[0, col].legend(); ax[0, col].grid(alpha=0.25)
        ax[1, col].plot(h["train_acc"], label="Entrenamiento"); ax[1, col].plot(h["val_acc"], "--", label="Validación")
        ax[1, col].set_title(f"{t} — Exactitud"); ax[1, col].legend(); ax[1, col].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "curvas_comparativas.png", dpi=150); plt.close(fig)

    cmn = cm_g / cm_g.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión dinámica — BiGRU+Attn (Acc={r_g['test_accuracy']:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_g[i,j]}\n({cmn[i,j]:.0%})", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "matriz_confusion_bigru.png", dpi=150); plt.close(fig)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
