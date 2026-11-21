"""
Compara dos modelos sobre landmarks de MediaPipe para el alfabeto estático LSM (MSL-ABC, 21 letras):
  1. MLP-Base      : ANN simple sobre las 60 coordenadas 3D normalizadas respecto a la muñeca.
  2. HandGeoResNet : Arquitectura residual con ingeniería geométrica 3D (184 features: vista +
                     marco canónico de palma + ángulos de flexión + apertura + distancias clave)
                     y aumento 3D (yaw/pitch/roll) + label smoothing.

Protocolo estricto SIN fuga de datos (separación POR PERSONA):
  train : S1..S19 excepto {S5, S12, S17}
  val   : S5, S12, S17  -> early stopping por Macro-F1
  test  : S18, S20      -> evaluación final por subconjunto (A, B, C) y global

Uso:
  python lsm/train_static.py
  python lsm/train_static.py --epochs 80 --out outputs/static
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsm.features import FEATURE_DIM, augment_hand_3d, compute_hand_features, pick_single_hand_np, renormalize_3d

VAL_PERSONS = {"S5", "S12", "S17"}


def load_static_npz(split_name: str, root: str = "data/landmarks"):
    Xs, ys, gs, subs, nhs = [], [], [], [], []
    for sub in ("A", "B", "C"):
        p = Path(root) / f"abc_{sub}_{split_name}.npz"
        d = np.load(p)
        Xs.append(d["X"])
        ys.append(d["y"])
        gs.append(d["groups"])
        nhs.append(d["n_hands"])
        subs.append(np.full(len(d["y"]), sub))
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(gs),
        np.concatenate(subs),
        np.concatenate(nhs),
    )


def load_static_data(landmarks_dir: str = "data/landmarks", parquet_dir: str = "data/hf_lsm_dataset/static"):
    """Carga landmarks estáticos desde `.npz` locales o con fallback desde Parquet ZSTD."""
    if (Path(landmarks_dir) / "abc_A_train.npz").exists():
        Xtr_raw, ytr_all, gtr_all, sub_tr, _ = load_static_npz("train", root=landmarks_dir)
        Xte_raw, yte_all, gte_all, sub_te, _ = load_static_npz("test", root=landmarks_dir)
        pts_tr_all, valid_tr = pick_single_hand_np(Xtr_raw)
        pts_te_all, valid_te = pick_single_hand_np(Xte_raw)
        return pts_tr_all, valid_tr, ytr_all, gtr_all, sub_tr, pts_te_all, valid_te, yte_all, gte_all, sub_te

    pq_root = Path(parquet_dir)
    if not pq_root.exists():
        raise FileNotFoundError(f"No se encontraron .npz en {landmarks_dir} ni Parquet en {pq_root}")

    import pyarrow.parquet as pq

    def _read_tbls(names: tuple[str, ...]):
        pts_l, det_l, y_l, g_l, s_l = [], [], [], [], []
        for nm in names:
            tbl = pq.read_table(pq_root / f"{nm}.parquet")
            pts_l.append(np.vstack(tbl["landmarks"].to_numpy(zero_copy_only=False)).reshape(-1, 21, 3).astype(np.float32))
            det_l.append(tbl["detected"].to_numpy(zero_copy_only=False).astype(bool))
            y_l.append(tbl["label"].to_numpy(zero_copy_only=False).astype(str))
            g_l.append(tbl["person"].to_numpy(zero_copy_only=False).astype(str))
            s_l.append(tbl["subset"].to_numpy(zero_copy_only=False).astype(str))
        return (
            np.concatenate(pts_l),
            np.concatenate(det_l),
            np.concatenate(y_l),
            np.concatenate(g_l),
            np.concatenate(s_l),
        )

    pts_tr_all, valid_tr, ytr_all, gtr_all, sub_tr = _read_tbls(("train", "validation"))
    pts_te_all, valid_te, yte_all, gte_all, sub_te = _read_tbls(("test",))
    return pts_tr_all, valid_tr, ytr_all, gtr_all, sub_tr, pts_te_all, valid_te, yte_all, gte_all, sub_te


class MLPBase(nn.Module):
    """ANN simple sobre 60 coordenadas 3D normalizadas (sin ingeniería geométrica ni bloques residuales)."""
    def __init__(self, n_cls: int, hidden=(256, 128), p: float = 0.25):
        super().__init__()
        layers, d = [], 60
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = renormalize_3d(pts)[:, 1:, :].reshape(pts.shape[0], 60)
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, dim: int, p: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(dim, dim),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HandGeoResNet(nn.Module):
    """Red residual regularizada sobre 184 features geométricas 3D."""
    def __init__(self, n_cls: int, width: int = 256, n_blocks: int = 2, p: float = 0.32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.BatchNorm1d(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(width, p=p) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(p * 0.5), nn.Linear(width, n_cls))

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        feats = compute_hand_features(pts)
        h = self.stem(feats)
        h = self.blocks(h)
        return self.head(h)


def train_one(
    name: str,
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    classes: list[str],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    use_aug3d: bool,
    label_smoothing: float,
    weight_decay: float = 2e-4,
    cosine: bool = False,
):
    dev = Xtr.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(Xtr) + batch_size - 1) // batch_size
    if cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=lr * 0.01)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ckpt_path = out_dir / f"{name}_best.pt"

    @torch.no_grad()
    def eval_split(X: torch.Tensor, y: torch.Tensor):
        model.eval()
        logits_all = []
        for i in range(0, len(X), 2048):
            logits_all.append(model(X[i : i + 2048]))
        logits = torch.cat(logits_all, dim=0)
        loss = lossf(logits, y).item()
        pred = logits.argmax(1)
        return loss, pred

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_ep, bad, t0 = -1.0, 0, 0, time.time()

    print(f"\n=== Entrenando {name} ({sum(p.numel() for p in model.parameters()):,} params) ===", flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tl, tc, n_seen = 0.0, 0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            xb, yb = Xtr[b], ytr[b]
            if use_aug3d:
                xb = augment_hand_3d(xb)
            opt.zero_grad()
            lo = model(xb)
            loss = lossf(lo, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tc += (lo.argmax(1) == yb).sum().item()
            n_seen += len(b)

        vl, vp = eval_split(Xva, yva)
        vf1 = float(f1_score(yva.cpu().numpy(), vp.cpu().numpy(), average="macro"))
        vacc = float((vp == yva).float().mean().item())
        hist["train_loss"].append(tl / n_seen)
        hist["train_acc"].append(tc / n_seen)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(vacc)
        hist["val_f1"].append(vf1)

        if ep == 1 or ep % 5 == 0 or vf1 > best_f1:
            print(
                f"  ep {ep:3d} | train loss {tl/n_seen:.4f} acc {tc/n_seen:.4f} | "
                f"val loss {vl:.4f} acc {vacc:.4f} f1 {vf1:.4f}",
                flush=True,
            )

        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            torch.save({"state": model.state_dict(), "classes": classes, "arch": name}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stopping en época {ep} (mejor: {best_ep}, val_f1={best_f1:.4f})")
                break

    ck = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(ck["state"])
    return model, hist, best_f1, best_ep, time.time() - t0


def plot_results(out: Path, classes: list[str], res_mlp: dict, res_geo: dict, cm_geo: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Curvas entrenamiento/validación comparativas
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for col, (title, r) in enumerate([("MLP-Base (60D crudas)", res_mlp), ("HandGeoResNet (184D geométricas)", res_geo)]):
        h = r["history"]
        ax[0, col].plot(h["train_loss"], label="Entrenamiento")
        ax[0, col].plot(h["val_loss"], "--", label="Validación")
        ax[0, col].axvline(r["best_epoch"] - 1, color="gray", ls=":", label="Mejor ep.")
        ax[0, col].set_title(f"{title} — Pérdida")
        ax[0, col].set_xlabel("Época")
        ax[0, col].legend()
        ax[0, col].grid(alpha=0.25)

        ax[1, col].plot(h["train_acc"], label="Entrenamiento")
        ax[1, col].plot(h["val_acc"], "--", label="Validación")
        ax[1, col].axvline(r["best_epoch"] - 1, color="gray", ls=":", label="Mejor ep.")
        ax[1, col].set_title(f"{title} — Exactitud")
        ax[1, col].set_xlabel("Época")
        ax[1, col].legend()
        ax[1, col].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "curvas_comparativas.png", dpi=150)
    plt.close(fig)

    # 2. F1 por letra comparativo
    x = np.arange(len(classes))
    w = 0.38
    f1_a = [res_mlp["per_class_f1"][c] for c in classes]
    f1_b = [res_geo["per_class_f1"][c] for c in classes]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(x - w / 2, f1_a, w, label=f"MLP-Base (F1={res_mlp['test_f1_macro']:.3f})", color="#8da0cb")
    ax.bar(x + w / 2, f1_b, w, label=f"HandGeoResNet (F1={res_geo['test_f1_macro']:.3f})", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("F1-Score (Test personas no vistas)")
    ax.set_title("Desempeño por letra en LSM Estático (MSL-ABC)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "f1_por_letra_comparacion.png", dpi=150)
    plt.close(fig)

    # 3. Matriz de confusión normalizada del mejor modelo
    cmn = cm_geo / cm_geo.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusión normalizada — HandGeoResNet (Acc={res_geo['test_accuracy']:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cmn[i, j]
            if val >= 0.01:
                ax.text(j, i, f"{val:.2f}" if i != j else f"{val:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if val > 0.55 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion_georesnet.png", dpi=150)
    plt.close(fig)

    # 4. Top confusiones más frecuentes
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm_geo[i, j] > 0:
                pairs.append((int(cm_geo[i, j]), f"{classes[i]} → {classes[j]}"))
    pairs.sort(reverse=True)
    top = pairs[:12][::-1]
    if top:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([p[1] for p in top], [p[0] for p in top], color="#e07a5f")
        for idx_p, (cnt, _) in enumerate(top):
            ax.text(cnt + max(p[0] for p in top) * 0.01, idx_p, str(cnt), va="center", fontsize=9)
        ax.set_xlabel("Ejemplos de prueba mal clasificados")
        ax.set_title("Confusiones más frecuentes — HandGeoResNet")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "confusiones_frecuentes.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--landmarks-dir", default="data/landmarks")
    ap.add_argument("--parquet-dir", default="data/hf_lsm_dataset/static")
    ap.add_argument("--out", default="outputs/static")
    a = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    (
        pts_tr_all, valid_tr, ytr_all, gtr_all, sub_tr,
        pts_te_all, valid_te, yte_all, gte_all, sub_te,
    ) = load_static_data(a.landmarks_dir, a.parquet_dir)

    classes = sorted({str(c) for c in ytr_all})
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    is_val = np.isin(gtr_all, list(VAL_PERSONS))
    m_tr = (~is_val) & valid_tr
    m_va = is_val & valid_tr
    m_te_det = valid_te

    print(
        f"Total train/val bruto: {len(ytr_all)} (sin mano: {(~valid_tr).mean():.2%}) | "
        f"train limpio: {m_tr.sum()} | val limpio: {m_va.sum()} (personas {sorted(VAL_PERSONS)})"
    )
    print(
        f"Total test bruto: {len(yte_all)} (sin mano: {(~valid_te).mean():.2%}) | "
        f"test detectado: {m_te_det.sum()} (personas {sorted(set(gte_all))})"
    )
    for sub in ("A", "B", "C"):
        ms = sub_te == sub
        print(f"  Test subconjunto {sub}: {ms.sum()} imgs | detección MediaPipe: {valid_te[ms].mean():.2%}")

    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    to_y = lambda y: torch.tensor([cls_to_idx[v] for v in y], dtype=torch.long, device=dev)

    Xtr_t, ytr_t = to_t(pts_tr_all[m_tr]), to_y(ytr_all[m_tr])
    Xva_t, yva_t = to_t(pts_tr_all[m_va]), to_y(ytr_all[m_va])
    Xte_t, yte_t = to_t(pts_te_all[m_te_det]), to_y(yte_all[m_te_det])
    sub_te_det = sub_te[m_te_det]

    # Modelo 1: Baseline ANN (60D crudas, sin aumento 3D ni ingeniería geométrica)
    mlp_base = MLPBase(len(classes)).to(dev)
    mlp_base, hist_mlp, bf1_mlp, bep_mlp, t_mlp = train_one(
        "mlp_base", mlp_base, Xtr_t, ytr_t, Xva_t, yva_t, classes, out,
        epochs=a.epochs, batch_size=a.batch, lr=2e-3, patience=a.patience,
        use_aug3d=False, label_smoothing=0.0,
    )

    # Modelo 2: Arquitectura avanzada (184D geométricas + bloques residuales + aumento 3D + label smoothing)
    georesnet = HandGeoResNet(len(classes)).to(dev)
    georesnet, hist_geo, bf1_geo, bep_geo, t_geo = train_one(
        "georesnet", georesnet, Xtr_t, ytr_t, Xva_t, yva_t, classes, out,
        epochs=a.epochs, batch_size=a.batch, lr=1e-3, patience=a.patience,
        use_aug3d=True, label_smoothing=0.05, weight_decay=1e-3, cosine=True,
    )

    def full_eval(name: str, model: nn.Module, hist, bf1_val, bep, secs):
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(Xte_t), 2048):
                preds.append(model(Xte_t[i : i + 2048]).argmax(1))
            yp = torch.cat(preds).cpu().numpy()
        yt = yte_t.cpu().numpy()
        rep_txt = classification_report(yt, yp, target_names=classes, digits=4)
        rep_dict = classification_report(yt, yp, target_names=classes, output_dict=True)
        cm = confusion_matrix(yt, yp)
        acc = float((yt == yp).mean())
        f1m = float(f1_score(yt, yp, average="macro"))
        precm = float(precision_score(yt, yp, average="macro"))
        recm = float(recall_score(yt, yp, average="macro"))
        det_rate = float(m_te_det.mean())
        eff_acc = acc * det_rate
        by_sub = {}
        for sub in ("A", "B", "C"):
            ms = sub_te_det == sub
            ms_all = sub_te == sub
            s_acc = float((yt[ms] == yp[ms]).mean())
            s_f1 = float(f1_score(yt[ms], yp[ms], average="macro"))
            s_det = float(valid_te[ms_all].mean())
            by_sub[sub] = {"acc_detected": s_acc, "f1_detected": s_f1, "det_rate": s_det, "acc_effective": s_acc * s_det}

        print(f"\n=== TEST {name} (S18, S20 no vistas) ===")
        print(f"Detected Acc: {acc:.4f} | Macro-F1: {f1m:.4f} | MediaPipe Det: {det_rate:.2%} | Effective Acc: {eff_acc:.4f}")
        for sub, sv in by_sub.items():
            print(f"  Sub {sub}: Acc_det={sv['acc_detected']:.4f}, F1_det={sv['f1_detected']:.4f}, Det={sv['det_rate']:.2%}, Acc_eff={sv['acc_effective']:.4f}")
        print(rep_txt)
        (out / f"reporte_{name}.txt").write_text(rep_txt, encoding="utf-8")
        np.savetxt(out / f"matriz_confusion_{name}.csv", cm, fmt="%d", delimiter=",")

        return {
            "model": name,
            "params": sum(p.numel() for p in model.parameters()),
            "test_accuracy": acc,
            "test_precision_macro": precm,
            "test_recall_macro": recm,
            "test_f1_macro": f1m,
            "mediapipe_detection_rate": det_rate,
            "test_effective_accuracy": eff_acc,
            "val_f1_macro": bf1_val,
            "best_epoch": bep,
            "training_seconds": secs,
            "by_subset": by_sub,
            "per_class_f1": {c: float(rep_dict[c]["f1-score"]) for c in classes},
            "history": hist,
        }, cm

    res_mlp, cm_mlp = full_eval("mlp_base", mlp_base, hist_mlp, bf1_mlp, bep_mlp, t_mlp)
    res_geo, cm_geo = full_eval("georesnet", georesnet, hist_geo, bf1_geo, bep_geo, t_geo)

    with open(out / "resumen_metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "params", "acc_detectado", "f1_macro", "precision_macro", "recall_macro",
                     "deteccion_mp", "acc_efectivo", "acc_A", "acc_B", "acc_C", "mejor_epoca", "segundos"])
        for r in (res_mlp, res_geo):
            w.writerow([
                r["model"], r["params"], f"{r['test_accuracy']:.4f}", f"{r['test_f1_macro']:.4f}",
                f"{r['test_precision_macro']:.4f}", f"{r['test_recall_macro']:.4f}",
                f"{r['mediapipe_detection_rate']:.4f}", f"{r['test_effective_accuracy']:.4f}",
                f"{r['by_subset']['A']['acc_detected']:.4f}",
                f"{r['by_subset']['B']['acc_detected']:.4f}",
                f"{r['by_subset']['C']['acc_detected']:.4f}",
                r["best_epoch"], f"{r['training_seconds']:.1f}",
            ])

    json.dump({"classes": classes, "val_persons": sorted(VAL_PERSONS),
               "mlp_base": res_mlp, "georesnet": res_geo},
              open(out / "resultados.json", "w", encoding="utf-8"), indent=2)

    plot_results(out, classes, res_mlp, res_geo, cm_geo)
    print(f"\nTodo guardado en {out}")


if __name__ == "__main__":
    main()
