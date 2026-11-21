"""
Empaqueta todo el dataset LSM (21 señas estáticas de MSL-ABC + 6 señas dinámicas J/K/Ñ/Q/X/Z)
en formato Parquet (compresión ZSTD) listo para subir a Hugging Face Hub (~60-80 MB en total
en lugar de 8 GB comprimidos / 20+ GB crudos).

Genera data/hf_lsm_dataset/ con:
  - static/train.parquet, static/validation.parquet, static/test.parquet
  - dynamic/train.parquet, dynamic/validation.parquet, dynamic/test.parquet
  - README.md (Dataset Card con metadatos YAML para load_dataset(..., "static"|"dynamic"))

Uso:
  python lsm/prepare_hf_package.py
  python lsm/prepare_hf_package.py --push-to tu_usuario/lsm-alfabeto-landmarks
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsm.features import pick_single_hand_np

STATIC_VAL_PERSONS = {"S5", "S12", "S17"}
DYN_VAL_PERSONS = {"S5", "S13", "S17"}


def write_static_parquet(out_dir: Path, landmarks_dir: Path):
    static_dir = out_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "validation": [], "test": []}

    for sp_file in ("train", "test"):
        for sub in ("A", "B", "C"):
            d = np.load(landmarks_dir / f"abc_{sub}_{sp_file}.npz")
            pts, valid = pick_single_hand_np(d["X"])
            # Guardamos solo la mano activa (63 floats en float16/float32) + bandera detected
            vecs = pts.reshape(len(pts), 63).astype(np.float32)
            labels = d["y"].astype(str)
            persons = d["groups"].astype(str)
            n_hands = d["n_hands"].astype(np.int8)
            paths = d["paths"].astype(str)

            for i in range(len(labels)):
                if sp_file == "test":
                    sp = "test"
                elif persons[i] in STATIC_VAL_PERSONS:
                    sp = "validation"
                else:
                    sp = "train"
                splits[sp].append((vecs[i], labels[i], persons[i], sub, bool(valid[i]), int(n_hands[i]), paths[i]))

    for sp, rows in splits.items():
        # Construir columnas PyArrow eficientes (FixedSizeList o list<float32>)
        arr_vec = pa.array([r[0] for r in rows], type=pa.list_(pa.float32(), 63))
        table = pa.table({
            "landmarks": arr_vec,
            "label": pa.array([r[1] for r in rows], type=pa.string()),
            "person": pa.array([r[2] for r in rows], type=pa.string()),
            "subset": pa.array([r[3] for r in rows], type=pa.string()),
            "detected": pa.array([r[4] for r in rows], type=pa.bool_()),
            "n_hands": pa.array([r[5] for r in rows], type=pa.int8()),
            "source_path": pa.array([r[6] for r in rows], type=pa.string()),
        })
        p = static_dir / f"{sp}.parquet"
        pq.write_table(table, p, compression="zstd", compression_level=9)
        print(f"  static/{sp}.parquet: {len(rows):,} filas -> {p.stat().st_size / 1e6:.2f} MB")


def write_dynamic_parquet(out_dir: Path, landmarks_dir: Path):
    dyn_dir = out_dir / "dynamic"
    dyn_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(landmarks_dir / "dynamic_lsm.npz", allow_pickle=True)
    seqs, dets, splits_raw, persons, labels, views, paths, fps = (
        d["seqs"], d["detected"], d["splits"], d["persons"], d["labels"], d["views"], d["paths"], d["fps"]
    )
    buckets = {"train": [], "validation": [], "test": []}
    for i in range(len(seqs)):
        sp_raw, per = str(splits_raw[i]), str(persons[i])
        sp = "test" if sp_raw == "test" else ("validation" if per in DYN_VAL_PERSONS else "train")
        seq = np.asarray(seqs[i], dtype=np.float32).reshape(-1).tolist()
        det = np.asarray(dets[i], dtype=bool).tolist()
        buckets[sp].append((seq, det, int(len(seqs[i])), str(labels[i]), per, str(views[i]), float(fps[i]), str(paths[i])))

    for sp, rows in buckets.items():
        table = pa.table({
            "sequence": pa.array([r[0] for r in rows], type=pa.list_(pa.float32())),  # T*63 plano (reshape (-1,21,3))
            "detected": pa.array([r[1] for r in rows], type=pa.list_(pa.bool_())),
            "num_frames": pa.array([r[2] for r in rows], type=pa.int32()),
            "label": pa.array([r[3] for r in rows], type=pa.string()),
            "person": pa.array([r[4] for r in rows], type=pa.string()),
            "view": pa.array([r[5] for r in rows], type=pa.string()),
            "fps": pa.array([r[6] for r in rows], type=pa.float32()),
            "source_path": pa.array([r[7] for r in rows], type=pa.string()),
        })
        p = dyn_dir / f"{sp}.parquet"
        pq.write_table(table, p, compression="zstd", compression_level=9)
        print(f"  dynamic/{sp}.parquet: {len(rows):,} clips -> {p.stat().st_size / 1e6:.2f} MB")


DATASET_CARD = """---
license: cc-by-4.0
language:
- es
task_categories:
- other
tags:
- mexican-sign-language
- lsm
- mediapipe
- hand-landmarks
- sign-language-recognition
configs:
- config_name: static
  default: true
  data_files:
  - split: train
    path: static/train.parquet
  - split: validation
    path: static/validation.parquet
  - split: test
    path: static/test.parquet
- config_name: dynamic
  data_files:
  - split: train
    path: dynamic/train.parquet
  - split: validation
    path: dynamic/validation.parquet
  - split: test
    path: dynamic/test.parquet
---

# Alfabeto Completo de Lengua de Señas Mexicana (LSM) — Landmarks 3D MediaPipe

Representación compacta (~88 MB en Parquet ZSTD frente a ~8 GB originales en `.7z`) de las **27 letras del alfabeto dactilológico de la LSM**, extraída con MediaPipe Hands (`0.10.21`) y dividida estrictamente **por persona** (sin fuga de datos entre entrenamiento, validación y prueba):

- **Configuración `static` (21 letras estáticas: A, B, C, D, E, F, G, H, I, L, M, N, O, P, R, S, T, U, V, W, Y)**
  - Fuente: *Mexican Sign Language Alphabet (static signs only)* (`10.5281/zenodo.10067509`, CC-BY-4.0), 279,716 imágenes de 20 participantes en 3 niveles de variación (A, B, C).
  - `train`: 15 personas (`S1..S19` excepto validación) | `validation`: `S5, S12, S17` | `test`: `S18, S20`.
- **Configuración `dynamic` (6 letras dinámicas: J, K, Ñ, Q, X, Z)**
  - Fuente: *Mexican Sign Language Alphabet (dynamic signs only)* (`10.5281/zenodo.14689869`, CC-BY-4.0), 1,242 videos (vistas frontal y perfil 45°) de 20 participantes.
  - `train`: 15 personas (`S1..S19` excepto validación) | `validation`: `S5, S13, S17` | `test`: `S20, S21`.

## Uso rápido en Jupyter / Colab

```python
from datasets import load_dataset

ds_static = load_dataset("<TU_USUARIO>/lsm-alfabeto-landmarks", "static")
ds_dynamic = load_dataset("<TU_USUARIO>/lsm-alfabeto-landmarks", "dynamic")
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-dir", default="data/landmarks")
    ap.add_argument("--out", default="data/hf_lsm_dataset")
    ap.add_argument("--push-to", default="", help="repo_id en Hugging Face (p.ej. usuario/lsm-alfabeto-landmarks)")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = Path(a.landmarks_dir)

    print("Empaquetando estáticas en Parquet...")
    write_static_parquet(out_dir, lm_dir)
    print("Empaquetando dinámicas en Parquet...")
    write_dynamic_parquet(out_dir, lm_dir)
    (out_dir / "README.md").write_text(DATASET_CARD, encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"\nPaquete listo en {out_dir} | Tamaño total: {total_mb:.2f} MB")

    if a.push_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=a.push_to, repo_type="dataset", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=a.push_to, repo_type="dataset")
        print(f"Subido exitosamente a https://huggingface.co/datasets/{a.push_to}")


if __name__ == "__main__":
    main()
