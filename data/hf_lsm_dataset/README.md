---
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

## Uso rápido en local / Google Colab (desde este repositorio)

Si clonaste el repositorio del proyecto, los archivos Parquet ZSTD ya están incluidos en `data/hf_lsm_dataset/` (~88 MB en total) y pueden leerse directamente con PyArrow:

```python
import pyarrow.parquet as pq

st_train = pq.read_table("data/hf_lsm_dataset/static/train.parquet")
dy_train = pq.read_table("data/hf_lsm_dataset/dynamic/train.parquet")
```

## Uso opcional vía Hugging Face Hub

Si subes esta carpeta a un repositorio de Hugging Face Datasets (por ejemplo con `python lsm/prepare_hf_package.py --push-to usuario/lsm-alfabeto-landmarks`), también puedes cargarlo directamente con `datasets`:

```python
from datasets import load_dataset

ds_static = load_dataset("usuario/lsm-alfabeto-landmarks", "static")
ds_dynamic = load_dataset("usuario/lsm-alfabeto-landmarks", "dynamic")
```
