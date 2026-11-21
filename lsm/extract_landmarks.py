"""
Extrae landmarks de manos (MediaPipe Hands) de un dataset de imágenes organizado como:

    ROOT/<etiqueta>/**/*.jpg|png     (las subcarpetas intermedias, p.ej. persona, son opcionales)

Salida: un .npz con
    X       (N, 126) float32  -> 2 manos x 21 puntos x (x,y,z), normalizado (ver normalize_hand)
    y       (N,)     str      -> etiqueta (nombre de la carpeta de primer nivel)
    groups  (N,)     str      -> subcarpeta intermedia (persona/sesión) o "" si no existe.
                                 Sirve para dividir train/val/test POR PERSONA y evitar fuga de datos.
    paths   (N,)     str      -> ruta relativa del archivo
    n_hands (N,)     int8     -> manos detectadas (0, 1 o 2)

Uso:
    python lsm/extract_landmarks.py --root data/raw/MSL-ABC --out data/landmarks_static.npz --workers 6
    python lsm/extract_landmarks.py --root split/.../train_224 --out data/test.npz --limit 50   # prueba rápida
"""
import argparse
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_hands = None  # una instancia por proceso (crearla por imagen es MUY lento)


def _init_worker():
    global _hands
    import mediapipe as mp

    _hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )


def normalize_hand(pts):
    """pts: (21,3). Resta la muñeca (punto 0) y escala por la distancia muñeca -> base del dedo
    medio (punto 9). Así el modelo no depende de la posición en el cuadro ni de la distancia a la cámara."""
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9, :2])
    return pts / (scale if scale > 1e-6 else 1.0)


def landmarks_from_bgr(image, hands):
    """Devuelve (vector de 126, n_manos). Slot 0 = mano 'Left', slot 1 = 'Right' según MediaPipe,
    para que el orden sea consistente y no dependa de cuál mano se detecte primero."""
    res = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out = np.zeros((2, 21, 3), dtype=np.float32)
    n = 0
    if res.multi_hand_landmarks:
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            slot = 0 if hd.classification[0].label == "Left" else 1
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            out[slot] = normalize_hand(pts)
            n += 1
    return out.reshape(-1), min(n, 2)


def _process(args):
    path, root = args
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # soporta rutas con ñ
    if img is None:
        return None
    vec, n = landmarks_from_bgr(img, _hands)
    rel = Path(path).relative_to(root)
    label = rel.parts[0]
    # persona: subcarpeta intermedia si existe; si no, prefijo del nombre de archivo (p.ej. "S4-N-1-23.jpg" -> "S4")
    m = re.match(r"(S\d+)-", rel.name)
    group = rel.parts[1] if len(rel.parts) > 2 else (m.group(1) if m else "")
    return vec, label, group, str(rel), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0, help="máx. imágenes por clase (0 = todas), para pruebas")
    a = ap.parse_args()

    root = Path(a.root)
    files, per_class = [], {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTS:
            lab = p.relative_to(root).parts[0]
            if a.limit and per_class.get(lab, 0) >= a.limit:
                continue
            per_class[lab] = per_class.get(lab, 0) + 1
            files.append((str(p), str(root)))
    print(f"{len(files)} imágenes, {len(per_class)} clases, {a.workers} procesos", flush=True)

    X, y, g, paths, nh = [], [], [], [], []
    with Pool(a.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(_process, files, chunksize=32), 1):
            if r is not None:
                X.append(r[0]); y.append(r[1]); g.append(r[2]); paths.append(r[3]); nh.append(r[4])
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    nh = np.array(nh, dtype=np.int8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y), groups=np.array(g),
                        paths=np.array(paths), n_hands=nh)
    print(f"Guardado {a.out}: {len(X)} muestras | sin mano detectada: {(nh == 0).mean():.1%} | 2 manos: {(nh == 2).mean():.1%}")
    for lab in sorted(set(y)):
        m = np.array(y) == lab
        print(f"  {lab}: {m.sum()} (sin mano: {(nh[m] == 0).mean():.0%})")


if __name__ == "__main__":
    main()
