"""
Extrae secuencias temporales de landmarks (MediaPipe Hands) de videos de señas dinámicas LSM
(Zenodo 14689869: J, K, Ñ, Q, X, Z).

Guarda las coordenadas crudas en espacio de imagen (T, 21, 3) para NO perder la trayectoria
espacial de la muñeca (fundamental para J, Z, Ñ, Q, X, K), junto con la máscara de detección
por cuadro, etiqueta, persona, vista (frontal/perfil) y split (train/test).

Uso:
    python lsm/extract_dynamic.py --root data/raw/MSL-dynamic-signs --out data/landmarks/dynamic_lsm.npz
"""
import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


TEST_PERSONS = {"S20", "S21"}


def _parse_meta(path: Path, root: Path):
    rel = path.relative_to(root)
    stem = path.stem
    m = re.match(r"^(S\d+)-([^-]+)-([^-.]+)", stem)
    if not m:
        raise ValueError(f"Nombre inesperado: {path}")
    person, label, view_raw = m.group(1), m.group(2).upper(), m.group(3).lower()
    if label not in {"J", "K", "Ñ", "Q", "X", "Z"}:
        raise ValueError(f"Etiqueta inesperada {label!r} en {path}")
    split = "test" if (rel.parts[0] == "test" or person in TEST_PERSONS) else "train"
    view = "frontal" if "front" in view_raw else "profile"
    return split, person, label, view, str(rel)


def _process_video(args):
    path_str, root_str = args
    import mediapipe as mp

    path, root = Path(path_str), Path(root_str)
    split, person, label, view, rel = _parse_meta(path, root)

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    frames_lm = []
    detected = []
    # Instancia nueva por video con static_image_mode=False para aprovechar tracking temporal sin contaminar entre videos
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                frames_lm.append(pts)
                detected.append(True)
            else:
                frames_lm.append(np.zeros((21, 3), dtype=np.float32))
                detected.append(False)
    cap.release()

    seq = np.stack(frames_lm) if frames_lm else np.zeros((0, 21, 3), dtype=np.float32)
    det = np.array(detected, dtype=bool)
    return seq, det, split, person, label, view, rel, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/raw/MSL-dynamic-signs", "data/raw/MSL dynamic-profile-signs"])
    ap.add_argument("--out", default="data/landmarks/dynamic_lsm.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    a = ap.parse_args()

    items = []
    for r_str in a.roots:
        r_path = Path(r_str)
        if not r_path.exists():
            print(f"Aviso: no existe {r_path}, se omite.")
            continue
        for v in sorted(r_path.rglob("*.mp4")):
            items.append((str(v), str(r_path)))
    print(f"Videos encontrados: {len(items)} | workers: {a.workers}", flush=True)

    seqs, dets, splits, persons, labels, views, paths, fpss = [], [], [], [], [], [], [], []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(_process_video, items, chunksize=4), 1):
            seq, det, sp, per, lab, vw, rel, fps = r
            seqs.append(seq)
            dets.append(det)
            splits.append(sp)
            persons.append(per)
            labels.append(lab)
            views.append(vw)
            paths.append(rel)
            fpss.append(fps)
            if i % 100 == 0 or i == len(items):
                print(f"  {i}/{len(items)} videos procesados", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        seqs=np.array(seqs, dtype=object),
        detected=np.array(dets, dtype=object),
        splits=np.array(splits),
        persons=np.array(persons),
        labels=np.array(labels),
        views=np.array(views),
        paths=np.array(paths),
        fps=np.array(fpss, dtype=np.float32),
    )
    det_rates = [float(d.mean()) if len(d) else 0.0 for d in dets]
    print(
        f"Guardado {a.out}: {len(seqs)} secuencias | detección promedio por video: {np.mean(det_rates):.1%} | "
        f"videos con <20% detección: {sum(r < 0.2 for r in det_rates)}"
    )
    for sp in sorted(set(splits)):
        m = np.array(splits) == sp
        print(f"  {sp}: {m.sum()} clips | personas: {len(set(np.array(persons)[m]))} | clases: {sorted(set(np.array(labels)[m]))}")


if __name__ == "__main__":
    main()
