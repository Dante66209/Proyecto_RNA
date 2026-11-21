"""
Demo en tiempo real con cámara web para el abecedario completo de LSM (27 letras):
  - 21 señas estáticas (HandGeoResNet)
  -  6 señas dinámicas: J, K, Ñ, Q, X, Z (DynamicBiGRUAttn)

Uso:
    python lsm/demo_camera.py
Controles en ventana:
    M : cambiar modo (AUTO / ESTÁTICA / DINÁMICA)
    L : alternar mano derecha / izquierda
    Espacio : agregar letra actual a la palabra
    Backspace : borrar última letra | C : limpiar texto
    Q / Esc : salir
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsm.features import PALM_ANCHORS
from lsm.train_dynamic import DynamicBiGRUAttn, SEQ_LEN, normalize_clip_np
from lsm.train_static import HandGeoResNet


def resample_buffer(frames: list[np.ndarray], target_len: int = SEQ_LEN) -> np.ndarray:
    arr = np.stack(frames)
    t_old = np.linspace(0, len(arr) - 1, len(arr))
    t_new = np.linspace(0, len(arr) - 1, target_len)
    out = np.empty((target_len, 21, 3), dtype=np.float32)
    for j in range(21):
        for c in range(3):
            out[:, j, c] = np.interp(t_new, t_old, arr[:, j, c])
    return normalize_clip_np(out)


def motion_score(frames: list[np.ndarray]) -> float:
    """Mide desplazamiento promedio reciente de la mano normalizado por tamaño de palma."""
    if len(frames) < 10:
        return 0.0
    recent = np.stack(frames[-16:])
    local = recent - recent[:, 0:1, :]
    scale = float(np.median(np.linalg.norm(local[:, PALM_ANCHORS, :], axis=-1).mean(axis=-1)))
    scale = scale if scale > 1e-6 else 1.0
    wrist_disp = np.linalg.norm(recent[-1, 0, :2] - recent[0, 0, :2]) / scale
    wrist_path = np.linalg.norm(np.diff(recent[:, 0, :2], axis=0), axis=-1).sum() / scale
    tip_path = np.linalg.norm(np.diff(recent[:, 8, :2], axis=0), axis=-1).sum() / scale
    return float(max(wrist_disp, 0.5 * wrist_path, 0.5 * tip_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-ckpt", default="outputs/static/georesnet_best.pt")
    ap.add_argument("--dynamic-ckpt", default="outputs/dynamic/bigru_attn_best.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--motion-thresh", type=float, default=0.55)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck_s = torch.load(a.static_ckpt, map_location=dev, weights_only=True)
    classes_s = ck_s["classes"]
    model_s = HandGeoResNet(len(classes_s)).to(dev).eval()
    model_s.load_state_dict(ck_s["state"])

    model_d, classes_d = None, []
    if Path(a.dynamic_ckpt).exists():
        ck_d = torch.load(a.dynamic_ckpt, map_location=dev, weights_only=True)
        classes_d = ck_d["classes"]
        model_d = DynamicBiGRUAttn(len(classes_d)).to(dev).eval()
        model_d.load_state_dict(ck_d["state"])
    else:
        print(f"Aviso: no se encontró {a.dynamic_ckpt}; iniciando solo en modo estático.")

    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {a.cam}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    modes = ["AUTO", "ESTATICA", "DINAMICA"] if model_d is not None else ["ESTATICA"]
    mode_idx = 0
    left_hand = False
    buf = collections.deque(maxlen=40)
    ema_static = None
    ema_dyn = None
    word = ""
    current_label, current_conf, source_tag = "-", 0.0, ""

    print("Cámara lista. Presiona Q o Esc para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        disp = cv2.flip(frame, 1)  # espejo visual cómodo para el usuario
        h, w, _ = disp.shape

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            # Dibujar landmarks sobre la imagen espejada
            lm_copy = type(lm)()
            lm_copy.CopyFrom(lm)
            for p in lm_copy.landmark:
                p.x = 1.0 - p.x
            mp_draw.draw_landmarks(disp, lm_copy, mp_hands.HAND_CONNECTIONS)

            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            if left_hand:
                pts[:, 0] = -pts[:, 0]
            buf.append(pts)

            mode = modes[mode_idx]
            mot = motion_score(list(buf))
            use_dyn = (model_d is not None) and (mode == "DINAMICA" or (mode == "AUTO" and mot >= a.motion_thresh))

            with torch.no_grad():
                if use_dyn and len(buf) >= 16:
                    seq = resample_buffer(list(buf)[-32:], SEQ_LEN)
                    t_seq = torch.tensor(seq, dtype=torch.float32, device=dev).unsqueeze(0)
                    probs = torch.softmax(model_d(t_seq), dim=-1).squeeze(0).cpu().numpy()
                    ema_dyn = probs if ema_dyn is None else 0.5 * ema_dyn + 0.5 * probs
                    k = int(np.argmax(ema_dyn))
                    current_label, current_conf, source_tag = classes_d[k], float(ema_dyn[k]), "DINAMICA"
                else:
                    pts_local = pts - pts[0:1]
                    t_pts = torch.tensor(pts_local, dtype=torch.float32, device=dev).unsqueeze(0)
                    probs = torch.softmax(model_s(t_pts), dim=-1).squeeze(0).cpu().numpy()
                    ema_static = probs if ema_static is None else 0.6 * ema_static + 0.4 * probs
                    k = int(np.argmax(ema_static))
                    current_label, current_conf, source_tag = classes_s[k], float(ema_static[k]), "ESTATICA"
        else:
            buf.clear()
            ema_static = None
            ema_dyn = None
            current_label, current_conf, source_tag = "Sin mano", 0.0, ""

        # Barra superior de estado
        cv2.rectangle(disp, (0, 0), (w, 95), (20, 25, 30), -1)
        hand_txt = "IZQ" if left_hand else "DER"
        cv2.putText(disp, f"LSM | Modo: {modes[mode_idx]} | Mano: {hand_txt} | {source_tag}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)
        color = (80, 230, 120) if current_conf >= 0.7 else (0, 200, 255)
        cv2.putText(disp, f"Letra: {current_label}  ({current_conf:.0%})",
                    (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.15, color, 3)

        # Barra inferior para palabra formada
        cv2.rectangle(disp, (0, h - 55), (w, h), (20, 25, 30), -1)
        cv2.putText(disp, f"Texto: {word}_", (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.imshow("LSMovil - Reconocimiento en Vivo", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(modes)
        elif key == ord("l"):
            left_hand = not left_hand
            buf.clear()
        elif key == ord(" ") and len(current_label) <= 2 and current_conf >= 0.5:
            word += current_label
        elif key == 8 and word:
            word = word[:-1]
        elif key == ord("c"):
            word = ""

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
