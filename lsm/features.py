"""
Utilidades geométricas compartidas para landmarks de MediaPipe Hands (21 puntos x 3D).

Se usa tanto en entrenamiento (estático y dinámico) como en la demo de cámara en vivo.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Aristas anatómicas de la mano (padre -> hijo) según MediaPipe Hands
HAND_PARENTS = [
    0, 1, 2, 3,        # pulgar:  0->1->2->3->4
    0, 5, 6, 7,        # índice:  0->5->6->7->8
    0, 9, 10, 11,      # medio:   0->9->10->11->12
    0, 13, 14, 15,     # anular:  0->13->14->15->16
    0, 17, 18, 19,     # meñique: 0->17->18->19->20
]

# Tripletas (a, b, c) para ángulos de flexión en cada articulación (15 ángulos)
FLEX_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # pulgar
    (0, 5, 6), (5, 6, 7), (6, 7, 8),       # índice
    (0, 9, 10), (9, 10, 11), (10, 11, 12), # medio
    (0, 13, 14), (13, 14, 15), (14, 15, 16), # anular
    (0, 17, 18), (17, 18, 19), (18, 19, 20), # meñique
]

# Pares de huesos proximales adyacentes para abducción/apertura entre dedos (4 ángulos)
SPREAD_PAIRS = [
    ((1, 2), (5, 6)),
    ((5, 6), (9, 10)),
    ((9, 10), (13, 14)),
    ((13, 14), (17, 18)),
]

FINGERTIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(FINGERTIPS[i], FINGERTIPS[j]) for i in range(5) for j in range(i + 1, 5)]  # 10 pares
THUMB_TO_JOINTS = [(4, j) for j in (5, 6, 9, 10, 13, 14, 17, 18)]  # 8 pares (clave para A/E/M/N/S/T)
PALM_ANCHORS = [5, 9, 13, 17]

FEATURE_DIM = 184
# Índices exactos de palm_orient [v_up(3), v_norm(3)] dentro del vector de 184 features:
# 60 (view) + 60 (palm) + 15 (flex) + 4 (spread) + 5 (tip_wrist) + 10 (tip_dist) + 8 (thumb_dist) = 162
PALM_ORIENT_SLICE = slice(162, 168)


def pick_single_hand_np(X126: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """De un arreglo (N, 126) con [Left, Right], extrae la mano activa (N, 21, 3) y una máscara válida (N,).
    Prefiere slot 1 (Right); si está vacío, toma slot 0 (Left) para cubrir errores de lateralidad de MediaPipe."""
    pts2 = X126.reshape(-1, 2, 21, 3).astype(np.float32)
    e_left = np.abs(pts2[:, 0]).sum(axis=(1, 2)) > 1e-6
    e_right = np.abs(pts2[:, 1]).sum(axis=(1, 2)) > 1e-6
    pts = np.where(e_right[:, None, None], pts2[:, 1], pts2[:, 0])
    valid = e_left | e_right
    return pts, valid


def renormalize_3d(pts: torch.Tensor) -> torch.Tensor:
    """Centra en la muñeca (punto 0) y normaliza por la escala promedio 3D de los nudillos.
    Corrige la distorsión que ocurre cuando se normaliza solo con la proyección 2D xy bajo inclinación."""
    pts = pts - pts[:, 0:1, :]
    palm_dist = pts[:, PALM_ANCHORS, :].norm(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    return pts / palm_dist.clamp_min(1e-6)


def augment_hand_3d(
    pts: torch.Tensor,
    max_yaw_deg: float = 30.0,
    max_pitch_deg: float = 26.0,
    max_roll_deg: float = 22.0,
    noise_std: float = 0.018,
    scale_jitter: float = 0.10,
) -> torch.Tensor:
    """Aumento de datos 3D realista en GPU: rotaciones yaw/pitch/roll, escala anisotrópica leve y ruido.
    No se aplica espejo horizontal porque cambiaría lateralidad de la seña."""
    b, dev, dt = pts.shape[0], pts.device, pts.dtype
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
    R = Rz @ Ry @ Rx

    out = torch.bmm(pts, R.transpose(1, 2))
    # Pequeña variación de proporción entre ejes (simula manos más anchas o largas)
    s = 1.0 + (torch.rand(b, 1, 3, device=dev, dtype=dt) * 2 - 1) * scale_jitter
    out = out * s + torch.randn_like(out) * noise_std
    return renormalize_3d(out)


def compute_hand_features(pts: torch.Tensor) -> torch.Tensor:
    """Convierte landmarks (B, 21, 3) en 184 features geométricas:
      - 60: coordenadas 3D normalizadas en vista de cámara (orientación global de la seña)
      - 60: coordenadas 3D alineadas al marco canónico de la palma (invariante a rotación 3D)
      - 15: cosenos de ángulos de flexión intra-dedo
      -  4: cosenos de apertura entre dedos adyacentes (falanges proximales)
      -  5: distancias punta-muñeca
      - 10: distancias entre pares de puntas de dedos
      -  8: distancias punta del pulgar a nudillos/falanges
      -  6: vectores unitarios de orientación de la palma (eje vertical y normal) [indices 162:168]
      - 16 (extra): 4 apertura rayo completo + 3 razones puntas/nudillos + 2 cruce/orden lateral R vs U
                    + 3 profundidad anular/pulgar M/N y E/S + 4 distancias puntas a nudillo medio
    """
    pts = renormalize_3d(pts)
    b = pts.shape[0]

    # 1. Coordenadas en cámara (60)
    view_coords = pts[:, 1:, :].reshape(b, 60)

    # 2. Marco canónico de la palma (60 + 6 orientación)
    v_up = F.normalize(pts[:, 9] - pts[:, 0], dim=-1, eps=1e-6)
    v_side = pts[:, 5] - pts[:, 17]
    v_norm = F.normalize(torch.linalg.cross(v_side, v_up, dim=-1), dim=-1, eps=1e-6)
    v_x = F.normalize(torch.linalg.cross(v_up, v_norm, dim=-1), dim=-1, eps=1e-6)
    R_palm = torch.stack([v_x, v_up, v_norm], dim=-1)  # (B, 3, 3)
    palm_pts = torch.bmm(pts, R_palm)                  # (B, 21, 3) en marco de palma
    palm_coords = palm_pts[:, 1:, :].reshape(b, 60)
    palm_orient = torch.cat([v_up, v_norm], dim=-1)    # (B, 6)

    # 3. Cosenos de flexión (15)
    a_idx = torch.tensor([t[0] for t in FLEX_TRIPLETS], device=pts.device)
    b_idx = torch.tensor([t[1] for t in FLEX_TRIPLETS], device=pts.device)
    c_idx = torch.tensor([t[2] for t in FLEX_TRIPLETS], device=pts.device)
    v1 = F.normalize(pts[:, a_idx] - pts[:, b_idx], dim=-1, eps=1e-6)
    v2 = F.normalize(pts[:, c_idx] - pts[:, b_idx], dim=-1, eps=1e-6)
    flex_cos = (v1 * v2).sum(dim=-1)  # (B, 15)

    # 4. Apertura entre dedos adyacentes: proximal (4) + rayos completos nudillo->punta (4)
    s1 = F.normalize(
        torch.stack([pts[:, p[0][1]] - pts[:, p[0][0]] for p in SPREAD_PAIRS], dim=1),
        dim=-1, eps=1e-6,
    )
    s2 = F.normalize(
        torch.stack([pts[:, p[1][1]] - pts[:, p[1][0]] for p in SPREAD_PAIRS], dim=1),
        dim=-1, eps=1e-6,
    )
    spread_cos = (s1 * s2).sum(dim=-1)  # (B, 4)

    rays = torch.stack([
        pts[:, 4] - pts[:, 1],
        pts[:, 8] - pts[:, 5],
        pts[:, 12] - pts[:, 9],
        pts[:, 16] - pts[:, 13],
        pts[:, 20] - pts[:, 17],
    ], dim=1)
    rays_u = F.normalize(rays, dim=-1, eps=1e-6)
    ray_spread = (rays_u[:, :-1] * rays_u[:, 1:]).sum(dim=-1)  # (B, 4) clave para U vs V vs W

    # 5. Distancias clave (5 + 10 + 8 = 23) + razones y cruces específicos (11)
    tip_wrist = pts[:, FINGERTIPS, :].norm(dim=-1)  # (B, 5)
    t_i = torch.tensor([p[0] for p in TIP_PAIRS], device=pts.device)
    t_j = torch.tensor([p[1] for p in TIP_PAIRS], device=pts.device)
    tip_dist = (pts[:, t_i] - pts[:, t_j]).norm(dim=-1)  # (B, 10)
    th_i = torch.tensor([p[0] for p in THUMB_TO_JOINTS], device=pts.device)
    th_j = torch.tensor([p[1] for p in THUMB_TO_JOINTS], device=pts.device)
    thumb_dist = (pts[:, th_i] - pts[:, th_j]).norm(dim=-1)  # (B, 8)

    # Razón separación de puntas vs nudillos (U vs V vs R)
    ratio_8_12 = (pts[:, 8] - pts[:, 12]).norm(dim=-1) / (pts[:, 5] - pts[:, 9]).norm(dim=-1).clamp_min(1e-4)
    ratio_12_16 = (pts[:, 12] - pts[:, 16]).norm(dim=-1) / (pts[:, 9] - pts[:, 13]).norm(dim=-1).clamp_min(1e-4)
    ratio_16_20 = (pts[:, 16] - pts[:, 20]).norm(dim=-1) / (pts[:, 13] - pts[:, 17]).norm(dim=-1).clamp_min(1e-4)
    # Orden lateral y cruce R vs U en marco de la palma + quiralidad
    cross_ru = (torch.linalg.cross(rays_u[:, 1], rays_u[:, 2], dim=-1) * v_norm).sum(dim=-1)
    lat_8_12 = palm_pts[:, 8, 0] - palm_pts[:, 12, 0]
    # Posición anular/pulgar para M vs N y E vs S
    ring_vs_mid_y = palm_pts[:, 16, 1] - palm_pts[:, 12, 1]
    ring_vs_mid_z = palm_pts[:, 16, 2] - palm_pts[:, 12, 2]
    thumb_z = palm_pts[:, 4, 2]
    # Distancia puntas [8,12,16,20] a nudillo medio 9 (E vs S vs B/F)
    tip_to_mid_mcp = (pts[:, [8, 12, 16, 20]] - pts[:, 9:10]).norm(dim=-1)  # (B, 4)

    extra = torch.cat([
        ray_spread,
        torch.stack([ratio_8_12, ratio_12_16, ratio_16_20, cross_ru, lat_8_12,
                     ring_vs_mid_y, ring_vs_mid_z, thumb_z], dim=-1),
        tip_to_mid_mcp,
    ], dim=-1)  # 4 + 8 + 4 = 16 -> total 184 features

    return torch.cat(
        [view_coords, palm_coords, flex_cos, spread_cos, tip_wrist, tip_dist, thumb_dist, palm_orient, extra],
        dim=-1,
    )


class HandFeatureLayer(nn.Module):
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return compute_hand_features(pts)
