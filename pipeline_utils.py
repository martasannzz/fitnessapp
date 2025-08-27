# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 13:08:53 2025

@author: marta
"""

import numpy as np
import logging
from collections import Counter

# Configuración de logging
logging.basicConfig(
    filename="pipeline_log.txt",  # archivo donde se guardan los descartes
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------
# Recomendación visible en la app (single source of truth para UI)
# ------------------------------------------------------------------
RECOMENDACION_ANGULO = {
    # Tirón/sentadilla y patrones de bisagra → vista lateral o 45°
    "deadlift": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "romanian deadlift": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "squat": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "t bar row": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "push up": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "plank": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "leg_extension": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "leg raises": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "hip thrust": "Lateral (perfil). Aceptable 45° semi-lateral.",

    # Press banca y variantes → lateral o 45°, frontal no tan útil
    "bench press": "Lateral (perfil) o 45° semi-lateral.",
    "incline bench press": "Lateral (perfil) o 45° semi-lateral.",
    "decline bench press": "Lateral (perfil) o 45° semi-lateral.",

    # Tirones verticales o ejercicios de simetría → frontal
    "lateral pulldown": "Frontal (de frente). Aceptable 45° semi-lateral.",
    "pull up": "Frontal (de frente). Aceptable 45° semi-lateral.",
    "lateral raise": "Frontal (de frente). Aceptable 45° semi-lateral.",
    "russian twist": "Frontal (de frente). Aceptable 45° semi-lateral.",
    "chest fly machine": "Frontal (de frente). Aceptable 45° semi-lateral.",
    "shoulder press": "Frontal (de frente). Aceptable 45° semi-lateral.",

    # Brazos en polea/mancuerna → lateral mejor para ver codo/hombro
    "biceps curl": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "barbell biceps curl": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "hammer curl": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "tricep pushdown": "Lateral (perfil). Aceptable 45° semi-lateral.",
    "tricep dips": "Lateral (perfil). Aceptable 45° semi-lateral.",
}

class PipelineVerificacion:
    """
    Verificaciones ligeras antes de aplicar reglas biomecánicas:
      - Keypoints mínimos y confianza
      - Saltos bruscos (tracking)
      - Estimación de ángulo de cámara y validación por ejercicio
    """
    # Mapeo de vista requerida por ejercicio
    _REQ_ANGULOS = {
        # LATERAL obligatorio/preferente (se acepta semi-lateral 45°)
        "deadlift": {"lateral", "semi-lateral"},
        "romanian deadlift": {"lateral", "semi-lateral"},
        "squat": {"lateral", "semi-lateral"},
        "t bar row": {"lateral", "semi-lateral"},
        "push up": {"lateral", "semi-lateral"},
        "plank": {"lateral", "semi-lateral"},
        "leg_extension": {"lateral", "semi-lateral"},
        "leg raises": {"lateral", "semi-lateral"},
        "hip thrust": {"lateral", "semi-lateral"},

        # Banca y variantes: lateral o semi-lateral
        "bench press": {"lateral", "semi-lateral"},
        "incline bench press": {"lateral", "semi-lateral"},
        "decline bench press": {"lateral", "semi-lateral"},

        # Vista FRONTAL preferente (se acepta semi-lateral)
        "lateral pulldown": {"frontal", "semi-lateral"},
        "pull up": {"frontal", "semi-lateral"},
        "lateral raise": {"frontal", "semi-lateral"},
        "russian twist": {"frontal", "semi-lateral"},
        "chest fly machine": {"frontal", "semi-lateral"},
        "shoulder press": {"frontal", "semi-lateral"},

        # Brazos: lateral o semi-lateral
        "biceps curl": {"lateral", "semi-lateral"},
        "barbell biceps curl": {"lateral", "semi-lateral"},
        "hammer curl": {"lateral", "semi-lateral"},
        "tricep pushdown": {"lateral", "semi-lateral"},
        "tricep dips": {"lateral", "semi-lateral"},
    }

    # (Opcional) Secuencias válidas si en el futuro quieres usarlas aquí
    _SECUENCIAS_VALIDAS = {
        "squat": ["setup","eccentric","bottom","concentric"],
        "deadlift": ["setup","pull","lockout","lowering"],
        "romanian deadlift": ["setup","eccentric","bottom","lockout"],
        "bench press": ["setup","lowering","press"],
        "incline bench press": ["setup","lowering","press"],
        "decline bench press": ["setup","lowering","press"],
        "chest fly machine": ["open","close"],
        "barbell biceps curl": ["setup","curl","lowering"],
        "biceps curl": ["setup","curl","lowering"],
        "hammer curl": ["setup","curl","lowering"],
        "tricep pushdown": ["setup","eccentric","press"],
        "tricep dips": ["setup","lowering","press"],
        "lateral raise": ["setup","raise","lowering"],
        "lateral pulldown": ["setup","pull","return"],
        "pull up": ["hang","pull","lowering"],
        "push up": ["setup","lowering","press"],
        "t bar row": ["setup","pull","lowering"],
        "leg_extension": ["setup","extend","return"],
        "leg raises": ["setup","raise","lowering"],
        "russian twist": ["neutral","rotate_left","rotate_right"],
        "plank": ["hold"],
        "hip thrust": ["bottom","concentric","lockout","eccentric"],
        "shoulder press": ["setup","lowering","press"],
    }

    def __init__(
        self,
        min_conf: float = 0.3,
        max_jump: float = 100.0,
        smooth_factor: float = 0.5,
        min_visible_ratio: float = 0.5,
        angle_decision_frames: int = 20,
    ):
        self.min_conf = min_conf
        self.max_jump = max_jump
        self.smooth_factor = smooth_factor
        self.min_visible_ratio = min_visible_ratio
        self.angle_decision_frames = angle_decision_frames

        self.prev_keypoints = None
        self.view_votes = Counter()
        self.frames_seen = 0
        self.decided_view = None  # 'frontal' | 'lateral' | 'semi-lateral'

    # -------------------------
    # Utils de formato
    # -------------------------
    @staticmethod
    def _ensure_kp_format(keypoints):
        """
        Acepta (N,2) o (N,3). Devuelve (N,3) con conf si falta.
        """
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.ndim != 2 or keypoints.shape[1] not in (2, 3):
            return None
        if keypoints.shape[1] == 2:
            conf = np.ones((keypoints.shape[0], 1), dtype=float)
            keypoints = np.concatenate([keypoints, conf], axis=1)
        return keypoints

    # -------------------------
    # Validación de keypoints
    # -------------------------
    def validar_keypoints(self, keypoints_3c):
        """
        Pone NaN en puntos con baja confianza o coords inválidas.
        """
        kp = keypoints_3c.copy()
        for i in range(kp.shape[0]):
            x, y, c = kp[i]
            if c < self.min_conf or x < 0 or y < 0:
                kp[i] = [np.nan, np.nan, 0.0]
        return kp

    # -------------------------
    # Suavizado temporal (EMA)
    # -------------------------
    def suavizar(self, keypoints_3c):
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints_3c
            return keypoints_3c

        out = self.prev_keypoints.copy()
        for i in range(keypoints_3c.shape[0]):
            x_new, y_new, c_new = keypoints_3c[i]
            x_prev, y_prev, c_prev = self.prev_keypoints[i]

            if np.isnan(x_new) or np.isnan(y_new):
                # Si se perdió, mantenemos el anterior
                out[i] = [x_prev, y_prev, c_prev]
            else:
                # Filtro anti-jump
                if abs(x_new - x_prev) > self.max_jump or abs(y_new - y_prev) > self.max_jump:
                    out[i] = [x_prev, y_prev, c_prev]
                else:
                    x_s = self.smooth_factor * x_new + (1 - self.smooth_factor) * x_prev
                    y_s = self.smooth_factor * y_new + (1 - self.smooth_factor) * y_prev
                    out[i] = [x_s, y_s, c_new]
        self.prev_keypoints = out
        return out

    # -------------------------
    # Estimar vista (frontal/lateral/45º)
    # -------------------------
    @staticmethod
    def _bbox_width(kp):
        xs = kp[:, 0][~np.isnan(kp[:, 0])]
        return (xs.max() - xs.min()) if xs.size > 0 else np.nan

    def estimar_vista(self, kp):
        """
        Heurística refinada usando hombros, caderas y visibilidad.
        """
        # COCO indices
        L_SH, R_SH = 5, 6
        L_HIP, R_HIP = 11, 12
        L_ELB, R_ELB = 7, 8
        # ensure indices exist
        needed = [L_SH, R_SH, L_HIP, R_HIP]
        if any(idx >= kp.shape[0] for idx in needed):
            return "semi-lateral"
         
        bbox_w = self._bbox_width(kp)
        if np.isnan(bbox_w) or bbox_w <= 1:
            return "semi-lateral"
         
        xL, xR = kp[L_SH, 0], kp[R_SH, 0]
        cL, cR = kp[L_SH, 2], kp[R_SH, 2]
        if np.isnan(xL) or np.isnan(xR):
            return "lateral"
         
        shoulder_dx = abs(xL - xR) / bbox_w
        conf_diff = abs(cL - cR)
         
        # hip
        xLH, xRH = kp[L_HIP, 0], kp[R_HIP, 0]
        hip_dx = abs(xLH - xRH) / bbox_w if not (np.isnan(xLH) or np.isnan(xRH)) else shoulder_dx
         
        # elbow spread (si están ambos visibles)
        elb_dx = 0.0
        if not np.isnan(kp[L_ELB,0]) and not np.isnan(kp[R_ELB,0]):
            elb_dx = abs(kp[L_ELB,0] - kp[R_ELB,0]) / bbox_w
         
        # Left/right visibility ratio (sum conf of left vs right keypoints)
        left_conf = sum(kp[i,2] for i in [5,7,11] if i < kp.shape[0])
        right_conf = sum(kp[i,2] for i in [6,8,12] if i < kp.shape[0])
        conf_ratio = (left_conf + 1e-6) / (right_conf + 1e-6)
         
        # Heurística combinada (umbrales empíricos)
        if shoulder_dx >= 0.34 and conf_diff < 0.25 and elb_dx >= 0.24:
            return "frontal"
        if shoulder_dx <= 0.16 or conf_diff >= 0.45 or elb_dx <= 0.12 or conf_ratio > 2.5 or conf_ratio < 0.4:
            return "lateral"
        return "semi-lateral"

    # -------------------------
    # Validación de ángulo por ejercicio
    # -------------------------
    def validar_angulos(self, keypoints_3c, ejercicio: str):
        vista = self.estimar_vista(keypoints_3c)
        self.view_votes[vista] += 1
        self.frames_seen += 1
        self.decided_view = self.view_votes.most_common(1)[0][0]

        if ejercicio not in self._REQ_ANGULOS:
            return True, None

        requeridas = self._REQ_ANGULOS[ejercicio]
        if self.decided_view in requeridas:
            return True, None

        recomendado = RECOMENDACION_ANGULO.get(ejercicio, "Ajusta la cámara para una vista apropiada al ejercicio.")
        msg = f"Ángulo detectado: {self.decided_view}. Recomendación: {recomendado}"
        return False, msg

    # -------------------------
    # Verificación global (usada por main.py)
    # -------------------------
    def verificar(self, keypoints, ejercicio, secuencia_actual):
        kp = self._ensure_kp_format(keypoints)
        if kp is None:
            return False, "Keypoints con formato inválido", self.decided_view
        kp_valid = self.validar_keypoints(kp)
        visibles = np.sum(kp_valid[:, 2] >= self.min_conf)
        ratio = visibles / float(kp_valid.shape[0])
        if ratio < self.min_visible_ratio:
            return False, "Keypoints insuficientes/oclusiones (baja visibilidad)", self.decided_view
        if self.prev_keypoints is not None:
            mask = (~np.isnan(kp_valid[:, 0])) & (~np.isnan(self.prev_keypoints[:, 0]))
            if np.any(mask):
                d = np.linalg.norm(kp_valid[mask, :2] - self.prev_keypoints[mask, :2], axis=1)
                if np.nanmedian(d) > self.max_jump:
                    self.suavizar(kp_valid)
                    return False, "Salto brusco detectado; esperando estabilizar tracking", self.decided_view
        self.suavizar(kp_valid)
        ok_angle, msg = self.validar_angulos(kp_valid, ejercicio)
        if not ok_angle:
            return False, msg, self.decided_view
        return True, None, self.decided_view
    
    @staticmethod
    def validar_secuencia_fases(secuencia, fases_validas):
        if not fases_validas:
            return True
        idx = 0
        for fase in secuencia:
            if fase == fases_validas[idx]:
                idx += 1
                if idx == len(fases_validas):
                    return True
        return False
    
