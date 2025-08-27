# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:47:46 2025

@author: marta
"""

import numpy as np
from collections import deque

# ==============================
# UTILIDADES BASE
# ==============================

def calculate_angle(a, b, c):
    """
    angulo ABC (en grados). a, b, c son (x, y).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosine = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(angle))

def _virtual_point_above(p, dy=50):
    # punto virtual por encima (vertical en imagen)
    return np.array([p[0], p[1] - dy], dtype=float)

def angle_to_vertical(upper, joint):
    """
    angulo del segmento (joint->upper) respecto a la vertical.
    0° = alineado vertical; ~90° = horizontal.
    """
    v_ref = _virtual_point_above(joint)
    return calculate_angle(upper, joint, v_ref)

# Historiales para detectar direccion (apertura/cierre; flexion/extension)
_hist = {}
def _direction(name, value, window=4, eps=3.0):
    """
    Devuelve 'increasing', 'decreasing' o 'stable' segun el cambio
    del valor (media ventana) con un umbral eps (grados).
    """
    dq = _hist.setdefault(name, deque(maxlen=window))
    dq.append(float(value))
    if len(dq) < 2:
        return "stable"
    delta = dq[-1] - dq[0]
    if delta > eps:
        return "increasing"
    if delta < -eps:
        return "decreasing"
    return "stable"

# Helper: acceso seguro a keypoints (evita IndexError si faltan)
def _kp(keypoints, idx, fallback=None):
    try:
        return keypoints[idx]
    except Exception:
        return fallback if fallback is not None else np.array([0.0, 0.0], dtype=float)

# ==============================
# SQUAT
# ==============================

def detect_squat_phase(keypoints):
    hip = _kp(keypoints, 11); knee = _kp(keypoints, 13); ankle = _kp(keypoints, 15)
    knee_angle = calculate_angle(hip, knee, ankle)
    # Fases por rango de angulo (simple y estable)
    if knee_angle > 160:
        phase = "setup"
    elif 90 < knee_angle <= 160:
        phase = "eccentric"
    elif 80 <= knee_angle <= 90:
        phase = "bottom"
    else:
        phase = "concentric"
    return phase, knee_angle

def check_squat(keypoints, phase=None):
    if phase is None:
        phase, knee_angle = detect_squat_phase(keypoints)
    else:
        _, knee_angle = detect_squat_phase(keypoints)

    hip = _kp(keypoints, 11); knee = _kp(keypoints, 13); ankle = _kp(keypoints, 15); shoulder = _kp(keypoints, 5)
    hip_angle = calculate_angle(shoulder, hip, knee)
    foot = _kp(keypoints, 19, fallback=None)  # algunos modelos traen 19/21 pies; si no, None
    ankle_angle = calculate_angle(knee, ankle, foot) if foot is not None else None

    feedback = f"[{phase}] "
    if phase == "setup":
        feedback += "Posicion inicial. "
        if knee_angle < 170:
            feedback += "Extiende casi por completo las rodillas (sin bloquear). "
        if hip_angle < 170:
            feedback += "Falta extension completa de cadera. "
        if ankle_angle is not None and ankle_angle < 85:
            feedback += "Dorsiflexion excesiva en setup. "
    elif phase == "eccentric":
        feedback += "Descenso controlado. "
        if knee_angle < 90:
            feedback += "Profundidad adecuada. "
        elif 90 <= knee_angle <= 110:
            feedback += "≈ Paralela, puedes buscar algo mas de rango si es seguro. "
        else:
            feedback += "Falta profundidad. "
        if hip_angle < 100:
            feedback += "Tronco muy inclinado; refuerza core. "
    elif phase == "bottom":
        feedback += "Punto mas bajo. "
        if knee_angle < 80:
            feedback += "Muy profunda; cuida estabilidad lumbar. "
        elif 80 <= knee_angle <= 110:
            feedback += "Buena profundidad. "
        else:
            feedback += "No alcanzas profundidad minima. "
        if hip_angle < 100:
            feedback += "Mucha flexion de tronco. "
    elif phase == "concentric":
        feedback += "Subida. "
        if knee_angle > 160 and hip_angle > 160:
            feedback += "Final con extension completa. "
        else:
            if knee_angle < 160:
                feedback += "Extiende mas las rodillas al final. "
            if hip_angle < 160:
                feedback += "Extiende mas la cadera. "

    angles_info = {
        "knee": int(knee_angle),
        "hip": int(hip_angle),
        "ankle": int(ankle_angle) if ankle_angle is not None else None
    }
    return angles_info, feedback

# ==============================
# DEADLIFT (convencional)
# ==============================

def detect_deadlift_phase(keypoints):
    hip = _kp(keypoints, 11); knee = _kp(keypoints, 13); ankle = _kp(keypoints, 15); shoulder = _kp(keypoints, 5)
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)

    d = _direction("deadlift_hip", hip_angle)  # up/down por cadera
    if hip_angle < 90 and knee_angle > 110:
        phase = "setup"
    elif d == "increasing" and hip_angle < 150 and knee_angle < 160:
        phase = "pull"
    elif hip_angle >= 150 and knee_angle >= 160:
        phase = "lockout"
    elif d == "decreasing" and hip_angle < 150:
        phase = "lowering"
    else:
        phase = "unknown"
    return phase, hip_angle, knee_angle

def check_deadlift(keypoints, phase=None):
    if phase is None:
        phase, hip_angle, knee_angle = detect_deadlift_phase(keypoints)
    else:
        _, hip_angle, knee_angle = detect_deadlift_phase(keypoints)

    feedback = f"[{phase}] "
    if phase == "setup":
        feedback += "Setup inicial."
        if not (60 <= hip_angle <= 80):
            feedback += "Ajusta angulo de cadera (≈60°–80°)."
        if not (100 <= knee_angle <= 120):
            feedback += "Ajusta angulo de rodilla (≈100°–120°)."
    elif phase == "pull":
        feedback += "Tiron."
        if hip_angle < 150:
            feedback += "Extiende mas la cadera."
        if knee_angle < 160:
            feedback += "Extiende mas las rodillas."
    elif phase == "lockout":
        feedback += "Bloqueo final."
        if hip_angle < 170:
            feedback += "Falta extension completa de cadera."
        else:
            feedback += "Lockout correcto."
    elif phase == "lowering":
        feedback += "Descenso controlado."

    return {"hip": int(hip_angle), "knee": int(knee_angle)}, feedback

# ==============================
# ROMANIAN DEADLIFT (RDL)
# ==============================

def detect_romanian_deadlift_phase(keypoints):
    shoulder = _kp(keypoints, 5); hip = _kp(keypoints, 11); knee = _kp(keypoints, 13); ankle = _kp(keypoints, 15)
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    d = _direction("rdl_hip", hip_angle)

    if knee_angle > 160 and hip_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "eccentric"
    elif hip_angle < 100:
        phase = "bottom"
    elif d == "increasing" and hip_angle >= 150:
        phase = "lockout"
    else:
        phase = "unknown"
    return phase, hip_angle, knee_angle

def check_romanian_deadlift(keypoints, phase=None):
    if phase is None:
        phase, hip_angle, knee_angle = detect_romanian_deadlift_phase(keypoints)
    else:
        _, hip_angle, knee_angle = detect_romanian_deadlift_phase(keypoints)

    feedback = f"[{phase}] "
    if phase == "setup":
        feedback += "Posicion inicial con rodillas semirrigidas. "
    elif phase == "eccentric":
        feedback += "Lleva la cadera atras; controla el tronco. "
        if knee_angle < 160:
            feedback += "Evita flexionar mucho las rodillas. "
    elif phase == "bottom":
        feedback += "Punto mas bajo. "
        if hip_angle < 90:
            feedback += "Excesiva flexion de tronco (cuidado lumbar). "
    elif phase == "lockout":
        feedback += "Extension completa de cadera."

    return {"hip": int(hip_angle), "knee": int(knee_angle)}, feedback

# ==============================
# BENCH PRESS (plano)
# ==============================

def detect_bench_press_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("bench_elbow", elbow_angle)
    if elbow_angle > 160:
        phase = "setup"
    elif d == "decreasing":
        phase = "lowering"
    elif d == "increasing":
        phase = "press"
    else:
        phase = "lowering" if elbow_angle > 90 else "press"
    return phase, elbow_angle

def check_bench_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_bench_press_phase(keypoints)
    else:
        _, elbow_angle = detect_bench_press_phase(keypoints)

    feedback = f"[{phase}] "
    if phase == "setup":
        feedback += "Codos extendidos sin bloquear; escapulas retraidas. "
    elif phase == "lowering":
        if elbow_angle < 90:
            feedback += "Buena profundidad de bajada. "
        else:
            feedback += "Baja un poco mas manteniendo control. "
    elif phase == "press":
        if elbow_angle > 160:
            feedback += "Extension final completa. "
        else:
            feedback += "Extiende mas los codos al final. "

    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# INCLINE BENCH PRESS
# ==============================

def detect_incline_bench_press_phase(keypoints):
    return detect_bench_press_phase(keypoints)

def check_incline_bench_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_incline_bench_press_phase(keypoints)
    else:
        _, elbow_angle = detect_incline_bench_press_phase(keypoints)
    feedback = f"[incline {phase}] "
    if phase == "lowering" and elbow_angle >= 100:
        feedback += "Busca mayor rango (sin dolor). "
    elif phase == "press" and elbow_angle <= 150:
        feedback += "Completa extension final. "
    else:
        feedback += "Ejecucion adecuada. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# DECLINE BENCH PRESS
# ==============================

def detect_decline_bench_press_phase(keypoints):
    return detect_bench_press_phase(keypoints)

def check_decline_bench_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_decline_bench_press_phase(keypoints)
    else:
        _, elbow_angle = detect_decline_bench_press_phase(keypoints)
    feedback = f"[decline {phase}] "
    if phase == "lowering" and elbow_angle >= 100:
        feedback += "Podrias aumentar un poco la profundidad. "
    elif phase == "press" and elbow_angle <= 150:
        feedback += "Extiende mas los codos. "
    else:
        feedback += "✅ Control y estabilidad correctos. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# CHEST FLY (maquina)
# ==============================

def detect_chest_fly_machine_phase(keypoints):
    # Usamos separacion de manos (ambas muñecas) para apertura/cierre
    lwrist = _kp(keypoints, 9); rwrist = _kp(keypoints, 10, fallback=None)
    if rwrist is None or (lwrist == 0).all():
        # fallback: usar angulo de hombro
        shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7)
        sh_angle = angle_to_vertical(elbow, shoulder)
        d = _direction("fly_shoulder", sh_angle)
        if d == "increasing":
            phase = "open"
        elif d == "decreasing":
            phase = "close"
        else:
            phase = "open" if sh_angle > 45 else "close"
        return phase, sh_angle

    # distancia horizontal aproximada
    hand_sep = abs(float(lwrist[0]) - float(rwrist[0]))
    d = _direction("fly_sep", hand_sep)
    if d == "increasing":
        phase = "open"
    elif d == "decreasing":
        phase = "close"
    else:
        phase = "open" if hand_sep > 120 else "close"
    return phase, hand_sep

def check_chest_fly_machine(keypoints, phase=None):
    if phase is None:
        phase, metric = detect_chest_fly_machine_phase(keypoints)
    else:
        _, metric = detect_chest_fly_machine_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "open":
        feedback += "Apertura controlada con hombros estables. "
    elif phase == "close":
        feedback += "Cierra apretando el pecho al centro. "
    return {"fly_metric": int(metric)}, feedback

# ==============================
# BICEPS CURL (barra recta)
# ==============================

def detect_barbell_biceps_curl_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("barbell_curl_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "curl"
    else:
        phase = "lowering"
    return phase, elbow_angle

def check_barbell_biceps_curl(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_barbell_biceps_curl_phase(keypoints)
    else:
        _, elbow_angle = detect_barbell_biceps_curl_phase(keypoints)

    hip = _kp(keypoints, 11); shoulder = _kp(keypoints, 5)
    trunk_angle = angle_to_vertical(shoulder, hip)  # tronco vs vertical

    feedback = f"[{phase}] "
    if phase == "setup" and elbow_angle < 170:
        feedback += "Extiende por completo los codos al inicio. "
    if phase == "curl" and elbow_angle <= 45:
        feedback += "✅ Buena contraccion final. "
    if trunk_angle > 20:
        feedback += "Evita balancear el tronco. "

    return {"elbow": int(elbow_angle), "trunk": int(trunk_angle)}, feedback

# ==============================
# HAMMER CURL
# ==============================

def detect_hammer_curl_phase(keypoints):
    return detect_barbell_biceps_curl_phase(keypoints)

def check_hammer_curl(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_hammer_curl_phase(keypoints)
    else:
        _, elbow_angle = detect_hammer_curl_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "setup" and elbow_angle < 170:
        feedback += "Extiende por completo los codos. "
    if phase == "curl" and elbow_angle <= 45:
        feedback += "Buena flexion final. "
    feedback += "Manten el agarre neutro y codos pegados. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# TRICEP PUSHDOWN (polea)
# ==============================

def detect_tricep_pushdown_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("pushdown_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "setup"    # arriba, codos extendidos
    elif d == "decreasing":
        phase = "eccentric"  # flexion
    else:
        phase = "press"      # extension
    return phase, elbow_angle

def check_tricep_pushdown(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_tricep_pushdown_phase(keypoints)
    else:
        _, elbow_angle = detect_tricep_pushdown_phase(keypoints)

    # Mantener brazo vertical (hombro->codo cerca de vertical)
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7)
    upper_arm_vert = angle_to_vertical(shoulder, elbow)

    feedback = f"[{phase}] "
    if upper_arm_vert > 20:
        feedback += "Manten el brazo mas vertical/pegado al torso. "
    if phase == "press" and elbow_angle > 160:
        feedback += "Extension completa. "
    elif phase == "press":
        feedback += "Completa la extension final. "
    return {"elbow": int(elbow_angle), "upper_arm_vert": int(upper_arm_vert)}, feedback

# ==============================
# TRICEP DIPS
# ==============================

def detect_tricep_dips_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("dips_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "lowering"
    else:
        phase = "press"
    return phase, elbow_angle

def check_tricep_dips(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_tricep_dips_phase(keypoints)
    else:
        _, elbow_angle = detect_tricep_dips_phase(keypoints)

    feedback = f"[{phase}] "
    if phase == "lowering" and elbow_angle <= 80:
        feedback += "Buena profundidad. "
    if phase == "press" and elbow_angle <= 150:
        feedback += "Extiende mas al final. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# LATERAL RAISE (elevacion lateral)
# ==============================

def detect_lateral_raise_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7)
    sh_abd = angle_to_vertical(elbow, shoulder)  # elevacion del brazo
    d = _direction("latraise_sh", sh_abd)
    if sh_abd < 20:
        phase = "setup"
    elif d == "increasing":
        phase = "raise"
    else:
        phase = "lowering"
    return phase, sh_abd

def check_lateral_raise(keypoints, phase=None):
    if phase is None:
        phase, sh_abd = detect_lateral_raise_phase(keypoints)
    else:
        _, sh_abd = detect_lateral_raise_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "raise":
        if sh_abd <= 90:
            feedback += "Detente a la altura de los hombros. "
        else:
            feedback += "No subas por encima de 90°. "
    return {"shoulder_abduction": int(sh_abd)}, feedback

# ==============================
# LAT PULLDOWN (jalon al pecho)
# ==============================

def detect_lateral_pulldown_phase(keypoints):
    # similar a pull-up pero sentado (sin barra visible)
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("latpull_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "pull"
    else:
        phase = "return"
    return phase, elbow_angle

def check_lateral_pulldown(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_lateral_pulldown_phase(keypoints)
    else:
        _, elbow_angle = detect_lateral_pulldown_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "pull" and elbow_angle <= 70:
        feedback += "Buena flexion de codo (barra al pecho). "
    elif phase == "pull":
        feedback += "Tira mas hacia abajo manteniendo torso estable. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# PULL-UP (dominada)
# ==============================

def detect_pull_up_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("pullup_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "hang"
    elif d == "decreasing":
        phase = "pull"
    else:
        phase = "lowering"
    return phase, elbow_angle

def check_pull_up(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_pull_up_phase(keypoints)
    else:
        _, elbow_angle = detect_pull_up_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "pull" and elbow_angle <= 60:
        feedback += "Menton a la barra. "
    elif phase == "pull":
        feedback += "Sube mas (flexiona mas el codo). "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# PUSH-UP (flexiones de suelo)
# ==============================

def detect_push_up_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("pushup_elbow", elbow_angle)
    if elbow_angle > 160:
        phase = "setup"
    elif d == "decreasing":
        phase = "lowering"
    else:
        phase = "press"
    return phase, elbow_angle

def check_push_up(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_push_up_phase(keypoints)
    else:
        _, elbow_angle = detect_push_up_phase(keypoints)

    shoulder = _kp(keypoints, 5); hip = _kp(keypoints, 11); ankle = _kp(keypoints, 15)
    body_line = calculate_angle(shoulder, hip, ankle)  # ideal ~180°

    feedback = f"[{phase}] "
    if not (170 <= body_line <= 185):
        feedback += "Manten cuerpo en linea (evita cadera caida/alta). "
    if phase == "press" and elbow_angle > 160:
        feedback += "Extension completa. "
    elif phase == "press":
        feedback += "Extiende mas los codos. "

    return {"elbow": int(elbow_angle), "body_line": int(body_line)}, feedback

# ==============================
# SHOULDER PRESS (press militar)
# ==============================

def detect_shoulder_press_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("shoulderpress_elbow", elbow_angle)
    if elbow_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "lowering"
    else:
        phase = "press"
    return phase, elbow_angle

def check_shoulder_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_shoulder_press_phase(keypoints)
    else:
        _, elbow_angle = detect_shoulder_press_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "press" and elbow_angle > 155:
        feedback += "Extension sobre cabeza. "
    elif phase == "press":
        feedback += "Extiende mas para finalizar. "
    if phase == "lowering" and elbow_angle < 90:
        feedback += "Evita bajar demasiado si molesta el hombro. "
    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# T-BAR ROW (remo en T)
# ==============================

def detect_t_bar_row_phase(keypoints):
    hip = _kp(keypoints, 11); shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7)
    back_incline = angle_to_vertical(hip, shoulder)  # torso vs vertical
    elbow_angle = calculate_angle(shoulder, elbow, _kp(keypoints, 9))
    d = _direction("tbar_elbow", elbow_angle)
    if back_incline < 40:
        phase = "setup"  # aun no te inclinaste
    elif d == "decreasing":
        phase = "pull"
    else:
        phase = "lowering"
    return phase, back_incline, elbow_angle

def check_t_bar_row(keypoints, phase=None):
    if phase is None:
        phase, back_incline, elbow_angle = detect_t_bar_row_phase(keypoints)
    else:
        _, back_incline, elbow_angle = detect_t_bar_row_phase(keypoints)
    feedback = f"[{phase}] "
    if not (50 <= back_incline <= 80):
        feedback += "Ajusta inclinacion del torso (≈50°–80°). "
    if phase == "pull" and elbow_angle <= 70:
        feedback += "Buena retraccion escapular. "
    elif phase == "pull":
        feedback += "Tira mas, lleva codos atras. "
    return {"back_incline": int(back_incline), "elbow": int(elbow_angle)}, feedback

# ==============================
# LEG EXTENSION (extension de rodilla)
# ==============================

def detect_leg_extension_phase(keypoints):
    hip = _kp(keypoints, 11); knee = _kp(keypoints, 13); ankle = _kp(keypoints, 15)
    knee_angle = calculate_angle(hip, knee, ankle)
    d = _direction("legext_knee", knee_angle)
    if knee_angle < 100:
        phase = "setup"    # pierna flexionada
    elif d == "increasing":
        phase = "extend"
    else:
        phase = "return"
    return phase, knee_angle

def check_leg_extension(keypoints, phase=None):
    if phase is None:
        phase, knee_angle = detect_leg_extension_phase(keypoints)
    else:
        _, knee_angle = detect_leg_extension_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "extend" and knee_angle > 160:
        feedback += "Extension completa. "
    elif phase == "extend":
        feedback += "Extiende un poco mas la rodilla. "
    return {"knee": int(knee_angle)}, feedback

# ==============================
# LEG RAISES (elevacion de piernas)
# ==============================

def detect_leg_raises_phase(keypoints):
    shoulder = _kp(keypoints, 5); hip = _kp(keypoints, 11); knee = _kp(keypoints, 13)
    hip_angle = calculate_angle(shoulder, hip, knee)  # cadera
    d = _direction("legraises_hip", hip_angle)
    if hip_angle > 150:
        phase = "setup"
    elif d == "decreasing":
        phase = "raise"
    else:
        phase = "lowering"
    return phase, hip_angle

def check_leg_raises(keypoints, phase=None):
    if phase is None:
        phase, hip_angle = detect_leg_raises_phase(keypoints)
    else:
        _, hip_angle = detect_leg_raises_phase(keypoints)
    feedback = f"[{phase}] "
    if phase == "raise" and hip_angle < 90:
        feedback += "Elevacion adecuada (>90°). "
    elif phase == "raise":
        feedback += "Eleva mas las piernas manteniendo lumbar pegada. "
    return {"hip": int(hip_angle)}, feedback

# ==============================
# RUSSIAN TWIST
# ==============================

def detect_russian_twist_phase(keypoints):
    hip = _kp(keypoints, 11); shoulder = _kp(keypoints, 5); wrist = _kp(keypoints, 9)
    torso_rot = calculate_angle(hip, shoulder, wrist)  # proxy de rotacion
    d = _direction("rtwist_torso", torso_rot)
    if d == "increasing":
        phase = "rotate_right"   # convencion arbitraria
    elif d == "decreasing":
        phase = "rotate_left"
    else:
        phase = "neutral"
    return phase, torso_rot

def check_russian_twist(keypoints, phase=None):
    if phase is None:
        phase, torso_rot = detect_russian_twist_phase(keypoints)
    else:
        _, torso_rot = detect_russian_twist_phase(keypoints)
    feedback = f"[{phase}] "
    if torso_rot > 40:
        feedback += "Buena amplitud de rotacion. "
    else:
        feedback += "Aumenta la rotacion manteniendo control del core. "
    return {"torso_rotation": int(torso_rot)}, feedback

# ==============================
# PLANK
# ==============================

def detect_plank_phase(_keypoints):
    return "hold", 0.0

def check_plank(keypoints, phase=None):
    shoulder = _kp(keypoints, 5); hip = _kp(keypoints, 11); ankle = _kp(keypoints, 15)
    body_line = calculate_angle(shoulder, hip, ankle)  # ideal ~180°
    feedback = "[hold] "
    if 170 <= body_line <= 185:
        feedback += "Linea recta (core activo). "
    else:
        feedback += "Ajusta la cadera para mantener alineacion. "
    return {"body_line": int(body_line)}, feedback


# ==============================
# HIP THRUST
# ==============================
def detect_hip_thrust_phase(keypoints):
    shoulder = _kp(keypoints, 5)
    hip = _kp(keypoints, 11)
    knee = _kp(keypoints, 13)
    ankle = _kp(keypoints, 15)

    hip_angle = calculate_angle(shoulder, hip, knee)   # apertura de cadera
    knee_angle = calculate_angle(hip, knee, ankle)     # flexión rodilla

    d = _direction("hipthrust_hip", hip_angle)

    if hip_angle < 90:
        phase = "bottom"
    elif d == "increasing":
        phase = "concentric"
    elif d == "decreasing":
        phase = "eccentric"
    elif hip_angle >= 160:
        phase = "lockout"
    else:
        phase = "unknown"

    return phase, hip_angle, knee_angle


def check_hip_thrust(keypoints, phase=None):
    if phase is None:
        phase, hip_angle, knee_angle = detect_hip_thrust_phase(keypoints)
    else:
        _, hip_angle, knee_angle = detect_hip_thrust_phase(keypoints)

    feedback = f"[{phase}] "

    if phase == "bottom":
        feedback += "Posicion inicial con cadera baja. "
        if hip_angle > 100:
            feedback += "Desciende un poco más para mayor rango. "

    elif phase == "concentric":
        feedback += "Subida activando gluteos. "
        if hip_angle < 150:
            feedback += "Empuja mas fuerte hasta extender la cadera. "

    elif phase == "lockout":
        feedback += "Extension completa arriba. "
        if hip_angle < 170:
            feedback += "Extiende totalmente la cadera y aprieta gluteos. "
        if not (85 <= knee_angle <= 100):
            feedback += f"Ajusta rodillas: ideal aprox. 90 grados (actual: {int(knee_angle)}grados). "

    elif phase == "eccentric":
        feedback += "Descenso controlado manteniendo tension. "

    return {"hip": int(hip_angle), "knee": int(knee_angle)}, feedback


# ==============================
# SHOULDER PRESS
# ==============================
def detect_shoulder_press_phase(keypoints):
    shoulder = _kp(keypoints, 5); elbow = _kp(keypoints, 7); wrist = _kp(keypoints, 9)

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    d = _direction("shoulderpress_elbow", elbow_angle)

    if elbow_angle > 160:
        phase = "setup"
    elif d == "decreasing":
        phase = "lowering"
    elif d == "increasing":
        phase = "press"
    else:
        phase = "press" if elbow_angle < 150 else "setup"

    return phase, elbow_angle


def check_shoulder_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_shoulder_press_phase(keypoints)
    else:
        _, elbow_angle = detect_shoulder_press_phase(keypoints)

    feedback = f"[{phase}] "

    if phase == "setup":
        feedback += "Codos extendidos arriba; controla la postura. "
    elif phase == "lowering":
        feedback += "Bajada controlada. "
        if elbow_angle > 100:
            feedback += "Lleva los codos un poco mas abajo para rango completo. "
    elif phase == "press":
        feedback += "Empuje vertical. "
        if elbow_angle < 160:
            feedback += "Extiende completamente los codos arriba. "

    return {"elbow": int(elbow_angle)}, feedback

# ==============================
# MAPEOS A MAIN
# ==============================

phase_detectors = {
    "squat": detect_squat_phase,
    "deadlift": detect_deadlift_phase,
    "romanian deadlift": detect_romanian_deadlift_phase,
    "rdl": detect_romanian_deadlift_phase,
    "hip thrust": detect_hip_thrust_phase,

    "bench press": detect_bench_press_phase,
    "incline bench press": detect_incline_bench_press_phase,
    "decline bench press": detect_incline_bench_press_phase,
    "chest fly machine": detect_chest_fly_machine_phase,

    "barbell biceps curl": detect_barbell_biceps_curl_phase,
    "hammer curl": detect_hammer_curl_phase,

    "tricep pushdown": detect_tricep_pushdown_phase,
    "tricep dips": detect_tricep_dips_phase,

    "lateral raise": detect_lateral_raise_phase,
    "lateral pulldown": detect_lateral_pulldown_phase,
    "pull up": detect_pull_up_phase,

    "push up": detect_push_up_phase,

    "t bar row": detect_t_bar_row_phase,

    "leg_extension": detect_leg_extension_phase,
    "leg raises": detect_leg_raises_phase,

    "russian twist": detect_russian_twist_phase,
    "plank": detect_plank_phase,
    
    "shoulder press": detect_shoulder_press_phase,}

exercise_functions = {
    "squat": check_squat,
    "deadlift": check_deadlift,
    "romanian deadlift": check_romanian_deadlift,
    "rdl": check_romanian_deadlift,
    "hip thrust": check_hip_thrust,

    "bench press": check_bench_press,
    "incline bench press": check_incline_bench_press,
    "decline bench press": check_decline_bench_press,
    "chest fly machine": check_chest_fly_machine,

    "barbell biceps curl": check_barbell_biceps_curl,
    "hammer curl": check_hammer_curl,

    "tricep pushdown": check_tricep_pushdown,
    "tricep dips": check_tricep_dips,

    "lateral raise": check_lateral_raise,
    "lateral pulldown": check_lateral_pulldown,
    "pull up": check_pull_up,

    "push up": check_push_up,

    "t bar row": check_t_bar_row,

    "leg_extension": check_leg_extension,
    "leg raises": check_leg_raises,

    "russian twist": check_russian_twist,
    "plank": check_plank,
    
    "shoulder press": check_shoulder_press,}
