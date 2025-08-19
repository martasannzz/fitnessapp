# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:47:46 2025

@author: marta
"""

import numpy as np

# Calcula ángulo entre tres puntos
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # clip por estabilidad numérica
    return np.degrees(angle)


def detect_squat_phase(keypoints):
    hip = keypoints[11]
    knee = keypoints[13]
    ankle = keypoints[15]

    knee_angle = calculate_angle(hip, knee, ankle)

    # Umbrales simples para detectar fases, ajustar según sea necesario
    if knee_angle > 160:
        phase = "setup"
    elif 90 < knee_angle <= 160:
        phase = "eccentric"  # bajando
    elif 80 <= knee_angle <= 90:
        phase = "bottom"
    else:
        phase = "concentric"  # subiendo

    return phase, knee_angle

def check_squat(keypoints, phase=None):
    # Si no se pasa fase, la detecta internamente (compatibilidad)
    if phase is None:
        phase, knee_angle = detect_squat_phase(keypoints)
    else:
        _, knee_angle = detect_squat_phase(keypoints)
    
    hip = keypoints[11]
    knee = keypoints[13]
    ankle = keypoints[15]
    shoulder = keypoints[5]

    hip_angle = calculate_angle(shoulder, hip, knee)

    try:
        foot = keypoints[19]
        ankle_angle = calculate_angle(knee, ankle, foot)
    except:
        ankle_angle = None

    # Feedback base según fase
    if phase == "setup":
        feedback = "✅ Setup: Posición inicial correcta."
        # Feedback técnico detallado
        if knee_angle < 170:
            feedback += " Rodillas ligeramente flexionadas, extiéndelas casi por completo (sin bloquear)."
        if hip_angle < 170:
            feedback += "Cadera no totalmente extendida, revisa posición de inicio."
        if ankle_angle is not None and ankle_angle < 85:
            feedback += "Tobillos con dorsiflexión excesiva en setup."
    
    elif phase == "eccentric":
        feedback = "⬇️ Descenso controlado."
        if knee_angle < 90:
            feedback += " ✅ Profundidad adecuada (por debajo de paralela)."
        elif 90 <= knee_angle <= 110:
            feedback += "Queda justo en paralela, podría buscar mayor rango si movilidad lo permite."
        else:
            feedback += "Falta profundidad, baja más manteniendo técnica."

        if hip_angle < 100:
            feedback += "Inclinas mucho el tronco, mantén la espalda más erguida."

    elif phase == "bottom":
        feedback = "⏸️ Punto más bajo."
        if knee_angle < 80:
            feedback += "Sentadilla muy profunda, revisa estabilidad lumbar."
        elif 80 <= knee_angle <= 110:
            feedback += " Buena profundidad."
        else:
            feedback += " No alcanzas profundidad mínima recomendada."

        if hip_angle < 100:
            feedback += "Tronco muy inclinado, refuerza estabilidad lumbar y core."

    elif phase == "concentric":
        feedback = "⬆️ Subida controlada."
        if knee_angle > 160 and hip_angle > 160:
            feedback += "Finaliza la extensión correctamente."
        else:
            if knee_angle < 160:
                feedback += "No extiendes completamente las rodillas al final."
            if hip_angle < 160:
                feedback += "Falta extensión completa de cadera."

    else:
        feedback = "Ejecutando sentadilla."

    angles_info = {
        'knee_angle': int(knee_angle),
        'hip_angle': int(hip_angle),
        'ankle_angle': int(ankle_angle) if ankle_angle is not None else None
    }

    return angles_info, f"[{phase}] {feedback}"

########################################
### DEADLIFT (peso muerto)
########################################

hip_angle_history = []

def detect_deadlift_phase(keypoints):
    hip = keypoints[11]
    knee = keypoints[13]
    ankle = keypoints[15]
    shoulder = keypoints[5]

    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)

    # Guardar historial de ángulo de cadera
    hip_angle_history.append(hip_angle)
    if len(hip_angle_history) > 5:
        hip_angle_history.pop(0)

    # Determinar dirección del movimiento
    if len(hip_angle_history) >= 2:
        delta = hip_angle_history[-1] - hip_angle_history[0]
        direction = "up" if delta > 5 else "down" if delta < -5 else "stable"
    else:
        direction = "stable"

    # Clasificación de fase
    if hip_angle < 90 and knee_angle > 110:
        phase = "setup"
    elif direction == "up" and hip_angle < 150 and knee_angle < 160:
        phase = "pull"
    elif hip_angle >= 150 and knee_angle >= 160:
        phase = "lockout"
    elif direction == "down" and hip_angle < 150:
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
        feedback += "Setup inicial. "
        if hip_angle < 60 or hip_angle > 80:
            feedback += "🔴 Ajusta ángulo de cadera (ideal 60°-80°). "
        if knee_angle < 100 or knee_angle > 120:
            feedback += "🔴 Ajusta ángulo de rodilla (ideal 100°-120°). "

    elif phase == "pull":
        feedback += "Fase de tirón. "
        if hip_angle < 150:
            feedback += "🔴 Extiende más la cadera. "
        if knee_angle < 160:
            feedback += "🔴 Extiende más las rodillas. "

    elif phase == "lockout":
        feedback += "Lockout final. "
        if hip_angle < 170:
            feedback +="🔴 Falta extensión completa de cadera."
        else:
            feedback +="✅ Lockout correcto."

    elif phase == "lowering":
        feedback += "Bajando el peso. Controla el descenso. "

    angles_info = f"Hip: {int(hip_angle)}°, Knee: {int(knee_angle)}°"
    return hip_angle, feedback + angles_info

########################################
### BENCH PRESS (press de banca)
########################################

def detect_bench_press_phase(keypoints):
    shoulder = keypoints[5]
    elbow = keypoints[7]
    wrist = keypoints[9]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if elbow_angle > 160:
        phase = "setup"
    elif 70 < elbow_angle <= 160:
        phase = "lowering"
    else:
        phase = "press"

    return phase, elbow_angle

def check_bench_press(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_bench_press_phase(keypoints)
    else:
        _, elbow_angle = detect_bench_press_phase(keypoints)
    feedback = f"[{phase}] "

    if phase == "setup":
        feedback += "Setup inicial. "
        if elbow_angle < 170:
            feedback += "🔴 Extiende más los codos. "

    elif phase == "lowering":
        feedback += "Bajando la barra. "
        if elbow_angle < 90:
            feedback += "✅ Buena profundidad. "
        else:
            feedback += "🔴 Baja más la barra. "

    elif phase == "press":
        feedback += "Empujando la barra. "
        if elbow_angle > 160:
            feedback += "✅ Buena extensión final. "
        else:
            feedback += "🔴 Extiende más los codos. "

    angles_info = f"Elbow: {int(elbow_angle)}°"
    return elbow_angle, feedback + angles_info

########################################
### BICEPS CURL (curl de biceps)
########################################

def detect_biceps_curl_phase(keypoints):
    shoulder = keypoints[5]
    elbow = keypoints[7]
    wrist = keypoints[9]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if elbow_angle > 150:
        phase = "setup"
    elif 40 < elbow_angle <= 150:
        phase = "curl"
    else:
        phase = "lowering"

    return phase, elbow_angle

def check_biceps_curl(keypoints, phase=None):
    if phase is None:
        phase, elbow_angle = detect_biceps_curl_phase(keypoints)
    else:
        _, elbow_angle = detect_biceps_curl_phase(keypoints)
    feedback = f"[{phase}] "

    if phase == "setup":
        feedback += "Inicio del curl. "
        if elbow_angle < 170:
            feedback += "🔴 Extiende completamente los brazos. "

    elif phase == "curl":
        feedback += "Flexionando codo. "
        if elbow_angle < 45:
            feedback += "✅ Buena contracción de bíceps. "
        else:
            feedback += "🔴 Flexiona más el codo. "

    elif phase == "lowering":
        feedback += "Bajando el peso. Controla el descenso."

    angles_info = f"Elbow: {int(elbow_angle)}°"
    return elbow_angle, feedback + angles_info

########################################
### Mapeo general para main.py
########################################

phase_detectors = {
    "squat": detect_squat_phase,
    "deadlift": detect_deadlift_phase,
    "bench_press": detect_bench_press_phase,
    "biceps_curl": detect_biceps_curl_phase
}

exercise_functions = {
    "squat": check_squat,
    "deadlift": check_deadlift,
    "bench_press": check_bench_press,
    "biceps_curl": check_biceps_curl
}