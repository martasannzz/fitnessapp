# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 20:42:43 2025

@author: marta
"""

from ultralytics import YOLO
import cv2
import pandas as pd
from feedback_rules import exercise_functions, phase_detectors
from dotenv import load_dotenv
import os
import torch
from openai import OpenAI
from pipeline_utils import PipelineVerificacion
import json
import numpy as np


# --- Parche para PyTorch 2.6 ---
_old_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Fuerza carga completa del checkpoint
    return _old_load(*args, **kwargs)
torch.load = _patched_load
# -------------------------------

RECOMMENDED_VIEWS_DICT = {
    "squat": {"lateral", "semi-lateral"},
    "deadlift": {"lateral", "semi-lateral"},
    "romanian deadlift": {"lateral", "semi-lateral"},
    "bench press": {"lateral", "semi-lateral"},
    "incline bench press": {"lateral", "semi-lateral"},
    "decline bench press": {"lateral", "semi-lateral"},
    "lateral pulldown": {"frontal", "semi-lateral"},
    "pull up": {"frontal", "semi-lateral"},
    "lateral raise": {"frontal", "semi-lateral"},
    "russian twist": {"frontal", "semi-lateral"},
    "chest fly machine": {"frontal", "semi-lateral"},
    "biceps curl": {"lateral", "semi-lateral"},
    "barbell biceps curl": {"lateral", "semi-lateral"},
    "hammer curl": {"lateral", "semi-lateral"},
    "tricep pushdown": {"lateral", "semi-lateral"},
    "tricep dips": {"lateral", "semi-lateral"},
    "push up": {"lateral", "semi-lateral"},
    "plank": {"lateral", "semi-lateral"},
    "leg_extension": {"lateral", "semi-lateral"},
    "leg raises": {"lateral", "semi-lateral"},
    "t bar row": {"lateral", "semi-lateral"},
    "hip thrust": {"lateral", "semi-lateral"},
    "shoulder press": {"frontal", "semi-lateral"},
}

def analizar_video(video_path, ejercicio, usar_gpt=True):
    """
    Analiza un video de entrenamiento aplicando YOLOv8-pose,
    verifica los keypoints con el pipeline y finalmente evalúa
    con las reglas biomecánicas definidas.
    """
    # -------------------------------
    # ✅ 0. Cargar variables de entorno y API key
    # -------------------------------
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")
    client = OpenAI(api_key=api_key)
    
    # -------------------------------
    # ✅ 1. Definir prompt del sistema (entrenador GPT)
    # -------------------------------
    SYSTEM_PROMPT = """Eres un entrenador personal experto en biomecánica y técnica de ejercicios. Tu estilo es claro, cercano y motivador, como si estuvieras entrenando a alguien en el gimnasio.
                Tu objetivo es ayudar a mejorar la técnica de forma práctica, honesta y útil, sin buscar errores donde no los hay. No hagas un analisis por fase o repetición si no es necesario.
                
                Sigue estas pautas:
                - Si la técnica es correcta, reconócelo claramente. No inventes fallos si no los hay.
                - Si hay errores, identifica solo los importantes. Evita ser excesivamente crítico.
                - Explica por qué un punto es bueno o malo, y cómo corregirlo de forma sencilla si es necesario.
                - No repitas fases o repeticiones de forma mecánica. Resume lo importante como lo harías hablando con alguien.
                - Si hay cosas que se pueden mejorar pero no son graves, trátalas como recomendaciones suaves, no como fallos.
                
                Finaliza con un resumen general si hay patrones, y con consejos realistas para seguir progresando.
                
                No uses tecnicismos innecesarios ni des diagnósticos clínicos. No inventes datos si no están disponibles.
                """
    
    # -------------------------------
    # 2. Inicializar modelo y vídeo
    # -------------------------------
    model = YOLO('yolov8m-pose.pt')
    verificador = PipelineVerificacion()
    
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = cv2.VideoWriter('output_feedback.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if not cap.isOpened():
        print("❌ Error: no se pudo abrir el vídeo.")
        return None, None, None
    else:
        print("✅ Vídeo cargado correctamente.")
    
    # -------------------------------
    # Variables para seguimiento
    # -------------------------------
    feedback_storage = []
    debug_feedback_storage = []
    current_repetition = 0
    last_counted_phase = None
    rep_phase_sequence = []

    
     # ---------- variables de estabilidad ----------
    frame_idx = 0
    last_keypoints = None
    stable_idx = None
    stability_threshold = 80  # distancia máxima aceptada entre frames
    # ----------------------------------------------


    valid_phase_sequences = {
        "squat": ["setup","eccentric","bottom","concentric"],
        "deadlift": ["setup","pull","lockout","lowering"],
        "romanian deadlift": ["setup","eccentric","bottom","lockout"],
        "bench press": ["setup","lowering","press"],
        "incline bench press": ["setup","lowering","press"],
        "decline bench press": ["setup","lowering","press"],
        "chest fly machine": ["open","close"],
        "barbell biceps curl": ["setup","curl","lowering"],
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
    
    valid_sequence = valid_phase_sequences.get(ejercicio, [])
    
    # -------------------------------
    # 3. Procesar frames del vídeo
    # -------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
    
        results = model(frame, verbose = False)
        
        if not results or not hasattr(results[0], "keypoints") or results[0].keypoints is None:
            msg = "No se detecta persona"
            cv2.putText(frame, f"⚠️ {msg}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            output_video_path.write(frame)

            # ✅ (4.4) Guardamos este feedback en debug_feedback_storage
            debug_feedback_storage.append({
                "exercise": ejercicio,
                "phase": None,
                "repetition": current_repetition,
                "feedback": msg,
                "angle": None,
                "decided_view": None
            })
            continue

        if len(results[0].keypoints.xy) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints_all = results[0].keypoints.xy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # ---------- Filtro de estabilidad ----------
            if stable_idx is None:
                # Primer frame:  elegir mayor área
                stable_idx = areas.argmax()
            else:
                # Comparar distancia de keypoints con el último sujeto seguido
                distancias = []
                for kp in keypoints_all:
                    if last_keypoints is None:
                        distancias.append(np.inf)
                    else:
                        # nan_to_num para evitar NaNs en cálculo
                        distancias.append(np.linalg.norm(np.nan_to_num(kp - last_keypoints)))
                candidato_idx = int(np.argmin(distancias))
                if distancias[candidato_idx] < stability_threshold:
                    stable_idx = candidato_idx
                else:
                    # mantener stable_idx anterior (más estable)
                    pass

            # Protección si stable_idx está fuera de rango por alguna razón
            if stable_idx < 0 or stable_idx >= len(keypoints_all):
                stable_idx = int(areas.argmax())

            keypoints = keypoints_all[stable_idx]
            last_keypoints = keypoints.copy()
            
            # ---- Inicializar variables por frame ----
            phase = "unknown"
            angle = None
            feedback = ""   # feedback de las reglas (no GPT)
            msg = None      # mensaje de verificación (tracking/ángulo)
            decided_view = None
            # ------------------------------------------------------------------


            # Verificación pipeline
            valid, msg, decided_view = verificador.verificar(keypoints, ejercicio, rep_phase_sequence)

            # Nota sobre ángulo de cámara
            angle_note = ""
            recommended_views = RECOMMENDED_VIEWS_DICT.get(ejercicio, set())
            if decided_view and decided_view not in recommended_views:
                angle_note = f"Nota: El ángulo detectado de cámara es **{decided_view}**, " \
                             f"pero se recomienda {', '.join(recommended_views)}."

            # Detección de fase y reglas del ejercicio
            if ejercicio in phase_detectors:
                phase_info = phase_detectors[ejercicio](keypoints)
                phase = phase_info[0] if isinstance(phase_info, (list, tuple)) and len(phase_info) > 0 else "unknown"
                if ejercicio in exercise_functions:
                    angle_calc, feedback_calc = exercise_functions[ejercicio](keypoints, phase)
                    angle = angle_calc
                    if not msg:
                        feedback = feedback_calc

            # Conteo de repeticiones
            if phase != "unknown" and phase != last_counted_phase:
                last_counted_phase = phase
                if not rep_phase_sequence or phase != rep_phase_sequence[-1]:
                    rep_phase_sequence.append(phase)
                if all(p in rep_phase_sequence for p in valid_sequence):
                    valid_indices = [rep_phase_sequence.index(p) for p in valid_sequence]
                    if valid_indices == sorted(valid_indices):
                        current_repetition += 1
                        rep_phase_sequence = []

            stored_feedback = msg if msg else feedback
            # Guardar feedback SIEMPRE
            feedback_storage.append({
                'exercise': ejercicio,
                'phase': phase,
                'repetition': current_repetition,
                'feedback': msg if msg else feedback,
                'angle': angle,
                'decided_view': decided_view
            })

            # Dibujar keypoints y feedback
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            label = f"{ejercicio.title()} Rep: {current_repetition} [{phase}]"
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if feedback:
                cv2.putText(frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if angle_note:
                cv2.putText(frame, angle_note, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            if not valid and msg:
                cv2.putText(frame, f"⚠️ {msg}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        else:
            cv2.putText(frame, "No se detecta persona", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            feedback_storage.append({
                'exercise': ejercicio,
                'phase': "unknown",
                'repetition': current_repetition,
                'feedback': "No se detecta persona",
                'angle': None,
                'decided_view': None
            })

        output_video_path.write(frame)
        cv2.imshow('Exercise Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video_path.release()
    cv2.destroyAllWindows()

    # Guardar feedback en JSON para debug
    with open("debug_feedback_storage.json", "w", encoding="utf-8") as f:
        json.dump(feedback_storage, f, ensure_ascii=False, indent=4)

    feedback_df = pd.DataFrame(feedback_storage)
    
    # -------------------------------
    # 4. Preparar resumen con GPT
    # -------------------------------
    if usar_gpt and not feedback_df.empty:
        agrupado = (
            feedback_df.groupby(['repetition', 'phase'])
            .agg({
                'feedback': lambda x: ' '.join(set(x)),
                'angle': lambda x: [a for a in x if a is not None]
            })
            .reset_index()
        )
    
        num_reps = len(agrupado['repetition'].unique())
    
        if num_reps == 1:
            rep_text = ("Este análisis corresponde a 1 única repetición del ejercicio. "
                        "Por favor, genera un feedback directo, claro y en singular, "
                        "como si estuvieras evaluando un único intento completo. "
                        "No uses expresiones como 'a veces', 'en algunas repeticiones', "
                        "'la mayoría de las veces', 'en ciertos momentos' ni nada que implique "
                        "variabilidad o múltiples intentos. "
                        "Aunque el análisis esté dividido en fases, resume el feedback como un "
                        "solo evento continuo y no hagas referencia a repeticiones múltiples.")
        else:
            rep_text = f"Este análisis corresponde a {num_reps} repeticiones del ejercicio. Las repeticiones han sido divididas por fases, junto con datos de feedback y ángulos articulares."
        
        resumen_agrupado = []
        for rep in sorted(agrupado['repetition'].unique()):
            rep_data = agrupado[agrupado['repetition'] == rep]
            resumen_agrupado.append({
                'repeticion': int(rep),
                'fases': rep_data['phase'].tolist(),
                'feedbacks': rep_data['feedback'].tolist(),
                'angulos': rep_data['angle'].tolist()
            })
        

        decided_views_in_video = set(entry['decided_view'] for entry in feedback_storage if entry.get('decided_view') is not None)

        angle_note_prompt = ""
        recommended_views = RECOMMENDED_VIEWS_DICT.get(ejercicio, set())
        for v in decided_views_in_video:
            if v not in recommended_views:
                angle_note_prompt = f"Nota: El ángulo detectado de cámara es **{v}**, " \
                                    f"pero para este ejercicio se recomienda {', '.join(recommended_views)}. " \
                                    f"Esto podría afectar la precisión del análisis técnico."
                break

    
        prompt_usuario = f"""
                        Hola entrenador. Este es el análisis técnico del ejercicio "{ejercicio}". {rep_text}.
                        
                        Tu tarea es dar feedback como si fueras un entrenador experimentado observando el ejercicio completo. No me interesa un análisis por fases. Quiero que:
                        Por favor, si es 1 repetición, **no uses plurales ni expresiones de variabilidad**. Trata toda la repetición como un único intento.
                            
                        - Expliques qué se está haciendo bien, si hay algo destacable.
                        - Detectes los errores técnicos importantes, diciendo en qué fase ocurre si es relevante.
                        - Expliques por qué importa corregirlo y cómo se puede mejorar, con consejos prácticos y claros.
                        
                        {angle_note_prompt}
                        
                        Ve al grano. No estructures por repetición ni por fase. No des tecnicismos ni explicaciones largas. El objetivo es que el usuario entienda qué está haciendo mal y cómo corregirlo para mejorar su técnica.
                        
                        Aquí tienes los datos:
                        {resumen_agrupado}
                        """
    
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario}
            ],
            max_tokens=800,
            temperature=0.7
        )
    
        feedback_text = response.choices[0].message.content.strip()
        print("\n📋 Feedback técnico del ejercicio completo:")
        print(feedback_text)
    else:
        feedback_text = "No se generó feedback GPT debido a datos insuficientes o desactivación."
    
    return feedback_df, feedback_text, 'output_feedback.mp4', feedback_text