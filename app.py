# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:10:50 2025

@author: marta
"""

import streamlit as st
import tempfile
from main import analizar_video
import os
from pipeline_utils import RECOMENDACION_ANGULO

st.set_page_config(page_title="Análisis Técnico", layout="centered")
st.title("📈 Análisis Técnico de Ejercicio")

EJERCICIOS = [
    "deadlift", "romanian deadlift", "squat", "bench press", "biceps curl",
    "incline bench press", "decline bench press", "chest fly machine", "barbell biceps curl",
    "hammer curl", "tricep pushdown", "tricep dips", "lateral raise", "lateral pulldown",
    "pull up", "push up", "t bar row", "leg_extension", "leg raises", "russian twist", "plank", "hip thrust", "shoulder press"
]

# Selección de ejercicio
ejercicio = st.selectbox("Selecciona el ejercicio", EJERCICIOS, index=0)

# Mostrar recomendación de ángulo al seleccionar
reco = RECOMENDACION_ANGULO.get(
    ejercicio,
    "Coloca la cámara de forma que se vean bien hombros, caderas y piernas."
)
st.info(f"🎥 Ángulo recomendado para **{ejercicio.title()}**: {reco}")

# Subida de video
video = st.file_uploader("Sube tu video", type=["mp4", "mov", "avi"])

# Botón para iniciar análisis
if video is not None and st.button("🔍 Analizar"):
    with st.spinner("⏳ Procesando el video..."):
        # Crear archivo temporal para guardar el video subido
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video.read())
            tmp_path = tmp.name

        # Ejecutar análisis
        feedback_df, resumen, output_path, feedback_text = analizar_video(tmp_path, ejercicio, usar_gpt = True)

    st.success("✅ Análisis completado")

    st.subheader("📋 Feedback técnico del ejercicio")
    st.write(resumen)


    # (Opcional) Botón para descargar el video analizado
    with open(output_path, 'rb') as f:
        st.download_button("⬇️ Descargar video analizado", f, file_name="analisis_feedback.mp4")

