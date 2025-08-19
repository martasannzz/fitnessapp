# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:10:50 2025

@author: marta
"""

import streamlit as st
import tempfile
from main import analizar_video  # función principal adaptada
import os

st.set_page_config(page_title="Análisis Técnico", layout="centered")
st.title("📈 Análisis Técnico de Ejercicio")

# Selección de ejercicio
ejercicio = st.selectbox("Selecciona el ejercicio", ["deadlift", "squat", "bench_press", "biceps_curl"])

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

