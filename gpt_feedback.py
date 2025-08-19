# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:29:47 2025

@author: marta
"""

import openai
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env.")

# Crear cliente
openai.api_key = api_key

response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """
                        Eres un entrenador personal virtual experto en biomecánica y técnica de ejercicios. Tu tarea es analizar los datos de ejecución de un ejercicio (incluyendo repetición, fase, ángulos articulares y feedbacks detectados) para generar un análisis claro, útil y motivador.
                        Tu estilo debe ser como si estuvieras al lado del usuario en el gimnasio: directo, claro, sin tecnicismos innecesarios y centrado en ayudarle a mejorar.
                        
                        Sigue estas reglas:
                        - Si la técnica es sólida, reconócelo. No señales errores si no los hay.
                        - Si hay errores, identifícalos claramente, pero sin exagerar. Explica en qué momento se cometen (si se sabe) y cómo corregirlos con consejos prácticos.
                        - Si hay patrones de mejora a lo largo de las repeticiones, menciónalo.
                        - Da recomendaciones realistas, fáciles de entender y aplicar.
                        - No repitas feedback de forma innecesaria si ya se entiende el problema.
                        
                        Evita tecnicismos clínicos. No hagas suposiciones si no hay suficiente información. Sé claro, motivador y profesional.
                        """
        },
        {
            "role": "user",
            "content": "Este es el resultado del análisis de keypoints de un {ejercicio}: {datos}"
        }
    ],
    max_tokens=500
)

# ✅ Mostrar respuesta del entrenador inteligente
print(response.choices[0].message.content)