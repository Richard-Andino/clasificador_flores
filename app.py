import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Clasificación de Flores IA_ISC",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS PERSONALIZADO (NUEVO DISEÑO PREMIUM DARK)
# ============================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
        
        /* Ajuste de Fuente Global */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Space Grotesk', sans-serif;
        }
        
        /* Fondo General de la App en Modo Oscuro */
        .stApp {
            background-color: #0F0F1A;
            color: #E2E8F0;
        }
        
        /* Título Principal */
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #A855F7 0%, #6366F1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
            letter-spacing: -1px;
        }
        
        /* Subtítulo */
        .subtitle {
            text-align: center;
            color: #94A3B8;
            font-size: 0.95rem;
            margin-bottom: 2rem;
            font-weight: 300;
            letter-spacing: 0.5px;
        }
        
        /* Tarjeta de Resultado Principal (Glassmorphism) */
        .result-card {
            background: rgba(30, 30, 50, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 12px 40px rgba(124, 58, 237, 0.2);
            border: 1px solid rgba(168, 85, 247, 0.3);
            text-align: center;
        }
        
        /* Badge del Ganador */
        .winner-badge {
            display: inline-block;
            background: linear-gradient(135deg, #7C3AED, #4F46E5);
            color: white;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.5rem;
            box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }
        
        /* Texto de Confianza */
        .confidence-text {
            color: #A78BFA;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Contenedor de Barras de Probabilidad */
        .prob-container {
            background: rgba(15, 15, 26, 0.6);
            padding: 1rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .prob-row {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }

        .prob-label {
            width: 30%;
            text-align: left;
            font-weight: 500;
            color: #CBD5E1;
        }

        .prob-bar-bg {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            height: 24px;
            width: 70%;
            overflow: hidden;
            position: relative;
        }
        
        .prob-bar-fill {
            height: 100%;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.8rem;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Icono Principal flotante */
        .flower-icon {
            font-size: 3.5rem;
            text-align: center;
            margin-bottom: -0.5rem;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        /* Footer Reestilizado */
        .footer {
            text-align: center;
            color: #64748B;
            font-size: 0.8rem;
            margin-top: 4rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Rediseño del Drag & Drop File Uploader */
        .stFileUploader > div > div {
            background-color: rgba(30, 30, 50, 0.4) !important;
            border: 2px dashed rgba(168, 85, 247, 0.4) !important;
            border-radius: 20px !important;
            color: #E2E8F0 !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #6366F1 !important;
            background-color: rgba(49, 46, 129, 0.2) !important;
        }

        /* Estilo para las imágenes subidas */
        div[data-testid="stImage"] img {
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO (CON TU NOMBRE Y NUEVO ESTILO)
# ============================================================
st.markdown('<div class="flower-icon">🔮</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Clasificador de Flores Inteligente</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IA-ISC • Campus Comayagua • 2026 • Richard Andino • 20231900184</p>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #94A3B8; margin-bottom: 2rem; font-size: 1.05rem;">
        Despliega el poder de la red <b>MobileNetV2</b> para reconocer especies de flores en tiempo real.
    </div>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================
IMG_SIZE = (224, 224)
MODEL_DIR = Path("Clasificacion_flores_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [MODEL_DIR / "Clasificacion_flores_mobilenet.keras", MODEL_DIR / "flores_mobilenet.h5"]

LABELS_ES = {
    "daisy": "Margarita",
    "dandelion": "Diente de León",
    "rose": "Rosa",
    "sunflower": "Girasol",
    "tulip": "Tulipán"
}

FLOWER_ICONS = {
    "daisy": "🌼",
    "dandelion": "🌾",
    "rose": "🌹",
    "sunflower": "🌻",
    "tulip": "🌷"
}

# Paleta de colores Neon/Premium para las barras de progreso
FLOWER_COLORS = {
    "daisy": "linear-gradient(90deg, #F59E0B, #10B981)",
    "dandelion": "linear-gradient(90deg, #FBBF24, #F59E0B)",
    "rose": "linear-gradient(90deg, #EF4444, #EC4899)",
    "sunflower": "linear-gradient(90deg, #F59E0B, #EAB308)",
    "tulip": "linear-gradient(90deg, #EC4899, #8B5CF6)"
}

@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("❌ No se encontró el modelo. Coloque la carpeta `Clasificacion_flores_mobilenet` junto a `app.py`.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    top3 = np.argsort(preds)[-3:][::-1]
    return [
        (clases[i], LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in top3
    ]


# ============================================================
# CARGAR MODELO
# ============================================================
with st.spinner("⚡ Sintonizando visión artificial de alta precisión..."):
    modelo = cargar_modelo()
    clases = cargar_clases()

# ============================================================
# UPLOADER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
archivo = st.file_uploader(
    "📷 Sube o arrastra la imagen aquí",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if archivo:
    imagen = Image.open(archivo)
    
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.image(imagen, caption="🔮 Entrada del sistema", use_container_width=True)
    
    with st.spinner("🧠 Computando mapa de características..."):
        resultados = predecir(imagen)
    
    ganador_key, ganador_nombre, ganador_prob = resultados[0]
    icono = FLOWER_ICONS.get(ganador_key, "🌸")
    
    # ============================================================
    # PANEL DE RESULTADOS REESTILIZADO
    # ============================================================
    st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 4.5rem; margin-bottom: -0.5rem;">{icono}</div>
            <div class="winner-badge">{ganador_nombre}</div>
            <div class="confidence-text">Confianza del Sistema: {ganador_prob:.2f}%</div>
            
            <div class="prob-container">
                <div style="text-align: left; font-size: 0.9rem; color: #94A3B8; margin-bottom: 0.5rem; font-weight: 600;">
                    TOP 3 PROBABILIDADES:
                </div>
    """, unsafe_allow_html=True)

    # Renderizado dinámico de las 3 predicciones top
    for key, nombre, prob in resultados:
        color_bar = FLOWER_COLORS.get(key, "linear-gradient(90deg, #6366F1, #A855F7)")
        icon_small = FLOWER_ICONS.get(key, "✨")
        st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label">{icon_small} {nombre}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {prob}%; background: {color_bar};">
                        {prob:.1f}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    # Estado inicial vacío (Dashboard vacío elegante)
    st.markdown("""
        <div style="text-align: center; padding: 4rem 1rem; color: #64748B; border: 1px dashed rgba(255,255,255,0.05); border-radius: 24px; background: rgba(255,255,255,0.01);">
            <div style="font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.6;">🛸</div>
            <p style="font-size: 1.1rem; color: #94A3B8;">Esperando archivo de imagen para inicializar diagnóstico...</p>
            <p style="font-size: 0.85rem; margin-top: 1rem; color: #475569;">
                Soporte para: Margarita • Diente de León • Rosa • Girasol • Tulipán
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        🧬 Powered by TensorFlow & MobileNetV2 • Arquitectura de Software 2026 • Desarrollado por Richard Andino
    </div>
""", unsafe_allow_html=True)
