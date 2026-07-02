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
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main-title {
            text-align: center;
            color: #2E7D32;
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        
        .subtitle {
            text-align: center;
            color: #558B2F;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .upload-box {
            border: 2px dashed #81C784;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background-color: #F1F8E9;
            transition: all 0.3s ease;
        }
        
        .result-card {
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
            border-radius: 20px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 8px 32px rgba(46, 125, 50, 0.15);
            border: 1px solid #A5D6A7;
        }
        
        .winner-badge {
            display: inline-block;
            background: linear-gradient(135deg, #2E7D32, #43A047);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.3rem;
            box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
            margin-bottom: 1rem;
        }
        
        .confidence-text {
            color: #1B5E20;
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        .prob-bar-bg {
            background-color: #E0E0E0;
            border-radius: 10px;
            height: 28px;
            margin: 8px 0;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            padding-left: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            transition: width 1s ease-out;
        }
        
        .flower-icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .footer {
            text-align: center;
            color: #9E9E9E;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #EEEEEE;
        }
        
        .stFileUploader > div > div {
            background-color: #F1F8E9 !important;
            border: 2px dashed #81C784 !important;
            border-radius: 15px !important;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #4CAF50 !important;
            background-color: #E8F5E9 !important;
        }
        
        div[data-testid="stImage"] img {
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO
# ============================================================
st.markdown('<div class="flower-icon">🌺</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Clasificador de Flores con IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IA-ISC • Campus Comayagua • 2026 • Richard Andino •  20231900184</p>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #616161; margin-bottom: 2rem;">
        🌿 Sube una imagen de una flor y la inteligencia artificial la identificará al instante
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

# Colores para cada flor
FLOWER_COLORS = {
    "daisy": "#FFCA28",
    "dandelion": "#FFD54F",
    "rose": "#E53935",
    "sunflower": "#FFB300",
    "tulip": "#E91E63"
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
with st.spinner("🔄 Cargando modelo de inteligencia artificial..."):
    modelo = cargar_modelo()
    clases = cargar_clases()

# ============================================================
# UPLOADER
# ============================================================
st.markdown("---")
archivo = st.file_uploader(
    "📷 Selecciona una imagen de flor",
    type=["jpg", "jpeg", "png"],
    help="Formatos permitidos: JPG, JPEG, PNG"
)

if archivo:
    # Mostrar imagen cargada
    imagen = Image.open(archivo)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(imagen, caption="🖼️ Imagen analizada", use_container_width=True)
    
    # Realizar predicción
    with st.spinner("🔍 Analizando imagen con IA..."):
        resultados = predecir(imagen)
    
    ganador_key, ganador_nombre, ganador_prob = resultados[0]
    
    # ============================================================
    # RESULTADO PRINCIPAL
    # ============================================================
    icono = FLOWER_ICONS.get(ganador_key, "🌸")
    color = FLOWER_COLORS.get(ganador_key, "#4CAF50")
    
    st.markdown(f"""
        <div class="result-card">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icono}</div>
                <div class="winner-badge">{ganador_nombre}</div>
        </div>
    """, unsafe_allow_html=True)


else:
    # Estado inicial cuando no hay imagen
    st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #9E9E9E;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
            <p style="font-size: 1.1rem;">Arrastra o selecciona una imagen de flor<br>para comenzar la clasificación</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                🌼 Margarita &nbsp;•&nbsp; 🌾 Diente de León &nbsp;•&nbsp; 🌹 Rosa<br>
                🌻 Girasol &nbsp;•&nbsp; 🌷 Tulipán
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        🌿 Clasificador de Flores con MobileNetV2 • Proyecto IA-ISC 2026 
    </div>
""", unsafe_allow_html=True)
