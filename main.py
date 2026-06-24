import os
import io
import torch
import requests
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Doctor de Cultivos API - Backend Render")

# Configurar CORS para recibir peticiones desde tu Frontend en Angular o local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LISTA COMPLETA DE LAS 38 CLASES EN EL ORDEN ESTÁNDAR DE PLANTVILLAGE
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# 2. CONFIGURACIÓN Y DESCARGA AUTOMÁTICA DEL MODELO DESDE TU LINK DE DRIVE
MODEL_PATH = "plant_disease_mobilenet_v3_final.pt"
DRIVE_FILE_ID = "1g8Cv2-uX_ixbOMtIi6ncNFTVb89suGpA" # 🎯 Tu ID de archivo vinculado con éxito

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: 
                f.write(chunk)

if not os.path.exists(MODEL_PATH):
    print("📥 Descargando modelo optimizado TorchScript desde tu Google Drive...")
    try:
        download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
        print("✅ Descarga completa y exitosa.")
    except Exception as e:
        print(f"❌ Error al descargar el modelo: {e}")

# Cargar el modelo de forma segura en la CPU (Render gratuito)
device = torch.device("cpu")
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print("🤖 Modelo de Red Neuronal cargado en memoria con éxito.")
except Exception as e:
    print(f"❌ Error crítico al levantar el modelo TorchScript: {e}")

# Transformaciones de imagen necesarias para MobileNetV3
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. DICCIONARIO DE CONFIGURACIÓN ESTÁTICA (Mapeo de tratamientos en Español)
TRATAMIENTOS = {
    "Tomato___Late_blight": {
        "nombre_comun": "Tizón Tardío del Tomate (Hongo: Phytophthora infestans)",
        "tratamiento": "Eliminar inmediatamente hojas y frutos afectados para frenar la espora, evitar por completo el riego por aspersión y aplicar fungicidas protectores como Oxicloruro de Cobre o Mancozeb."
    },
    "Apple___Apple_scab": {
        "nombre_comun": "Sarna de la Manzana (Hongo: Venturia inaequalis)",
        "tratamiento": "Aplicar fungicidas a base de cobre de forma preventiva y retirar de inmediato los residuos de hojas caídas infectadas."
    }
}

# 4. ENDPOINT QUE SE CONECTARÁ CON TU FORMULARIO EN ANGULAR
@app.post("/diagnosticar")
async def diagnosticar(
    lat: float = Form(...),
    lon: float = Form(...),
    file: UploadFile = File(...)
):
    # --- Clima simulado o por defecto requerido para las reglas ---
    humedad = 85
    clima_actual = "Rain" 

    # --- Inferencia de la Imagen por la Red Neuronal ---
    contenido_imagen = await file.read()
    imagen = Image.open(io.BytesIO(contenido_imagen)).convert("RGB")
    imagen_tensor = transform_val(imagen).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(imagen_tensor)
        _, preds = torch.max(outputs, 1)
        plaga_detectada = CLASS_NAMES[preds.item()]
    
    # --- Ejecución de la Regla Matemática de Negocio ---
    perdida_soles = 0
    # Regla: Si el hongo es agresivo y las condiciones climáticas son óptimas para su propagación
    if plaga_detectada == "Tomato___Late_blight" and humedad > 80 and clima_actual == "Rain":
        perdida_soles = 5000

    # --- Cruzar con el Diccionario de Configuración ---
    info_tratamiento = TRATAMIENTOS.get(
        plaga_detectada, 
        {
            "nombre_comun": f"Plaga Detectada ({plaga_detectada.replace('___', ' - ')})", 
            "tratamiento": "De momento no se registra un tratamiento específico en el diccionario estático. Consultar con un ingeniero agrónomo local."
        }
    )

    # --- RESPUESTA JSON ENVIADA AL FRONTEND ---
    return {
        "plaga": info_tratamiento["nombre_comun"],
        "clase_tecnica": plaga_detectada,
        "clima": {
            "humedad": humedad,
            "condicion": clima_actual
        },
        "regla_negocio": {
            "alerta_riesgo": perdida_soles > 0,
            "perdida_soles": perdida_soles
        },
        "tratamiento": info_tratamiento["tratamiento"]
    }