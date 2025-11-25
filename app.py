from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
import json
import base64
import atexit # Para cerrar la cámara correctamente al salir

app = Flask(__name__)

# --- CONFIGURACIÓN ---
DB_FILE = "usuarios_db.json"
camera = None # No la iniciamos globalmente todavía

# Variables Globales
current_face_encoding = None

# --- FUNCIONES BD ---
def cargar_db():
    if not os.path.exists(DB_FILE): return {}
    try:
        with open(DB_FILE, 'r') as f: return json.load(f)
    except: return {}

def guardar_db(db):
    with open(DB_FILE, 'w') as f: json.dump(db, f, indent=4)

def array_a_base64(np_array):
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def base64_a_array(b64_str):
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float64)

# --- GESTIÓN INTELIGENTE DE CÁMARA ---
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        # Intentamos abrir la cámara con backend DSHOW (más compatible en Windows)
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return camera

def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None

# Aseguramos que la cámara se apague si el servidor se cae
atexit.register(release_camera)

# --- GENERADOR DE VIDEO ---
def generar_frames():
    global current_face_encoding
    cam = get_camera() # Obtenemos la cámara aquí, bajo demanda
    
    while True:
        success, frame = cam.read()
        if not success:
            # Si falla, intentamos reiniciarla una vez
            cam.release()
            cam = get_camera()
            success, frame = cam.read()
            if not success:
                break
        
        # 1. Procesamiento
        frame = cv2.flip(frame, 1)
        
        # 2. Detección (Optimizada)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_locations = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in face_locations]

        current_face_encoding = None
        
        if face_locations:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if encodings:
                current_face_encoding = encodings[0]

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 3. Enviar a la web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- RUTAS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/registro', methods=['POST'])
def registro():
    data = request.json
    user_id = data.get('id')       # Número de identificación
    password = data.get('password') # Contraseña
    
    # Validaciones básicas
    if not user_id or not password: 
        return jsonify({"status": "error", "msg": "Faltan datos (ID o Contraseña)"})
    
    db = cargar_db()
    
    # Verificar si el ID ya existe
    if user_id in db: 
        return jsonify({"status": "error", "msg": "Este número de identificación ya está registrado"})
    
    # GUARDAR DATOS (Texto plano por ahora, en prod usaríamos Hashing como bcrypt)
    # Estructura: ID -> {password, fecha_creacion, etc}
    db[user_id] = {
        "password": password,
        "face_encoding": None # Dejamos el espacio listo para agregar la cara después
    }
    
    guardar_db(db)
    return jsonify({"status": "success", "msg": f"Usuario {user_id} creado correctamente"})

@app.route('/api/login', methods=['POST'])
def login():
    global current_face_encoding
    if current_face_encoding is None: return jsonify({"status": "waiting", "msg": "Buscando..."})
    
    db = cargar_db()
    for u, b64 in db.items():
        saved = base64_a_array(b64)
        if face_recognition.compare_faces([saved], current_face_encoding, tolerance=0.5)[0]:
            return jsonify({"status": "success", "msg": u})
            
    return jsonify({"status": "error", "msg": "No reconocido"})

if __name__ == '__main__':
    # TRUCO IMPORTANTE: use_reloader=False evita que la cámara se abra dos veces
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)