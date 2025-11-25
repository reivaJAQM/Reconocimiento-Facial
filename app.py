from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os
import json
import base64
import atexit

app = Flask(__name__)
app.secret_key = 'super_secreta_clave_flask'

# --- CONFIGURACIÓN ---
DB_FILE = "usuarios_db.json"
camera = None 
current_face_encoding = None

# --- FUNCIONES ---
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

# --- CÁMARA ---
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    return camera

def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None

atexit.register(release_camera)

def generar_frames():
    global current_face_encoding
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small)
        current_face_encoding = None
        if locs:
            encs = face_recognition.face_encodings(rgb_small, locs)
            if encs: current_face_encoding = encs[0]
            for (t, r, b, l) in locs:
                cv2.rectangle(frame, (l*4, t*4), (r*4, b*4), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- RUTAS ---
@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('index'))
    user_id = session['user_id']
    db = cargar_db()
    
    # Construimos el nombre completo uniendo Nombre + Apellido
    user_data = db.get(user_id, {})
    nombre = user_data.get('nombre', '')
    apellido = user_data.get('apellido', '')
    nombre_completo = f"{nombre} {apellido}".strip()
    
    # Si por alguna razón están vacíos, mostramos el ID
    if not nombre_completo: nombre_completo = user_id
    
    tiene_rostro = True if user_data.get('face_encoding') else False
        
    return render_template('dashboard.html', user=nombre_completo, has_face=tiene_rostro)

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- APIS ---
@app.route('/api/registro', methods=['POST'])
def registro():
    data = request.json
    user_id = data.get('id')
    nombre = data.get('nombre')    # Nuevo campo
    apellido = data.get('apellido') # Nuevo campo
    password = data.get('password')
    
    if not user_id or not password or not nombre or not apellido: 
        return jsonify({"status": "error", "msg": "Faltan datos (Llene todos los campos)"})
    
    db = cargar_db()
    if user_id in db: return jsonify({"status": "error", "msg": "Usuario ya existe"})
    
    # Guardamos por separado
    db[user_id] = {
        "nombre": nombre,
        "apellido": apellido,
        "password": password, 
        "face_encoding": None
    }
    guardar_db(db)
    return jsonify({"status": "success", "msg": "Usuario creado correctamente"})

@app.route('/api/login_step1', methods=['POST'])
def login_step1():
    data = request.json
    user_id = data.get('id')
    password = data.get('password')
    db = cargar_db()
    
    if user_id in db and db[user_id]['password'] == password:
        # Recuperamos datos para saludar
        user_data = db[user_id]
        n = user_data.get('nombre', '')
        a = user_data.get('apellido', '')
        nombre_real = f"{n} {a}".strip() or user_id
        
        if db[user_id].get('face_encoding'):
            session['pre_login_id'] = user_id
            return jsonify({"status": "partial", "msg": "Validación requerida"})
        else:
            session['user_id'] = user_id
            return jsonify({"status": "success", "msg": f"Bienvenido, {nombre_real}"})
            
    return jsonify({"status": "error", "msg": "Credenciales incorrectas"})

@app.route('/api/login_step2_face', methods=['POST'])
def login_step2_face():
    global current_face_encoding
    if 'pre_login_id' not in session: return jsonify({"status": "error", "msg": "Sesión expirada"})
    if current_face_encoding is None: return jsonify({"status": "waiting", "msg": "Buscando..."})
    
    target_user = session['pre_login_id']
    db = cargar_db()
    rostro_guardado = base64_a_array(db[target_user]['face_encoding'])
    
    if face_recognition.compare_faces([rostro_guardado], current_face_encoding, tolerance=0.5)[0]:
        session['user_id'] = target_user
        session.pop('pre_login_id', None)
        return jsonify({"status": "success", "msg": "Identidad verificada"})
    
    return jsonify({"status": "waiting", "msg": "No coincide"})

@app.route('/api/registrar_rostro', methods=['POST'])
def registrar_rostro():
    global current_face_encoding
    if 'user_id' not in session: return jsonify({"status": "error", "msg": "Error de sesión"})
    if current_face_encoding is None: return jsonify({"status": "error", "msg": "No se detecta rostro"})
    db = cargar_db()
    db[session['user_id']]['face_encoding'] = array_a_base64(current_face_encoding)
    guardar_db(db)
    return jsonify({"status": "success", "msg": "Rostro registrado"})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)