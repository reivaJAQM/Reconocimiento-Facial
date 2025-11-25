import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import face_recognition
import numpy as np
import base64
import json
import os
import threading

# --- CONFIGURACIÓN ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
DB_FILE = "usuarios_db.json"
BACKEND_CAMARA = cv2.CAP_DSHOW 

# --- FUNCIONES BASE DE DATOS ---
def cargar_db():
    if not os.path.exists(DB_FILE): return {}
    try:
        with open(DB_FILE, 'r') as f: return json.load(f)
    except json.JSONDecodeError: return {}

def guardar_db(db):
    with open(DB_FILE, 'w') as f: json.dump(db, f, indent=4)

def array_a_base64(np_array):
    return base64.b64encode(np_array.tobytes()).decode('utf-8')

def base64_a_array(b64_str):
    return np.frombuffer(base64.b64decode(b64_str), dtype=np.float64)


# --- CLASE PRINCIPAL ---
class FacialAuthApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de Ventana
        self.title("BioAccess - Seguridad")
        self.geometry("1100x700")
        self.minsize(800, 600)
        self.after(200, lambda: self.state('zoomed'))
        self.resizable(True, True)

        # Tipografías Globales
        self.main_font = "Segoe UI"
        self.font_title = ctk.CTkFont(family=self.main_font, size=26, weight="bold")
        self.font_button = ctk.CTkFont(family=self.main_font, size=20, weight="bold")
        self.font_text = ctk.CTkFont(family=self.main_font, size=16)
        self.font_status = ctk.CTkFont(family=self.main_font, size=18, weight="bold")

        # Variables de Estado
        self.db = cargar_db()
        self.cap = None 
        self.modo_actual = "IDLE"
        self.current_face_encoding = None
        self.nombre_registro_temp = ""
        self.active_video_label = None 

        # Layout Principal
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)

        # Cargar Recursos e Iniciar
        self.cargar_recursos()
        self.setup_welcome_view()
        self.video_loop()

    def cargar_recursos(self):
        """Carga la imagen del escudo una sola vez."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "escudo.png")
        
        # Intentamos cargar la imagen. Si falla, las variables quedarán en None.
        try:
            pil_image = Image.open(image_path)
            self.icon_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(160, 160))
            self.icon_image_small = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(80, 80))
        except Exception as e:
            print(f"Advertencia: No se pudo cargar escudo.png ({e})")
            self.icon_image = None
            self.icon_image_small = None

    def limpiar_frame(self):
        """Elimina todos los widgets de la pantalla actual."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # ===================================================================
    # VISTAS (PANTALLAS)
    # ===================================================================

    def setup_welcome_view(self):
        self.limpiar_frame()
        self.apagar_camara()
        self.modo_actual = "IDLE"

        # Contenedor centrado
        center_box = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        center_box.place(relx=0.5, rely=0.5, anchor="center")

        # Icono (Solo si existe la imagen)
        if self.icon_image:
            lbl_icon = ctk.CTkLabel(center_box, text="", image=self.icon_image)
            lbl_icon.pack(pady=(0, 40))

        # Botones Principales
        btn_reg = ctk.CTkButton(center_box, text="Crear Nueva Cuenta", width=300, height=64, corner_radius=32,
                                font=self.font_button, fg_color="#34495e", hover_color="#2c3e50",
                                command=self.setup_register_view)
        btn_reg.pack(pady=15)

        btn_log = ctk.CTkButton(center_box, text="Acceder al Sistema", width=300, height=64, corner_radius=32,
                                font=self.font_button, fg_color="#2ecc71", hover_color="#27ae60",
                                command=self.setup_login_view)
        btn_log.pack(pady=15)

    def setup_register_view(self):
        self.limpiar_frame()
        self.modo_actual = "IDLE"

        # Tarjeta Central
        self.register_card = ctk.CTkFrame(self.main_frame, width=500, corner_radius=20, fg_color="#1e1e1e") 
        self.register_card.place(relx=0.5, rely=0.5, anchor="center")

        lbl_title = ctk.CTkLabel(self.register_card, text="Nuevo Usuario", font=self.font_title)
        lbl_title.pack(pady=(30, 20))

        self.entry_nombre = ctk.CTkEntry(self.register_card, placeholder_text="Ingresa tu nombre completo", 
                                         width=350, height=40, font=self.font_text)
        self.entry_nombre.pack(pady=10)

        # Contenedor de Video Pequeño
        self.camera_container = ctk.CTkFrame(self.register_card, width=400, height=300, fg_color="black")
        self.camera_container.pack(pady=20)
        self.camera_container.pack_propagate(False)

        self.small_video_label = ctk.CTkLabel(self.camera_container, text="Cámara Apagada", text_color="gray", font=self.font_text)
        self.small_video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Botones de Acción
        self.btn_accion_registro = ctk.CTkButton(self.register_card, text="Continuar", width=350, height=50, 
                                                 corner_radius=25, fg_color="#34495e", font=self.font_button,
                                                 command=self.validar_y_encender_registro)
        self.btn_accion_registro.pack(pady=(0, 20))

        btn_back = ctk.CTkButton(self.register_card, text="Cancelar", fg_color="transparent", text_color="gray", 
                                 hover_color="#2b2b2b", font=self.font_text, command=self.setup_welcome_view)
        btn_back.pack(pady=(0, 20))

    def setup_login_view(self):
        if not self.db: return
        self.limpiar_frame()
        
        card = ctk.CTkFrame(self.main_frame, width=500, corner_radius=20, fg_color="#1e1e1e")
        card.place(relx=0.5, rely=0.5, anchor="center")

        lbl_title = ctk.CTkLabel(card, text="Acceso al Sistema", font=self.font_title)
        lbl_title.pack(pady=(30, 5))
        
        lbl_subtitle = ctk.CTkLabel(card, text="Escáner biométrico activo", text_color="gray", font=self.font_text)
        lbl_subtitle.pack(pady=(0, 20))

        # Contenedor de Video Pequeño
        self.camera_container = ctk.CTkFrame(card, width=400, height=300, fg_color="black")
        self.camera_container.pack(pady=10)
        self.camera_container.pack_propagate(False)

        self.small_video_label = ctk.CTkLabel(self.camera_container, text="Iniciando...", text_color="gray")
        self.small_video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_status_login = ctk.CTkLabel(card, text="🔍 Buscando rostro...", font=self.font_status)
        self.lbl_status_login.pack(pady=20)

        self.btn_back_login = ctk.CTkButton(card, text="Cancelar", fg_color="transparent", 
                                            text_color="gray", hover_color="#2b2b2b", font=self.font_text,
                                            command=self.setup_welcome_view)
        self.btn_back_login.pack(pady=(0, 20))

        self.active_video_label = self.small_video_label
        self.encender_camara()
        self.modo_actual = "LOGIN"

    # ===================================================================
    # LÓGICA DE NEGOCIO (REGISTRO Y LOGIN)
    # ===================================================================

    def validar_y_encender_registro(self):
        nombre = self.entry_nombre.get().strip()
        if not nombre: return
        if nombre in self.db:
            self.entry_nombre.configure(placeholder_text="¡Usuario ya existe!", text_color="red")
            return

        self.nombre_registro_temp = nombre
        self.entry_nombre.configure(state="disabled")
        
        self.active_video_label = self.small_video_label
        self.encender_camara()
        self.modo_actual = "REGISTRO"

        self.btn_accion_registro.configure(text="📷 CAPTURAR FOTO", fg_color="orange", hover_color="#d35400", command=self.ejecutar_captura)

    def ejecutar_captura(self):
        if self.current_face_encoding is None: return
        
        # Validar duplicados
        for u, b64 in self.db.items():
            saved = base64_a_array(b64)
            if face_recognition.compare_faces([saved], self.current_face_encoding, tolerance=0.5)[0]:
                self.small_video_label.configure(image=None, text=f"ERROR:\nRostro duplicado de\n{u}")
                return

        # Guardar datos
        self.db[self.nombre_registro_temp] = array_a_base64(self.current_face_encoding)
        guardar_db(self.db)
        
        # Guardar foto visual para confirmación
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.foto_confirmacion = Image.fromarray(rgb_frame)
        
        self.apagar_camara()
        self.mostrar_confirmacion_registro()

    def mostrar_confirmacion_registro(self):
        """Reemplaza la tarjeta de registro con la confirmación de éxito."""
        for widget in self.register_card.winfo_children(): widget.destroy()

        lbl_check = ctk.CTkLabel(self.register_card, text="✅", font=ctk.CTkFont(size=60))
        lbl_check.pack(pady=(20, 10))

        lbl_title = ctk.CTkLabel(self.register_card, text="Registro Exitoso", font=self.font_title, text_color="#2ecc71")
        lbl_title.pack(pady=(0, 10))

        if hasattr(self, 'foto_confirmacion'):
            img_resized = self.foto_confirmacion.resize((320, 240), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(320, 240))
            lbl_foto = ctk.CTkLabel(self.register_card, text="", image=ctk_img, corner_radius=10)
            lbl_foto.pack(pady=10)

        lbl_nombre = ctk.CTkLabel(self.register_card, text=f"Usuario: {self.nombre_registro_temp.upper()}", font=self.font_text)
        lbl_nombre.pack(pady=10)

        btn_finalizar = ctk.CTkButton(self.register_card, text="Finalizar", width=300, height=50, 
                                      corner_radius=25, fg_color="#2ecc71", hover_color="#27ae60", font=self.font_button,
                                      command=self.setup_welcome_view)
        btn_finalizar.pack(pady=20)

    def verificar_login_thread(self):
        """Corre en segundo plano para no congelar la GUI."""
        found = None
        for u, b64 in self.db.items():
            saved = base64_a_array(b64)
            if face_recognition.compare_faces([saved], self.current_face_encoding, tolerance=0.5)[0]:
                found = u
                break
        # Volvemos al hilo principal para actualizar la GUI
        self.after(0, lambda: self.resultado_login(found))

    def resultado_login(self, nombre):
        if nombre:
            self.apagar_camara()
            self.modo_actual = "IDLE"
            self.lbl_status_login.configure(text=f"🔓 ¡BIENVENIDO, {nombre.upper()}!", text_color="#2ecc71")
            self.btn_back_login.configure(text="Cerrar Sesión", fg_color="#c0392b", text_color="white", hover_color="#e74c3c")
        else:
            self.lbl_status_login.configure(text="⛔ NO RECONOCIDO", text_color="#ff5555")
            self.modo_actual = "LOGIN"

    # ===================================================================
    # CONTROL DE HARDWARE (CÁMARA)
    # ===================================================================

    def encender_camara(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, BACKEND_CAMARA)

    def apagar_camara(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def video_loop(self):
        """Bucle infinito que actualiza el video."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. Optimización: Detectar en imagen pequeña
                small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb_small)
                locs = [(t*4, r*4, b*4, l*4) for (t, r, b, l) in locs]

                self.current_face_encoding = None
                color = (0, 255, 0)
                
                # 2. Calcular Encoding si hay rostro
                if locs:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb, locs)
                    if encs:
                        self.current_face_encoding = encs[0]

                # 3. Lógica por Modo
                if self.modo_actual == "REGISTRO": 
                    color = (255, 165, 0)
                elif self.modo_actual == "LOGIN": 
                    color = (0, 191, 255)
                    if self.current_face_encoding is not None:
                        self.modo_actual = "BUSCANDO"
                        threading.Thread(target=self.verificar_login_thread, daemon=True).start()

                # 4. Dibujar Recuadros
                for (t, r, b, l) in locs:
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)

                # 5. Renderizar en GUI
                if self.active_video_label is not None:
                    frame_show = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb_frame)

                    # Ajuste de tamaño para las tarjetas
                    if self.active_video_label == getattr(self, 'small_video_label', None):
                        img_pil = img_pil.resize((400, 300), Image.Resampling.LANCZOS)
                        
                    ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
                    try: self.active_video_label.configure(image=ctk_img, text="")
                    except: pass 
        
        self.after(20, self.video_loop)

if __name__ == "__main__":
    app = FacialAuthApp()
    app.mainloop()