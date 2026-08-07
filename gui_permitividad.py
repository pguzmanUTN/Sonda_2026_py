"""
gui_permitividad.py
--------------------
Interfaz grafica (Tkinter) para configurar y correr
`analisis_permitividad.py` sin tener que editar el .py a mano cada vez.

Toda la logica "de fondo" (guardar/cargar configuracion, correr el
analisis en un hilo aparte, listar los modelos teoricos disponibles,
etc.) vive en `gui_funciones.py`; este archivo solo arma la ventana y
conecta los botones con esas funciones.

Para arrancar la GUI: `python main_gui.py`.
"""
import os
import io
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import gui_funciones as gf

# matplotlib ya queda forzado al backend 'Agg' por gui_funciones.py (se
# importa arriba, y ese modulo lo hace ANTES de importar analisis_
# permitividad). Estos imports son para generar la vista previa rapida de
# S11 (pestaña "Vista previa S11"): se arman las figuras en memoria (sin
# guardar .png a disco, a diferencia del resto del proyecto) y se
# muestran directamente en la ventana.
import matplotlib.pyplot as plt
from Touchstone import leer_s1p, graficar_s11_mag_fase, graficar_smith

try:
    from PIL import Image, ImageTk
    _HAY_PIL = True
except ImportError:
    _HAY_PIL = False


TITULO_APP = "Analizador de Permitividad - Sonda Coaxial (NPL MAT 23)"
ANCHO_PREVIEW = 560
ANCHO_PREVIEW_S11 = 480  # resguardo si el panel todavia no tiene tamaño real asignado

# Etiquetas legibles para los patrones de calibracion, para la pestaña de
# "Vista previa S11" (analogo a _ETIQUETAS_CALIBRACION de gui_funciones.py,
# pero esa es privada y esta la necesitamos aca para armar el combobox).
_ETIQUETAS_CALIBRACION_PREVIEW = {
    'corto': "Cortocircuito",
    'aire': "Aire",
    'agua': "Agua",
    'alc_isoprop': "Alcohol isopropilico (patron adicional)",
}


def _generar_preview_s11(ruta_s1p, log_x=True):
    """
    Lee un archivo .s1p y arma, EN MEMORIA (sin guardar ningun .png a
    disco), dos imagenes PIL con el S11 medido: modulo/fase y diagrama de
    Smith. Se usa para la vista previa rapida de la pestaña "Vista previa
    S11", antes de correr el analisis completo (que si guarda .png).

    Devuelve (imagen_modulo_fase, imagen_smith, datos), donde `datos` es
    el dict que devuelve `Touchstone.leer_s1p` (por si el llamador quiere
    mostrar la cantidad de puntos / rango de frecuencias).
    """
    datos = leer_s1p(ruta_s1p)
    nombre = datos.get('Nombre') or os.path.splitext(os.path.basename(ruta_s1p))[0]
    datos_dict = {nombre: datos}

    fig_mf = graficar_s11_mag_fase(
        datos_dict, titulo=f"{nombre}: S11 medido (modulo y fase)", log_x=log_x)
    buf_mf = io.BytesIO()
    fig_mf.savefig(buf_mf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig_mf)
    buf_mf.seek(0)
    imagen_mf = Image.open(buf_mf)
    imagen_mf.load()  # decodificar ya mismo: el buffer se descarta al salir

    fig_sm = graficar_smith(datos_dict, titulo=f"{nombre}: S11 medido (diagrama de Smith)")
    buf_sm = io.BytesIO()
    fig_sm.savefig(buf_sm, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig_sm)
    buf_sm.seek(0)
    imagen_sm = Image.open(buf_sm)
    imagen_sm.load()

    return imagen_mf, imagen_sm, datos


# ===========================================================================
# Tooltip minimo (sin dependencias externas)
# ===========================================================================
class _Tooltip:
    """Un globito de ayuda simple que aparece al pasar el mouse por
    encima de un widget."""

    def __init__(self, widget, texto):
        self.widget = widget
        self.texto = texto
        self._toplevel = None
        widget.bind("<Enter>", self._mostrar)
        widget.bind("<Leave>", self._ocultar)

    def _mostrar(self, _event=None):
        if self._toplevel is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._toplevel = tk.Toplevel(self.widget)
        self._toplevel.wm_overrideredirect(True)
        self._toplevel.wm_geometry(f"+{x}+{y}")
        etiqueta = tk.Label(
            self._toplevel, text=self.texto, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("", 9), wraplength=320, padx=6, pady=4,
        )
        etiqueta.pack()

    def _ocultar(self, _event=None):
        if self._toplevel is not None:
            self._toplevel.destroy()
            self._toplevel = None


def _tooltip(widget, texto):
    _Tooltip(widget, texto)


# ===========================================================================
# Dialogo: agregar/editar un material
# ===========================================================================
class DialogoMaterial(tk.Toplevel):
    """Ventana modal para agregar o editar una fila de la lista de
    materiales. El resultado queda en `self.resultado` (dict) si se
    confirma con "Guardar", o en None si se cancela."""

    def __init__(self, master, material=None):
        super().__init__(master)
        self.resultado = None
        material = material or {
            'nombre': '', 'archivo': '', 'modelo': gf.SIN_MODELO,
            'temperatura': 25.0, 'metodos': ['simplificado', 'completo'],
        }

        self.title("Agregar material" if material.get('archivo') == '' and
                   material.get('nombre') == '' else f"Editar: {material['nombre']}")
        self.resizable(False, False)
        self.transient(master)

        frm = ttk.Frame(self, padding=14)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Nombre:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_nombre = tk.StringVar(value=material['nombre'])
        ttk.Entry(frm, textvariable=self.var_nombre, width=36).grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(frm, text="Archivo .s1p:").grid(row=1, column=0, sticky="w", pady=4)
        self.var_archivo = tk.StringVar(value=material['archivo'])
        ttk.Entry(frm, textvariable=self.var_archivo, width=36).grid(
            row=1, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="Examinar...", command=self._examinar_archivo).grid(
            row=1, column=2, padx=(6, 0))

        ttk.Label(frm, text="Modelo teorico:").grid(row=2, column=0, sticky="w", pady=4)
        self.var_modelo_etiqueta = tk.StringVar(value=gf.etiqueta_de_modelo(material.get('modelo')))
        self.combo_modelo = ttk.Combobox(
            frm, textvariable=self.var_modelo_etiqueta, state="readonly",
            values=[et for _, et in gf.listar_modelos_teoricos()], width=33)
        self.combo_modelo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        self.combo_modelo.bind("<<ComboboxSelected>>", self._al_cambiar_modelo)

        ttk.Label(frm, text="Temperatura (°C):").grid(row=3, column=0, sticky="w", pady=4)
        self.var_temperatura = tk.DoubleVar(value=float(material.get('temperatura') or 25.0))
        self.spin_temperatura = ttk.Spinbox(
            frm, from_=-20, to=150, increment=0.5, textvariable=self.var_temperatura, width=10)
        self.spin_temperatura.grid(row=3, column=1, sticky="w", pady=4)
        _tooltip(self.spin_temperatura,
                 "El modelo teorico usa la temperatura tabulada mas cercana\n"
                 "(pasos de 5°C, NPL MAT 23). No hace falta que coincida\n"
                 "exactamente con un escalon de la tabla.")

        ttk.Label(frm, text="Metodos a calcular:").grid(row=4, column=0, sticky="nw", pady=4)
        metodos = material.get('metodos') or ['simplificado', 'completo']
        self.var_simplificado = tk.BooleanVar(value='simplificado' in metodos)
        self.var_completo = tk.BooleanVar(value='completo' in metodos)
        frm_metodos = ttk.Frame(frm)
        frm_metodos.grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Checkbutton(frm_metodos, text="Simplificado (corto + aire + agua)",
                        variable=self.var_simplificado).pack(anchor="w")
        ttk.Checkbutton(frm_metodos, text="Completo (+ patron adicional)",
                        variable=self.var_completo).pack(anchor="w")

        ttk.Separator(frm).grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        frm_botones = ttk.Frame(frm)
        frm_botones.grid(row=6, column=0, columnspan=3, sticky="e")
        ttk.Button(frm_botones, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(frm_botones, text="Guardar", command=self._guardar).pack(side="right")

        self._al_cambiar_modelo()
        self.bind("<Return>", lambda _e: self._guardar())
        self.bind("<Escape>", lambda _e: self.destroy())

        self.update_idletasks()
        self.grab_set()
        self.focus_set()

    def _examinar_archivo(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo .s1p", parent=self,
            filetypes=[("Touchstone .s1p", "*.s1p"), ("Todos los archivos", "*.*")])
        if ruta:
            self.var_archivo.set(ruta)
            if not self.var_nombre.get().strip():
                base = os.path.splitext(os.path.basename(ruta))[0]
                self.var_nombre.set(base)

    def _al_cambiar_modelo(self, _event=None):
        activo = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get()) != gf.SIN_MODELO
        self.spin_temperatura.configure(state=("normal" if activo else "disabled"))

    def _guardar(self):
        nombre = self.var_nombre.get().strip()
        if not nombre:
            messagebox.showwarning("Falta el nombre", "Poné un nombre para el material.", parent=self)
            return
        if not self.var_archivo.get().strip():
            messagebox.showwarning("Falta el archivo", "Elegí el archivo .s1p de este material.", parent=self)
            return
        metodos = []
        if self.var_simplificado.get():
            metodos.append('simplificado')
        if self.var_completo.get():
            metodos.append('completo')
        if not metodos:
            messagebox.showwarning("Sin metodo", "Marca al menos un metodo (simplificado y/o completo).", parent=self)
            return
        try:
            temperatura = float(self.var_temperatura.get())
        except (tk.TclError, ValueError):
            temperatura = 25.0
        self.resultado = {
            'nombre': nombre,
            'archivo': self.var_archivo.get().strip(),
            'modelo': gf.clave_de_etiqueta(self.var_modelo_etiqueta.get()),
            'temperatura': temperatura,
            'metodos': metodos,
        }
        self.destroy()


# ===========================================================================
# Ventana principal
# ===========================================================================
class AppPermitividad(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITULO_APP)
        self.geometry("1100x780")
        self.minsize(960, 680)

        self.ruta_config_actual = None
        self._snapshot_guardado = None
        self._materiales = []  # lista de dicts (fuente de verdad para la tabla)
        self._cola_analisis = None
        self._analisis_corriendo = False
        self._evento_cancelar = None
        self._figuras_disponibles = {}  # {etiqueta: ruta_png}, se llena tras correr
        self._imagen_preview_tk = None  # referencia viva para que Tk no la libere
        self._pil_figura_seleccionada = None  # imagen PIL original, para reescalar si se agranda la ventana
        self._reescalado_preview_id = None  # id de after() pendiente (debounce)

        self._rutas_preview_s11 = {}  # {etiqueta: ruta_completa}, ver _refrescar_lista_preview_s11
        self._imagen_preview_s11_magfase_tk = None
        self._imagen_preview_s11_smith_tk = None
        self._pil_s11_magfase = None  # imagen PIL original (sin escalar), para poder
        self._pil_s11_smith = None    # reescalar de nuevo si se agranda la ventana
        self._reescalado_preview_s11_id = None  # id de after() pendiente (debounce)

        self._construir_estilos()
        self._construir_menu()
        self._construir_layout()
        self._construir_statusbar()

        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.bind_all("<Control-s>", lambda _e: self._guardar_config())
        self.bind_all("<Control-o>", lambda _e: self._abrir_config())
        self.bind_all("<Control-n>", lambda _e: self._nueva_config())
        self.bind_all("<Control-r>", lambda _e: self._ejecutar_analisis())

        # Auto-cargar la ultima configuracion usada, si hay una (para no
        # tener que volver a cargar todo cada vez que se abre la GUI).
        ruta = gf.ruta_ultima_config()
        cargada = False
        if ruta:
            try:
                self._cargar_config_desde(ruta)
                cargada = True
            except Exception:
                pass
        if not cargada:
            self._cargar_config_en_gui(gf.config_default())
            self._snapshot_guardado = self._config_desde_gui()

        self._actualizar_titulo()

    # -----------------------------------------------------------------
    # Estilos
    # -----------------------------------------------------------------
    def _construir_estilos(self):
        style = ttk.Style(self)
        for tema in ("clam", "alt", "default"):
            try:
                style.theme_use(tema)
                break
            except tk.TclError:
                continue
        style.configure("Titulo.TLabel", font=("", 12, "bold"))
        style.configure("Ejecutar.TButton", font=("", 11, "bold"), padding=8)
        style.configure("Ayuda.TLabel", foreground="#555555", font=("", 9))

    # -----------------------------------------------------------------
    # Menu
    # -----------------------------------------------------------------
    def _construir_menu(self):
        menu = tk.Menu(self)

        m_archivo = tk.Menu(menu, tearoff=False)
        m_archivo.add_command(label="Nueva configuracion", accelerator="Ctrl+N",
                               command=self._nueva_config)
        m_archivo.add_command(label="Abrir configuracion...", accelerator="Ctrl+O",
                               command=self._abrir_config)
        self.m_recientes = tk.Menu(m_archivo, tearoff=False,
                                    postcommand=self._refrescar_menu_recientes)
        m_archivo.add_cascade(label="Abrir reciente", menu=self.m_recientes)
        m_archivo.add_command(label="Guardar configuracion", accelerator="Ctrl+S",
                               command=self._guardar_config)
        m_archivo.add_command(label="Guardar como...", command=self._guardar_config_como)
        m_archivo.add_separator()
        m_archivo.add_command(label="Salir", command=self._al_cerrar)
        menu.add_cascade(label="Archivo", menu=m_archivo)

        m_ejecutar = tk.Menu(menu, tearoff=False)
        m_ejecutar.add_command(label="Verificar archivos", command=self._verificar_archivos)
        m_ejecutar.add_command(label="Correr analisis", accelerator="Ctrl+R",
                                command=self._ejecutar_analisis)
        m_ejecutar.add_command(label="Abrir carpeta de salida",
                                command=self._abrir_carpeta_salida)
        menu.add_cascade(label="Ejecutar", menu=m_ejecutar)

        m_ayuda = tk.Menu(menu, tearoff=False)
        m_ayuda.add_command(label="Acerca de...", command=self._mostrar_acerca_de)
        menu.add_cascade(label="Ayuda", menu=m_ayuda)

        self.config(menu=menu)

    # -----------------------------------------------------------------
    # Layout general: notebook con 4 pestañas
    # -----------------------------------------------------------------
    def _construir_layout(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._al_cambiar_tab_notebook)

        self.tab_calibracion = ttk.Frame(self.notebook, padding=12)
        self.tab_materiales = ttk.Frame(self.notebook, padding=12)
        self.tab_preview_s11 = ttk.Frame(self.notebook, padding=12)
        self.tab_salida = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.tab_calibracion, text="  1. Calibracion  ")
        self.notebook.add(self.tab_materiales, text="  2. Materiales  ")
        self.notebook.add(self.tab_preview_s11, text="  3. Vista previa S11  ")
        self.notebook.add(self.tab_salida, text="  4. Salida y ejecucion  ")

        self._construir_tab_calibracion()
        self._construir_tab_materiales()
        self._construir_tab_preview_s11()
        self._construir_tab_salida()

    def _al_cambiar_tab_notebook(self, _event=None):
        """Cada vez que se cambia de pestaña, si la nueva es 'Vista previa
        S11' se refresca la lista de que previsualizar -- asi si se
        agrego/edito un material o un patron de calibracion en otra
        pestaña, aparece actualizado sin tener que hacer nada mas."""
        if self.notebook.select() == str(self.tab_preview_s11):
            self._refrescar_lista_preview_s11()

    # -----------------------------------------------------------------
    # Pestaña 1: calibracion
    # -----------------------------------------------------------------
    def _construir_tab_calibracion(self):
        f = self.tab_calibracion
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Patrones de calibracion", style="Titulo.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(
            f, text="Los 4 son necesarios: el metodo simplificado usa corto + aire + "
                    "agua; el metodo completo usa los 4.",
            style="Ayuda.TLabel", wraplength=760, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(f, text="Carpeta base (opcional):").grid(row=2, column=0, sticky="w", pady=3)
        self.var_carpeta_datos = tk.StringVar()
        entry_carpeta = ttk.Entry(f, textvariable=self.var_carpeta_datos)
        entry_carpeta.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Examinar...", command=self._examinar_carpeta_datos).grid(
            row=2, column=2, padx=(6, 0), pady=3)
        _tooltip(entry_carpeta,
                 "Opcional. Si elegis los archivos con 'Examinar...' (rutas\n"
                 "absolutas), podes dejar esto vacio.")

        # Cortocircuito y aire: no tienen temperatura asociada (no son
        # liquidos con modelo de Debye).
        self.vars_calibracion = {}
        fila = 3
        for clave, etiqueta in [('corto', "Cortocircuito:"), ('aire', "Aire:")]:
            ttk.Label(f, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
            var = tk.StringVar()
            self.vars_calibracion[clave] = var
            ttk.Entry(f, textvariable=var).grid(row=fila, column=1, sticky="ew", pady=3)
            ttk.Button(f, text="Examinar...",
                       command=lambda c=clave: self._examinar_archivo_calibracion(c)).grid(
                row=fila, column=2, padx=(6, 0), pady=3)
            fila += 1

        # Agua y alcohol isopropilico: SI tienen temperatura, cada uno con
        # su propio campo al lado del archivo (la temperatura entra en el
        # calculo de ambos metodos de conversion, en funciones.py).
        ttk.Label(f, text="Agua:").grid(row=fila, column=0, sticky="w", pady=3)
        var_agua = tk.StringVar()
        self.vars_calibracion['agua'] = var_agua
        ttk.Entry(f, textvariable=var_agua).grid(row=fila, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Examinar...",
                   command=lambda: self._examinar_archivo_calibracion('agua')).grid(
            row=fila, column=2, padx=(6, 0), pady=3)
        ttk.Label(f, text="  T (°C):").grid(row=fila, column=3, sticky="w")
        self.var_temp_agua = tk.DoubleVar(value=25.0)
        spin_agua = ttk.Spinbox(f, from_=-20, to=150, increment=0.5,
                                 textvariable=self.var_temp_agua, width=8)
        spin_agua.grid(row=fila, column=4, sticky="w", padx=(2, 0), pady=3)
        _tooltip(spin_agua, "Temperatura real del agua destilada durante la calibracion.\n"
                             "Afecta la precision de los dos metodos.")
        fila += 1

        ttk.Label(f, text="Patron adicional\n(alc. isopropilico):", justify="left").grid(
            row=fila, column=0, sticky="w", pady=3)
        var_isoprop = tk.StringVar()
        self.vars_calibracion['alc_isoprop'] = var_isoprop
        ttk.Entry(f, textvariable=var_isoprop).grid(row=fila, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Examinar...",
                   command=lambda: self._examinar_archivo_calibracion('alc_isoprop')).grid(
            row=fila, column=2, padx=(6, 0), pady=3)
        ttk.Label(f, text="  T (°C):").grid(row=fila, column=3, sticky="w")
        self.var_temp_isoprop = tk.DoubleVar(value=25.0)
        spin_isoprop = ttk.Spinbox(f, from_=-20, to=150, increment=0.5,
                                    textvariable=self.var_temp_isoprop, width=8)
        spin_isoprop.grid(row=fila, column=4, sticky="w", padx=(2, 0), pady=3)
        _tooltip(spin_isoprop,
                  "Temperatura real del alcohol isopropilico usado como 4to\n"
                  "patron durante la calibracion. Solo la usa el metodo completo,\n"
                  "pero afecta el resultado de TODOS los materiales analizados\n"
                  "con ese metodo (entra en el calculo de la conductancia Gn),\n"
                  "no solo el del alcohol isopropilico.")
        fila += 1

        ttk.Separator(f).grid(row=fila, column=0, columnspan=5, sticky="ew", pady=12)
        fila += 1

        ttk.Label(
            f, text="ⓘ El metodo completo asume que el 4to patron es alcohol "
                    "isopropilico. Si se usa otro liquido ahi, agregalo tambien en "
                    "la pestaña de Materiales (con su propio modelo y temperatura) "
                    "para poder revisar su curva teorica.",
            style="Ayuda.TLabel", wraplength=760, justify="left",
        ).grid(row=fila, column=0, columnspan=5, sticky="w", pady=(2, 0))

    def _examinar_carpeta_datos(self):
        ruta = filedialog.askdirectory(title="Carpeta base de datos", parent=self)
        if ruta:
            self.var_carpeta_datos.set(ruta)

    def _examinar_archivo_calibracion(self, clave):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo .s1p", parent=self,
            filetypes=[("Touchstone .s1p", "*.s1p"), ("Todos los archivos", "*.*")])
        if ruta:
            self.vars_calibracion[clave].set(ruta)

    # -----------------------------------------------------------------
    # Pestaña 2: materiales
    # -----------------------------------------------------------------
    def _construir_tab_materiales(self):
        f = self.tab_materiales
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        ttk.Label(f, text="Materiales a analizar", style="Titulo.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        frm_tabla = ttk.Frame(f)
        frm_tabla.grid(row=1, column=0, sticky="nsew")
        frm_tabla.columnconfigure(0, weight=1)
        frm_tabla.rowconfigure(0, weight=1)

        columnas = ("nombre", "archivo", "modelo", "temperatura", "metodos")
        self.tabla_materiales = ttk.Treeview(
            frm_tabla, columns=columnas, show="headings", selectmode="browse", height=12)
        titulos = {"nombre": "Nombre", "archivo": "Archivo", "modelo": "Modelo teorico",
                   "temperatura": "T (°C)", "metodos": "Metodos"}
        anchos = {"nombre": 150, "archivo": 260, "modelo": 190, "temperatura": 70, "metodos": 150}
        for c in columnas:
            self.tabla_materiales.heading(c, text=titulos[c])
            self.tabla_materiales.column(c, width=anchos[c], anchor="w")
        self.tabla_materiales.grid(row=0, column=0, sticky="nsew")
        self.tabla_materiales.bind("<Double-1>", lambda _e: self._editar_material())
        self.tabla_materiales.bind("<Delete>", lambda _e: self._quitar_material())

        scroll = ttk.Scrollbar(frm_tabla, orient="vertical", command=self.tabla_materiales.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tabla_materiales.configure(yscrollcommand=scroll.set)

        frm_botones = ttk.Frame(f)
        frm_botones.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(frm_botones, text="Agregar...", command=self._agregar_material).pack(side="left")
        ttk.Button(frm_botones, text="Editar...", command=self._editar_material).pack(side="left", padx=6)
        ttk.Button(frm_botones, text="Quitar", command=self._quitar_material).pack(side="left")
        ttk.Separator(frm_botones, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(frm_botones, text="Subir", command=lambda: self._mover_material(-1)).pack(side="left")
        ttk.Button(frm_botones, text="Bajar", command=lambda: self._mover_material(1)).pack(side="left", padx=6)

    def _refrescar_tabla_materiales(self):
        self.tabla_materiales.delete(*self.tabla_materiales.get_children())
        for i, m in enumerate(self._materiales):
            metodos = "+".join(m.get('metodos', []))
            temp = f"{m['temperatura']:.1f}" if m.get('modelo') else "-"
            self.tabla_materiales.insert(
                "", "end", iid=str(i),
                values=(m['nombre'], m['archivo'], gf.etiqueta_de_modelo(m.get('modelo')), temp, metodos))

    def _indice_seleccionado(self):
        sel = self.tabla_materiales.selection()
        return int(sel[0]) if sel else None

    def _agregar_material(self):
        dialogo = DialogoMaterial(self)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._materiales.append(dialogo.resultado)
            self._refrescar_tabla_materiales()

    def _editar_material(self):
        idx = self._indice_seleccionado()
        if idx is None:
            messagebox.showinfo("Editar material", "Elegi primero una fila de la tabla.", parent=self)
            return
        dialogo = DialogoMaterial(self, material=copy.deepcopy(self._materiales[idx]))
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._materiales[idx] = dialogo.resultado
            self._refrescar_tabla_materiales()

    def _quitar_material(self):
        idx = self._indice_seleccionado()
        if idx is None:
            return
        nombre = self._materiales[idx]['nombre']
        if messagebox.askyesno("Quitar material", f"¿Quitar '{nombre}' de la lista?", parent=self):
            del self._materiales[idx]
            self._refrescar_tabla_materiales()

    def _mover_material(self, delta):
        idx = self._indice_seleccionado()
        if idx is None:
            return
        nuevo = idx + delta
        if not (0 <= nuevo < len(self._materiales)):
            return
        self._materiales[idx], self._materiales[nuevo] = self._materiales[nuevo], self._materiales[idx]
        self._refrescar_tabla_materiales()
        self.tabla_materiales.selection_set(str(nuevo))

    # -----------------------------------------------------------------
    # Pestaña 3: vista previa de S11 (modulo/fase + Smith), antes de
    # correr el analisis completo -- lee el .s1p directamente y arma las
    # figuras en memoria con _generar_preview_s11 (ver arriba).
    # -----------------------------------------------------------------
    def _construir_tab_preview_s11(self):
        f = self.tab_preview_s11
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(3, weight=1)

        ttk.Label(f, text="Vista previa de S11", style="Titulo.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(
            f, text="Previsualiza el S11 crudo (modulo/fase y diagrama de Smith) de "
                    "cualquier patron de calibracion o material ya cargado en las "
                    "pestañas anteriores, leyendo directamente el .s1p. No hace falta "
                    "correr el analisis completo para ver esto.",
            style="Ayuda.TLabel", wraplength=920, justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        frm_selector = ttk.Frame(f)
        frm_selector.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(frm_selector, text="Elegi que previsualizar:").pack(side="left")
        self.var_preview_s11_seleccion = tk.StringVar()
        self.combo_preview_s11 = ttk.Combobox(
            frm_selector, textvariable=self.var_preview_s11_seleccion,
            state="readonly", values=[], width=42)
        self.combo_preview_s11.pack(side="left", padx=(8, 8))
        self.combo_preview_s11.bind(
            "<<ComboboxSelected>>", lambda _e: self._mostrar_preview_s11_seleccionado())
        ttk.Button(frm_selector, text="Refrescar lista",
                   command=self._refrescar_lista_preview_s11).pack(side="left")

        frm_magfase = ttk.LabelFrame(f, text="S11 medido (modulo y fase)")
        frm_magfase.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        frm_smith = ttk.LabelFrame(f, text="S11 medido (diagrama de Smith)")
        frm_smith.grid(row=3, column=1, sticky="nsew", padx=(6, 0))

        # Sin esto, el frame se encoge/agranda para ajustarse al tamaño
        # "natural" de lo que tiene adentro (la imagen ya escalada), en vez
        # de ser al reves. Con propagate(False) el tamaño del panel lo
        # decide unicamente el layout de la pestaña (columnas con weight=1
        # -> usan todo el ancho disponible de la ventana), y la imagen se
        # escala DESPUES para entrar ahi -- lo que evita que quede chica
        # con un margen enorme de espacio vacio alrededor.
        frm_magfase.pack_propagate(False)
        frm_smith.pack_propagate(False)

        if _HAY_PIL:
            self.label_preview_s11_magfase = ttk.Label(frm_magfase, anchor="center")
            self.label_preview_s11_magfase.pack(fill="both", expand=True, padx=6, pady=6)
            self.label_preview_s11_smith = ttk.Label(frm_smith, anchor="center")
            self.label_preview_s11_smith.pack(fill="both", expand=True, padx=6, pady=6)
            # Si se redimensiona la ventana (y por lo tanto estos paneles),
            # se reescala la imagen ya generada para aprovechar el nuevo
            # tamaño (con un pequeño debounce para no recalcular en cada
            # pixel mientras se arrastra el borde de la ventana).
            frm_magfase.bind("<Configure>", self._programar_reescalado_preview_s11)
            frm_smith.bind("<Configure>", self._programar_reescalado_preview_s11)
        else:
            texto_sin_pil = ("Instala Pillow (pip install pillow) para ver la\n"
                              "vista previa aca.")
            self.label_preview_s11_magfase = ttk.Label(
                frm_magfase, anchor="center", justify="left",
                wraplength=ANCHO_PREVIEW_S11, text=texto_sin_pil)
            self.label_preview_s11_magfase.pack(fill="both", expand=True, padx=6, pady=6)
            self.label_preview_s11_smith = ttk.Label(
                frm_smith, anchor="center", justify="left",
                wraplength=ANCHO_PREVIEW_S11, text=texto_sin_pil)
            self.label_preview_s11_smith.pack(fill="both", expand=True, padx=6, pady=6)

        self.var_preview_s11_status = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_preview_s11_status,
                  style="Ayuda.TLabel", wraplength=920, justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _refrescar_lista_preview_s11(self):
        """Reconstruye la lista de 'que previsualizar' a partir de lo que
        haya cargado AHORA MISMO en las pestañas de Calibracion y
        Materiales. Se llama sola al entrar a esta pestaña (ver
        _al_cambiar_tab_notebook) y tambien desde el boton 'Refrescar
        lista', por si se prefiere refrescar sin cambiar de pestaña."""
        carpeta = self.var_carpeta_datos.get().strip()
        rutas = {}

        for clave in ('corto', 'aire', 'agua', 'alc_isoprop'):
            archivo = self.vars_calibracion[clave].get().strip()
            if archivo:
                etiqueta = f"[Calibracion] {_ETIQUETAS_CALIBRACION_PREVIEW.get(clave, clave)}"
                rutas[etiqueta] = os.path.join(carpeta, archivo)

        for m in self._materiales:
            archivo = (m.get('archivo') or "").strip()
            if archivo:
                etiqueta = f"[Material] {m.get('nombre') or archivo}"
                rutas[etiqueta] = os.path.join(carpeta, archivo)

        self._rutas_preview_s11 = rutas
        seleccion_previa = self.var_preview_s11_seleccion.get()
        self.combo_preview_s11.configure(values=list(rutas.keys()))

        if not rutas:
            self.var_preview_s11_seleccion.set("")
            self._limpiar_preview_s11(
                "Todavia no hay ningun archivo cargado en 'Calibracion' ni en "
                "'Materiales' para previsualizar.")
            return

        if seleccion_previa not in rutas:
            self.var_preview_s11_seleccion.set(next(iter(rutas)))
        self._mostrar_preview_s11_seleccionado()

    def _limpiar_preview_s11(self, mensaje=""):
        if _HAY_PIL:
            self.label_preview_s11_magfase.configure(image="", text=mensaje)
            self.label_preview_s11_smith.configure(image="", text=mensaje)
            self._imagen_preview_s11_magfase_tk = None
            self._imagen_preview_s11_smith_tk = None
            self._pil_s11_magfase = None
            self._pil_s11_smith = None
        self.var_preview_s11_status.set(mensaje)

    def _mostrar_preview_s11_seleccionado(self):
        etiqueta = self.var_preview_s11_seleccion.get()
        ruta = self._rutas_preview_s11.get(etiqueta)
        if not ruta:
            return

        if not _HAY_PIL:
            self.var_preview_s11_status.set(
                "Instala Pillow (pip install pillow) para ver la vista previa aca.")
            return

        if not os.path.isfile(ruta):
            self._limpiar_preview_s11(f"No se encontro el archivo de '{etiqueta}':\n{ruta}")
            return

        self.var_preview_s11_status.set(f"Generando vista previa de '{etiqueta}'...")
        self.update_idletasks()  # forzar el redibujado antes de la parte lenta
        try:
            imagen_mf, imagen_sm, datos = _generar_preview_s11(
                ruta, log_x=bool(self.var_log_x.get()))
        except Exception as exc:
            self._limpiar_preview_s11(
                f"No se pudo generar la vista previa de '{etiqueta}':\n{exc}")
            return

        # Se guardan las imagenes SIN escalar: _refrescar_imagenes_preview_s11
        # las escala al tamaño real disponible en cada panel ahora mismo, y
        # las vuelve a escalar solas si despues se agranda/achica la ventana.
        self._pil_s11_magfase = imagen_mf
        self._pil_s11_smith = imagen_sm
        self._refrescar_imagenes_preview_s11()

        frec = datos['Frec']
        self.var_preview_s11_status.set(
            f"{etiqueta}  ({os.path.basename(ruta)})  \u2014  {len(frec)} puntos, "
            f"{frec[0] / 1e9:.3f}-{frec[-1] / 1e9:.3f} GHz")

    def _programar_reescalado_preview_s11(self, _event=None):
        """Se llama en cada <Configure> de los paneles de preview (o sea,
        cada vez que cambian de tamaño, tipicamente porque se redimensiono
        la ventana). Reprograma el reescalado real con un pequeño retraso
        (debounce): mientras se arrastra el borde de la ventana llegan
        muchisimos eventos de Configure seguidos, y recalcular la imagen en
        cada uno seria lento y tironearia la interfaz."""
        if self._reescalado_preview_s11_id is not None:
            try:
                self.after_cancel(self._reescalado_preview_s11_id)
            except (tk.TclError, ValueError):
                pass
        self._reescalado_preview_s11_id = self.after(150, self._refrescar_imagenes_preview_s11)

    def _refrescar_imagenes_preview_s11(self):
        """Vuelve a escalar (a partir de las imagenes PIL originales ya
        cacheadas) y mostrar ambas vistas previas de S11, ajustandolas al
        tamaño real disponible AHORA MISMO en cada panel. No vuelve a leer
        el .s1p ni a generar los graficos de nuevo -- solo re-escala."""
        self._reescalado_preview_s11_id = None
        if self._pil_s11_magfase is not None:
            self._imagen_preview_s11_magfase_tk = self._escalar_para_caja(
                self._pil_s11_magfase, self.label_preview_s11_magfase)
            self.label_preview_s11_magfase.configure(
                image=self._imagen_preview_s11_magfase_tk, text="")
        if self._pil_s11_smith is not None:
            self._imagen_preview_s11_smith_tk = self._escalar_para_caja(
                self._pil_s11_smith, self.label_preview_s11_smith)
            self.label_preview_s11_smith.configure(
                image=self._imagen_preview_s11_smith_tk, text="")

    @staticmethod
    def _escalar_para_caja(imagen_pil, widget_destino, margen=14):
        """PIL.Image -> ImageTk.PhotoImage, escalado (preservando la
        relacion de aspecto) para entrar en el espacio REAL actualmente
        disponible de `widget_destino` (con `pack_propagate(False)` en su
        contenedor, ese tamaño lo fija el layout de la pestaña, no la
        imagen -- ver _construir_tab_preview_s11). Si el widget todavia no
        tiene un tamaño asignado (arranque en frio, antes de que la
        ventana termine de dibujarse), usa ANCHO_PREVIEW_S11 de resguardo."""
        ancho_disp = widget_destino.winfo_width() - margen
        alto_disp = widget_destino.winfo_height() - margen
        if ancho_disp < 80 or alto_disp < 80:
            ancho_disp = alto_disp = ANCHO_PREVIEW_S11

        ancho_img, alto_img = imagen_pil.size
        escala = min(ancho_disp / max(ancho_img, 1), alto_disp / max(alto_img, 1))
        escala = max(escala, 0.05)
        nuevo_tamano = (max(int(ancho_img * escala), 1), max(int(alto_img * escala), 1))
        return ImageTk.PhotoImage(imagen_pil.resize(nuevo_tamano, Image.LANCZOS))

    # -----------------------------------------------------------------
    # Pestaña 4: salida y ejecucion
    # -----------------------------------------------------------------
    def _construir_tab_salida(self):
        f = self.tab_salida
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Salida", style="Titulo.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(f, text="Carpeta de salida:").grid(row=1, column=0, sticky="w", pady=3)
        self.var_carpeta_salida = tk.StringVar()
        ttk.Entry(f, textvariable=self.var_carpeta_salida).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Examinar...", command=self._examinar_carpeta_salida).grid(
            row=1, column=2, padx=(6, 0), pady=3)

        ttk.Label(f, text="Nombre del informe PDF:").grid(row=2, column=0, sticky="w", pady=3)
        self.var_nombre_pdf = tk.StringVar(value="informe_permitividad.pdf")
        ttk.Entry(f, textvariable=self.var_nombre_pdf).grid(row=2, column=1, sticky="ew", pady=3)

        frm_freq = ttk.Frame(f)
        frm_freq.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(frm_freq, text="Rango de frecuencias:").pack(side="left")
        self.var_f_min = tk.DoubleVar(value=0.5)
        self.var_f_max = tk.DoubleVar(value=6.0)
        ttk.Label(frm_freq, text="  F min (GHz)").pack(side="left", padx=(12, 4))
        ttk.Spinbox(frm_freq, from_=0.0001, to=50, increment=0.1, textvariable=self.var_f_min,
                    width=8).pack(side="left")
        ttk.Label(frm_freq, text="  F max (GHz)").pack(side="left", padx=(12, 4))
        ttk.Spinbox(frm_freq, from_=0.0001, to=50, increment=0.1, textvariable=self.var_f_max,
                    width=8).pack(side="left")

        self.var_log_x = tk.BooleanVar(value=True)
        check_log_x = ttk.Checkbutton(
            f, text="Escala logaritmica en frecuencia (recomendado para barridos de varias decadas, p.ej. MHz a GHz)",
            variable=self.var_log_x)
        check_log_x.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        _tooltip(check_log_x,
                 "En escala lineal, si el barrido va de (por ejemplo) 1 MHz a\n"
                 "2 GHz, todo lo que pasa por debajo de un par de cientos de MHz\n"
                 "queda comprimido contra el borde izquierdo del grafico y no se\n"
                 "distingue nada. En log, cada decada ocupa el mismo ancho visual.\n"
                 "Se ignora automaticamente (cae a lineal) si el rango incluye 0 Hz.")

        ttk.Separator(f).grid(row=5, column=0, columnspan=3, sticky="ew", pady=12)

        frm_acciones = ttk.Frame(f)
        frm_acciones.grid(row=6, column=0, columnspan=3, sticky="w")
        ttk.Button(frm_acciones, text="Verificar archivos", command=self._verificar_archivos).pack(side="left")
        self.boton_ejecutar = ttk.Button(
            frm_acciones, text="▶  Ejecutar analisis", style="Ejecutar.TButton",
            command=self._ejecutar_analisis)
        self.boton_ejecutar.pack(side="left", padx=10)
        self.boton_cancelar = ttk.Button(
            frm_acciones, text="■  Cancelar", command=self._cancelar_analisis, state="disabled")
        self.boton_cancelar.pack(side="left", padx=(0, 10))
        self.boton_abrir_salida = ttk.Button(
            frm_acciones, text="Abrir carpeta de salida", command=self._abrir_carpeta_salida)
        self.boton_abrir_salida.pack(side="left")

        self.var_paso_actual = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_paso_actual, style="Ayuda.TLabel").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.barra_progreso = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.barra_progreso.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        # --- Consola + vista previa, lado a lado ---
        frm_abajo = ttk.Frame(f)
        frm_abajo.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        f.rowconfigure(9, weight=1)
        frm_abajo.columnconfigure(0, weight=2)
        frm_abajo.columnconfigure(1, weight=3)
        frm_abajo.rowconfigure(0, weight=1)

        frm_consola = ttk.LabelFrame(frm_abajo, text="Consola")
        frm_consola.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frm_consola.columnconfigure(0, weight=1)
        frm_consola.rowconfigure(0, weight=1)
        self.texto_consola = tk.Text(frm_consola, height=14, wrap="word", state="disabled",
                                      font=("Courier New", 9))
        self.texto_consola.grid(row=0, column=0, sticky="nsew")
        scroll_consola = ttk.Scrollbar(frm_consola, orient="vertical", command=self.texto_consola.yview)
        scroll_consola.grid(row=0, column=1, sticky="ns")
        self.texto_consola.configure(yscrollcommand=scroll_consola.set)

        frm_consola_botones = ttk.Frame(frm_consola)
        frm_consola_botones.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(frm_consola_botones, text="Copiar", command=self._copiar_consola).pack(side="left")
        ttk.Button(frm_consola_botones, text="Guardar como...",
                   command=self._guardar_consola).pack(side="left", padx=(6, 0))

        frm_preview = ttk.LabelFrame(frm_abajo, text="Vista previa (permitividad medida vs. teorica)")
        frm_preview.grid(row=0, column=1, sticky="nsew")
        frm_preview.columnconfigure(0, weight=1)
        frm_preview.rowconfigure(1, weight=1)
        # Igual que en la pestaña "Vista previa S11": sin esto, el panel se
        # encoge/agranda para ajustarse al tamaño de la imagen ya escalada
        # en vez de al reves, y la imagen termina quedando chica con un
        # margen enorme de espacio vacio alrededor.
        frm_preview.grid_propagate(False)

        self.var_figura_seleccionada = tk.StringVar()
        self.combo_figuras = ttk.Combobox(frm_preview, textvariable=self.var_figura_seleccionada,
                                           state="readonly", values=[])
        self.combo_figuras.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.combo_figuras.bind("<<ComboboxSelected>>", lambda _e: self._mostrar_preview_seleccionada())

        if _HAY_PIL:
            self.label_preview = ttk.Label(frm_preview, anchor="center")
            self.label_preview.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
            # El S11 (modulo/fase y Smith) de calibracion y materiales se ve
            # en la pestaña "3. Vista previa S11" -- ahi ademas se puede ver
            # antes de correr el analisis. Esta lista se reescala sola si
            # se redimensiona la ventana (ver _programar_reescalado_preview).
            frm_preview.bind("<Configure>", self._programar_reescalado_preview)
        else:
            self.label_preview = ttk.Label(
                frm_preview, anchor="center", justify="left", wraplength=ANCHO_PREVIEW,
                text="Instala Pillow (pip install pillow) para ver la vista previa de las "
                     "figuras aca. Mientras tanto, usa 'Abrir carpeta de salida'.")
            self.label_preview.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

    def _examinar_carpeta_salida(self):
        ruta = filedialog.askdirectory(title="Carpeta de salida", parent=self)
        if ruta:
            self.var_carpeta_salida.set(ruta)

    # -----------------------------------------------------------------
    # Status bar
    # -----------------------------------------------------------------
    def _construir_statusbar(self):
        self.status_var = tk.StringVar(value="Listo.")
        barra = ttk.Frame(self, relief="sunken", padding=(8, 3))
        barra.pack(fill="x", side="bottom")
        ttk.Label(barra, textvariable=self.status_var).pack(side="left")

    def _status(self, texto):
        self.status_var.set(texto)

    # -----------------------------------------------------------------
    # Config: leer de la GUI / volcar a la GUI
    # -----------------------------------------------------------------
    def _config_desde_gui(self):
        return {
            'version': gf.CONFIG_VERSION,
            'carpeta_datos': self.var_carpeta_datos.get().strip(),
            'archivos_calibracion': {
                clave: var.get().strip() for clave, var in self.vars_calibracion.items()
            },
            'temperatura_agua_c': float(self._obtener_double(self.var_temp_agua, 25.0)),
            'temperatura_isoprop_cal_c': float(self._obtener_double(self.var_temp_isoprop, 25.0)),
            'materiales': copy.deepcopy(self._materiales),
            'carpeta_salida': self.var_carpeta_salida.get().strip(),
            'nombre_informe_pdf': self.var_nombre_pdf.get().strip() or "informe_permitividad.pdf",
            'f_min_ghz': float(self._obtener_double(self.var_f_min, 0.5)),
            'f_max_ghz': float(self._obtener_double(self.var_f_max, 6.0)),
            'escala_log_frecuencia': bool(self.var_log_x.get()),
        }

    @staticmethod
    def _obtener_double(var, default):
        try:
            return var.get()
        except (tk.TclError, ValueError):
            return default

    def _cargar_config_en_gui(self, config):
        self.var_carpeta_datos.set(config.get('carpeta_datos', "") or "")
        archivos = config.get('archivos_calibracion', {}) or {}
        for clave, var in self.vars_calibracion.items():
            var.set(archivos.get(clave, "") or "")
        self.var_temp_agua.set(float(config.get('temperatura_agua_c', 25.0) or 25.0))
        self.var_temp_isoprop.set(float(config.get('temperatura_isoprop_cal_c', 25.0) or 25.0))
        self._materiales = copy.deepcopy(config.get('materiales', []) or [])
        self._refrescar_tabla_materiales()
        self.var_carpeta_salida.set(config.get('carpeta_salida', "") or "")
        self.var_nombre_pdf.set(config.get('nombre_informe_pdf', "informe_permitividad.pdf"))
        self.var_f_min.set(float(config.get('f_min_ghz', 0.5) or 0.5))
        self.var_f_max.set(float(config.get('f_max_ghz', 6.0) or 6.0))
        self.var_log_x.set(bool(config.get('escala_log_frecuencia', True)))

    def _hay_cambios_sin_guardar(self):
        if self._snapshot_guardado is None:
            return True
        return self._config_desde_gui() != self._snapshot_guardado

    def _marcar_como_guardada(self):
        self._snapshot_guardado = self._config_desde_gui()
        self._actualizar_titulo()

    def _actualizar_titulo(self):
        nombre = os.path.basename(self.ruta_config_actual) if self.ruta_config_actual else "sin guardar"
        marca = " *" if self._hay_cambios_sin_guardar() else ""
        self.title(f"{TITULO_APP} — {nombre}{marca}")

    # -----------------------------------------------------------------
    # Menu Archivo: nuevo / abrir / guardar
    # -----------------------------------------------------------------
    def _confirmar_descartar_cambios(self):
        """Si hay cambios sin guardar, pregunta si continuar igual.
        Devuelve True si esta OK seguir adelante (guardo, o el usuario
        decidio descartar), False si hay que cancelar la accion."""
        if not self._hay_cambios_sin_guardar():
            return True
        respuesta = messagebox.askyesnocancel(
            "Cambios sin guardar",
            "Hay cambios sin guardar en la configuracion actual.\n\n"
            "¿Queres guardarlos antes de continuar?",
            parent=self,
        )
        if respuesta is None:
            return False
        if respuesta is True:
            return self._guardar_config()
        return True  # descartar y seguir

    def _nueva_config(self):
        if not self._confirmar_descartar_cambios():
            return
        self.ruta_config_actual = None
        self._cargar_config_en_gui(gf.config_default())
        self._marcar_como_guardada()
        self._status("Configuracion nueva.")

    def _abrir_config(self):
        if not self._confirmar_descartar_cambios():
            return
        ruta = filedialog.askopenfilename(
            title="Abrir configuracion", parent=self,
            filetypes=[("Configuracion JSON", "*.json"), ("Todos los archivos", "*.*")])
        if not ruta:
            return
        try:
            self._cargar_config_desde(ruta)
        except Exception as exc:
            messagebox.showerror("Error al abrir", f"No se pudo abrir la configuracion:\n{exc}", parent=self)

    def _refrescar_menu_recientes(self):
        """Reconstruye el submenu 'Archivo > Abrir reciente' justo antes
        de mostrarse (via postcommand), asi siempre refleja el historial
        actual sin tener que acordarse de refrescarlo a mano desde cada
        lugar donde se guarda/carga una configuracion."""
        self.m_recientes.delete(0, "end")
        recientes = [r for r in gf.rutas_recientes() if r != self.ruta_config_actual]
        if not recientes:
            self.m_recientes.add_command(label="(vacio)", state="disabled")
            return
        for ruta in recientes:
            carpeta = os.path.basename(os.path.dirname(ruta))
            etiqueta = f"{os.path.basename(ruta)}   \u2014   {carpeta}" if carpeta else os.path.basename(ruta)
            self.m_recientes.add_command(
                label=etiqueta, command=lambda r=ruta: self._abrir_config_reciente(r))
        self.m_recientes.add_separator()
        self.m_recientes.add_command(label="Limpiar lista", command=self._limpiar_recientes)

    def _abrir_config_reciente(self, ruta):
        if not self._confirmar_descartar_cambios():
            return
        try:
            self._cargar_config_desde(ruta)
        except Exception as exc:
            messagebox.showerror("Error al abrir", f"No se pudo abrir la configuracion:\n{exc}", parent=self)

    def _limpiar_recientes(self):
        gf.limpiar_recientes()

    def _cargar_config_desde(self, ruta):
        config = gf.cargar_config(ruta)
        self._cargar_config_en_gui(config)
        self.ruta_config_actual = ruta
        self._marcar_como_guardada()
        self._status(f"Configuracion cargada: {ruta}")

    def _guardar_config(self):
        if not self.ruta_config_actual:
            return self._guardar_config_como()
        try:
            gf.guardar_config(self.ruta_config_actual, self._config_desde_gui())
        except OSError as exc:
            messagebox.showerror("Error al guardar", f"No se pudo guardar la configuracion:\n{exc}", parent=self)
            return False
        self._marcar_como_guardada()
        self._status(f"Configuracion guardada: {self.ruta_config_actual}")
        return True

    def _guardar_config_como(self):
        ruta = filedialog.asksaveasfilename(
            title="Guardar configuracion como", parent=self, defaultextension=".json",
            filetypes=[("Configuracion JSON", "*.json"), ("Todos los archivos", "*.*")])
        if not ruta:
            return False
        self.ruta_config_actual = ruta
        return self._guardar_config()

    def _al_cerrar(self):
        if self._analisis_corriendo:
            if not messagebox.askyesno(
                    "Analisis en curso",
                    "Hay un analisis corriendo. ¿Salir igual? (el proceso en curso se corta)",
                    parent=self):
                return
            if self._evento_cancelar is not None:
                self._evento_cancelar.set()
        if self._confirmar_descartar_cambios():
            self.destroy()

    def _mostrar_acerca_de(self):
        messagebox.showinfo(
            "Acerca de",
            f"{TITULO_APP}\n\n"
            "GUI para configurar y correr el analisis de permitividad "
            "compleja medida con sonda coaxial open-ended.\n\n"
            "Modelos teoricos de referencia: NPL Report MAT 23 "
            "(Gregory & Clarke, 2012), implementados en Patrones.py.",
            parent=self,
        )

    # -----------------------------------------------------------------
    # Verificar archivos
    # -----------------------------------------------------------------
    def _verificar_archivos(self):
        problemas = gf.validar_config(self._config_desde_gui())
        if not problemas:
            messagebox.showinfo("Verificacion", "Todo OK: no se encontraron problemas.", parent=self)
        else:
            texto = "\n".join(f"• {p}" for p in problemas)
            messagebox.showwarning("Verificacion", f"Se encontraron algunos problemas:\n\n{texto}", parent=self)

    # -----------------------------------------------------------------
    # Ejecutar analisis (hilo de fondo + polling de la cola)
    # -----------------------------------------------------------------
    def _ejecutar_analisis(self):
        if self._analisis_corriendo:
            messagebox.showinfo("Analisis en curso", "Ya hay un analisis corriendo.", parent=self)
            return

        config = self._config_desde_gui()
        problemas = gf.validar_config(config)
        if problemas:
            texto = "\n".join(f"• {p}" for p in problemas)
            if not messagebox.askyesno(
                    "Se encontraron problemas",
                    f"{texto}\n\n¿Correr el analisis igual?", parent=self):
                return

        self.texto_consola.configure(state="normal")
        self.texto_consola.delete("1.0", "end")
        self.texto_consola.configure(state="disabled")
        self.barra_progreso.configure(value=0, maximum=max(len(config['materiales']) + 2, 1))
        self.var_paso_actual.set("")
        self._figuras_disponibles = {}
        self.combo_figuras.configure(values=[])
        self.var_figura_seleccionada.set("")

        self._analisis_corriendo = True
        self.boton_ejecutar.configure(state="disabled")
        self.boton_cancelar.configure(state="normal")
        self._status("Corriendo analisis...")

        _, self._cola_analisis, self._evento_cancelar = gf.lanzar_analisis_en_hilo(config)
        self.after(100, self._pollear_cola_analisis)

    def _cancelar_analisis(self):
        if not self._analisis_corriendo or self._evento_cancelar is None:
            return
        if self._evento_cancelar.is_set():
            return  # ya se pidio, no hace falta preguntar de nuevo
        if not messagebox.askyesno(
                "Cancelar analisis",
                "¿Cancelar el analisis en curso?\n\n"
                "Se termina de procesar el material actual (no se corta a "
                "mitad de un calculo) y se genera igual un informe PDF "
                "parcial con lo que ya se alcanzo a calcular.",
                parent=self):
            return
        self._evento_cancelar.set()
        self.boton_cancelar.configure(state="disabled")
        self._status("Cancelando... (termina el material actual y genera un informe parcial)")
        self.var_paso_actual.set("Cancelando...")

    def _agregar_linea_consola(self, texto):
        self.texto_consola.configure(state="normal")
        self.texto_consola.insert("end", texto + "\n")
        self.texto_consola.see("end")
        self.texto_consola.configure(state="disabled")

    def _copiar_consola(self):
        texto = self.texto_consola.get("1.0", "end-1c")
        if not texto.strip():
            messagebox.showinfo("Copiar log", "La consola todavia esta vacia.", parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(texto)
        self._status("Log de la consola copiado al portapapeles.")

    def _guardar_consola(self):
        texto = self.texto_consola.get("1.0", "end-1c")
        if not texto.strip():
            messagebox.showinfo("Guardar log", "La consola todavia esta vacia.", parent=self)
            return
        ruta = filedialog.asksaveasfilename(
            title="Guardar log como", parent=self, defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Todos los archivos", "*.*")])
        if not ruta:
            return
        try:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(texto)
        except OSError as exc:
            messagebox.showerror("Error al guardar", f"No se pudo guardar el log:\n{exc}", parent=self)
            return
        self._status(f"Log guardado: {ruta}")

    def _pollear_cola_analisis(self):
        try:
            while True:
                evento = self._cola_analisis.get_nowait()
                tipo = evento[0]
                if tipo == 'log':
                    self._agregar_linea_consola(evento[1])
                elif tipo == 'progreso':
                    _, paso, total, etiqueta = evento
                    self.barra_progreso.configure(maximum=max(total, 1), value=paso)
                    if etiqueta:
                        self.var_paso_actual.set(etiqueta)
                elif tipo == 'ok':
                    self._al_terminar_analisis_ok(evento[1])
                    return
                elif tipo == 'cancelado':
                    self._al_terminar_analisis_cancelado(evento[1])
                    return
                elif tipo == 'error':
                    self._al_terminar_analisis_error(evento[1], evento[2])
                    return
        except Exception:
            pass  # queue.Empty -> no hay mas eventos por ahora

        if self._analisis_corriendo:
            self.after(100, self._pollear_cola_analisis)

    def _figuras_desde_resultado(self, resultado):
        """Arma el dict {etiqueta: ruta_png} de figuras disponibles para
        la vista previa a partir de un resultado de `ejecutar_analisis`
        (usado tanto si termino OK como si se cancelo a mitad de camino,
        para no duplicar esta logica en los dos lugares)."""
        figuras = {}
        chequeo = resultado.get('chequeo_calibracion') or {}
        if chequeo.get('figura'):
            figuras["Chequeo de calibracion (agua)"] = chequeo['figura']
        if chequeo.get('figura_Gn'):
            figuras["Diagnostico: Gn(f) (calibracion)"] = chequeo['figura_Gn']
        for r in resultado.get('resultados_materiales', []):
            # Ojo: antes tambien se listaban aca "<material> - S11
            # (modulo/fase)" y "<material> - S11 (Smith)". Se sacaron
            # porque ahora esa vista existe en la pestaña "3. Vista previa
            # S11" (y ahi ademas se puede ver ANTES de correr el analisis,
            # leyendo el .s1p directo). Esta lista queda solo con lo que es
            # resultado del ANALISIS en si: la permitividad medida vs.
            # teorica de cada material, y el chequeo de calibracion.
            if r.get('figura'):
                figuras[r['nombre']] = r['figura']
        return figuras

    def _mostrar_figuras_disponibles(self, figuras):
        self._figuras_disponibles = figuras
        self.combo_figuras.configure(values=list(figuras.keys()))
        if figuras:
            primera = next(iter(figuras))
            self.var_figura_seleccionada.set(primera)
            self._mostrar_preview_seleccionada()

    def _al_terminar_analisis_ok(self, resultado):
        self._analisis_corriendo = False
        self.boton_ejecutar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self._status("Analisis terminado con exito.")
        self.var_paso_actual.set("Listo.")
        self._agregar_linea_consola("\n=== Analisis terminado con exito ===")

        self._mostrar_figuras_disponibles(self._figuras_desde_resultado(resultado))

        messagebox.showinfo(
            "Listo", f"Analisis terminado.\nInforme PDF:\n{resultado.get('ruta_informe_pdf', '')}",
            parent=self)

    def _al_terminar_analisis_cancelado(self, resultado):
        self._analisis_corriendo = False
        self.boton_ejecutar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self._status("Analisis cancelado por el usuario.")
        self.var_paso_actual.set("Cancelado.")
        self._agregar_linea_consola("\n=== Analisis cancelado por el usuario ===")

        # Se dejan disponibles las figuras que se llegaron a generar antes
        # de cancelar (chequeo de calibracion + los materiales que si se
        # alcanzaron a procesar), para no perder ese trabajo parcial.
        self._mostrar_figuras_disponibles(self._figuras_desde_resultado(resultado))

        ruta_pdf = resultado.get('ruta_informe_pdf')
        detalle_pdf = f"\nInforme PDF parcial:\n{ruta_pdf}" if ruta_pdf else ""
        messagebox.showinfo(
            "Analisis cancelado",
            f"Se cancelo el analisis. Se conserva lo que ya se calculo "
            f"({len(resultado.get('resultados_materiales', []))} material(es)).{detalle_pdf}",
            parent=self)

    def _al_terminar_analisis_error(self, mensaje, detalle):
        self._analisis_corriendo = False
        self.boton_ejecutar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self._status("Error durante el analisis.")
        self.var_paso_actual.set("Error.")
        self._agregar_linea_consola(f"\n=== ERROR: {mensaje} ===\n{detalle}")
        messagebox.showerror("Error durante el analisis", mensaje, parent=self)

    # -----------------------------------------------------------------
    # Vista previa de figuras (requiere Pillow)
    # -----------------------------------------------------------------
    def _mostrar_preview_seleccionada(self):
        if not _HAY_PIL:
            return
        etiqueta = self.var_figura_seleccionada.get()
        ruta = self._figuras_disponibles.get(etiqueta)
        if not ruta or not os.path.isfile(ruta):
            return
        try:
            imagen = Image.open(ruta)
            imagen.load()
        except Exception as exc:
            self.label_preview.configure(text=f"No se pudo mostrar la vista previa:\n{exc}", image="")
            return
        # Se guarda sin escalar: _refrescar_imagen_preview_seleccionada la
        # escala al tamaño real disponible ahora mismo, y se vuelve a
        # llamar sola si despues se redimensiona la ventana.
        self._pil_figura_seleccionada = imagen
        self._refrescar_imagen_preview_seleccionada()

    def _programar_reescalado_preview(self, _event=None):
        """Debounce del <Configure> del panel de vista previa (pestaña
        Salida y ejecucion) -- mismo motivo que
        _programar_reescalado_preview_s11: no recalcular en cada pixel
        mientras se arrastra el borde de la ventana."""
        if self._reescalado_preview_id is not None:
            try:
                self.after_cancel(self._reescalado_preview_id)
            except (tk.TclError, ValueError):
                pass
        self._reescalado_preview_id = self.after(150, self._refrescar_imagen_preview_seleccionada)

    def _refrescar_imagen_preview_seleccionada(self):
        self._reescalado_preview_id = None
        if self._pil_figura_seleccionada is None:
            return
        self._imagen_preview_tk = self._escalar_para_caja(
            self._pil_figura_seleccionada, self.label_preview, margen=10)
        self.label_preview.configure(image=self._imagen_preview_tk, text="")

    # -----------------------------------------------------------------
    # Abrir carpeta de salida
    # -----------------------------------------------------------------
    def _abrir_carpeta_salida(self):
        carpeta = self.var_carpeta_salida.get().strip()
        if not carpeta or not os.path.isdir(carpeta):
            messagebox.showinfo("Carpeta de salida",
                                 "La carpeta de salida todavia no existe (corre el analisis "
                                 "primero, o revisa la ruta).", parent=self)
            return
        gf.abrir_carpeta(carpeta)