"""
combinatoria/gui_combinatoria.py
--------------------------------
Interfaz gráfica de usuario para el análisis combinatorio y selección
óptima de patrones de calibración de permitividad dieléctrica.

Pestañas:
  1. Configuración y ejecución:
     - Selección de archivos Short y Open
     - Rango de frecuenciqas y los 3 intervalos de evaluación
     - Métodos de cálculo (Simplificado, Completo original, Completo acondicionado)
     - Parámetros de acondicionamiento de Gn(f)
     - Tabla editable de candidatos a patrones
     - Consola de ejecución y barra de progreso
  2. Ranking por muestra:
     - Selección de muestra evaluada
     - Clasificación interactiva según puntaje en intervalo 1, 2, 3 o promedio global
     - Visualización gráfica instantánea de er' y er'' medido vs teórico
  3. Resultados por combinación:
     - Selección de par de patrones (P3, P4)
     - Diagnóstico de conductancia Gn(f)
     - Tabla y gráficos de todas las muestras evaluadas con esa calibración
"""
import os
import sys
import copy
import json
import queue
import threading
import contextlib
import traceback
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Asegurar que el directorio raíz esté en sys.path
_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR_RAIZ not in sys.path:
    sys.path.insert(0, _DIR_RAIZ)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FormatStrFormatter

import Patrones as P
import gui_funciones as gf
from analisis_permitividad import graficar_comparacion, graficar_Gn, _slug
from combinatoria.analisis_combinatoria import ejecutar_analisis_combinatoria, filtrar_combinaciones_redundantes
from combinatoria.reporte_combinatoria_pdf import generar_reporte_combinatoria_pdf, generar_reporte_combinacion_pdf


_TIPOS_ACOND_MAP = {
    "Transicion cuadratica (f^2)": "transicion_polinomica_cero",
    "Tapering (coseno alzado)": "tapering",
    "Cero clamping": "cero_clamping",
}
_TIPOS_ACOND_INV = {v: k for k, v in _TIPOS_ACOND_MAP.items()}


def _normalizar_ruta_s1p(ruta, carpeta_datos):
    """
    Normaliza la ruta de un archivo .s1p seleccionado con el explorador
    de archivos: si se encuentra dentro de `carpeta_datos`, devuelve la
    ruta relativa a `carpeta_datos` (mismo criterio para Short, Open y
    Candidatos). Si está fuera de `carpeta_datos`, devuelve la ruta
    completa `ruta`.
    """
    if not ruta:
        return ""
    carpeta_str = (carpeta_datos or "").strip()
    if carpeta_str:
        try:
            abs_ruta = os.path.abspath(ruta)
            abs_carpeta = os.path.abspath(carpeta_str)
            if os.path.commonpath([abs_ruta, abs_carpeta]) == abs_carpeta:
                return os.path.relpath(abs_ruta, abs_carpeta)
        except ValueError:
            pass
    return ruta


def _crear_indicador_archivo(parent, var_archivo, var_carpeta=None):
    """
    Crea un Label compacto que muestra ✓ (verde) si el archivo existe o
    ✗ (rojo) si no existe, actualizándose automáticamente al editar el
    campo o cambiar la carpeta de datos (igual que en `gui_permitividad.py`).
    """
    label = ttk.Label(parent, width=2, anchor="center")

    def _actualizar(*_args):
        archivo = var_archivo.get().strip()
        if not archivo:
            label.configure(text="", foreground="black")
            return
        carpeta = var_carpeta.get().strip() if var_carpeta is not None else ""
        ruta = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta):
            label.configure(text="\u2713", foreground="#2e7d32")
        else:
            label.configure(text="\u2717", foreground="#c62828")

    var_archivo.trace_add("write", _actualizar)
    if var_carpeta is not None:
        var_carpeta.trace_add("write", _actualizar)
    _actualizar()
    return label


# ===========================================================================
# Tooltip mínimo
# ===========================================================================
class _Tooltip:
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
            font=("", 9), wraplength=340, padx=6, pady=4,
        )
        etiqueta.pack()

    def _ocultar(self, _event=None):
        if self._toplevel is not None:
            self._toplevel.destroy()
            self._toplevel = None


def _tooltip(widget, texto):
    _Tooltip(widget, texto)


# ===========================================================================
# Visor de figuras interactivo con NavigationToolbar2Tk
# ===========================================================================
class VisorFigura(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._canvas = None
        self._toolbar = None
        self._label_vacio = ttk.Label(self, anchor="center", justify="center",
                                       text="Sin datos para graficar")
        self._label_vacio.grid(row=0, column=0, sticky="nsew")

    def mostrar(self, fig):
        self._limpiar_widgets()
        self._label_vacio.grid_remove()

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._toolbar = NavigationToolbar2Tk(self._canvas, self, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.grid(row=1, column=0, sticky="ew")

    def limpiar(self, mensaje="Sin datos para graficar"):
        self._limpiar_widgets()
        self._label_vacio.configure(text=mensaje)
        self._label_vacio.grid()

    def _limpiar_widgets(self):
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None


# ===========================================================================
# Diálogo para agregar / editar candidato
# ===========================================================================
def generar_alias_defecto(nombre: str) -> str:
    """
    Genera el alias por defecto (máximo 3 letras, primera en mayúscula):
    - Si el nombre consta de 2 o más palabras separadas por espacios,
      usa las primeras 3 letras de la segunda palabra.
    - Si consta de 1 palabra, usa las primeras 3 letras de esa palabra.
    """
    if not nombre:
        return ""
    palabras = nombre.strip().split()
    if len(palabras) >= 2:
        palabra = palabras[1]
    else:
        palabra = palabras[0]
    return palabra[:3].capitalize()


class DialogoCandidato(tk.Toplevel):
    def __init__(self, master, candidato=None, carpeta_datos=""):
        super().__init__(master)
        self.resultado = None
        self.carpeta_datos = carpeta_datos
        candidato = candidato or {
            'nombre': '', 'alias': '', 'archivo': '', 'modelo': 'agua', 'temperatura': 25.0, 'habilitado': True
        }

        self.title("Agregar candidato a patrón" if not candidato.get('nombre') else
                   f"Editar candidato: {candidato['nombre']}")
        self.resizable(False, False)
        self.transient(master)

        frm = ttk.Frame(self, padding=14)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Nombre:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_nombre = tk.StringVar(value=candidato.get('nombre', ''))
        entry_nom = ttk.Entry(frm, textvariable=self.var_nombre, width=22)
        entry_nom.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="Alias (máx 3):").grid(row=0, column=2, sticky="w", padx=(8, 4), pady=4)
        alias_inicial = candidato.get('alias') or generar_alias_defecto(self.var_nombre.get())
        self.var_alias = tk.StringVar(value=alias_inicial)
        self._alias_editado_manual = bool(candidato.get('alias'))
        entry_alias = ttk.Entry(frm, textvariable=self.var_alias, width=8)
        entry_alias.grid(row=0, column=3, sticky="w", pady=4)

        def _al_teclear_alias(_e):
            self._alias_editado_manual = True

        entry_alias.bind("<Key>", _al_teclear_alias)

        def _al_cambiar_nombre(*_):
            if not self._alias_editado_manual:
                self.var_alias.set(generar_alias_defecto(self.var_nombre.get()))

        self.var_nombre.trace_add("write", _al_cambiar_nombre)

        ttk.Label(frm, text="Archivo .s1p:").grid(row=1, column=0, sticky="w", pady=4)
        self.var_archivo = tk.StringVar(value=candidato.get('archivo', ''))
        ttk.Entry(frm, textvariable=self.var_archivo, width=34).grid(row=1, column=1, sticky="ew", pady=4)
        _crear_indicador_archivo(
            frm, self.var_archivo, getattr(master, 'var_carpeta_datos', None)).grid(row=1, column=2, padx=(4, 0))
        ttk.Button(frm, text="Examinar...", command=self._examinar).grid(row=1, column=3, padx=(6, 0))

        ttk.Label(frm, text="Modelo teórico:").grid(row=2, column=0, sticky="w", pady=4)
        modelos = gf.listar_modelos_teoricos()
        self.var_modelo_etiqueta = tk.StringVar(value=gf.etiqueta_de_modelo(candidato.get('modelo', 'agua')))
        self.combo_modelo = ttk.Combobox(
            frm, textvariable=self.var_modelo_etiqueta, state="readonly",
            values=[et for _, et in modelos], width=32)
        self.combo_modelo.grid(row=2, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(frm, text="Temperatura (°C):").grid(row=3, column=0, sticky="w", pady=4)
        self.var_temperatura = tk.DoubleVar(value=float(candidato.get('temperatura', 25.0)))
        spin_t = ttk.Spinbox(frm, from_=-20, to=150, increment=0.5, textvariable=self.var_temperatura, width=10)
        spin_t.grid(row=3, column=1, sticky="w", pady=4)

        self.var_habilitado = tk.BooleanVar(value=bool(candidato.get('habilitado', True)))
        ttk.Checkbutton(frm, text="Habilitado para la combinatoria", variable=self.var_habilitado).grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(8, 4))

        frm_btn = ttk.Frame(frm)
        frm_btn.grid(row=5, column=0, columnspan=4, sticky="e", pady=(12, 0))
        ttk.Button(frm_btn, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(frm_btn, text="Guardar", command=self._guardar).pack(side="right")

        self.grab_set()
        self.wait_window()

    def _examinar(self):
        inicial = self.carpeta_datos if os.path.isdir(self.carpeta_datos) else "."
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo .s1p",
            initialdir=inicial, parent=self,
            filetypes=[("Archivos Touchstone S1P", "*.s1p"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            self.var_archivo.set(_normalizar_ruta_s1p(ruta, self.carpeta_datos))
            if not self.var_nombre.get().strip():
                base = os.path.splitext(os.path.basename(ruta))[0]
                nom = base.replace('_', ' ').capitalize()
                self.var_nombre.set(nom)
                if not self._alias_editado_manual:
                    self.var_alias.set(generar_alias_defecto(nom))

    def _guardar(self):
        nombre = self.var_nombre.get().strip()
        archivo = self.var_archivo.get().strip()
        modelo = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get())
        if not nombre:
            messagebox.showwarning("Falta nombre", "Ingresá un nombre identificador para el líquido.", parent=self)
            return
        if not archivo:
            messagebox.showwarning("Falta archivo", "Seleccioná el archivo .s1p correspondiente.", parent=self)
            return

        alias = self.var_alias.get().strip()[:3]
        if not alias:
            alias = generar_alias_defecto(nombre)
        else:
            alias = alias.capitalize()

        try:
            temp = float(self.var_temperatura.get())
        except (ValueError, tk.TclError):
            temp = 25.0

        self.resultado = {
            'nombre': nombre,
            'alias': alias,
            'archivo': archivo,
            'modelo': modelo,
            'temperatura': temp,
            'habilitado': bool(self.var_habilitado.get()),
        }
        self.destroy()


# ===========================================================================
# Aplicación Principal GUI de Combinatoria
# ===========================================================================
class AppCombinatoria(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sonda 2026 — Selección Óptima de Patrones (Combinatoria)")
        self.geometry("1180x840")
        self.minsize(980, 720)

        self._cola_hilo = queue.Queue()
        self._evento_cancelar = threading.Event()
        self._hilo_ejecucion = None

        self._datos_resultados = None
        self._ultima_config = None
        self.var_criterio_orden = tk.StringVar(value="Promedio Global")
        self.var_omitir_redundantes = tk.BooleanVar(value=True)
        self.var_excluir_patrones_pdf = tk.BooleanVar(value=True)
        self.var_orden_combos = tk.StringVar(value="Menor error (mejor primero)")
        self._mapa_display_a_combo_id = {}
        self._combo_id_activo = None

        self.var_limitar_y_tab2 = tk.BooleanVar(value=True)
        self.var_limitar_y_tab3 = tk.BooleanVar(value=True)
        self.var_filtro_palabra_combo = tk.StringVar(value="")
        self.var_filtro_liquido_combo = tk.StringVar(value="Todos")
        self.var_comparar_activo = tk.BooleanVar(value=False)
        self.var_combo_b_seleccionada = tk.StringVar(value="")
        self._combo_id_b_activo = None
        self._todas_opciones_combos = []

        self.var_carpeta_salida = tk.StringVar(value=r"..\salidas")
        self.var_guardar_fechayhora = tk.BooleanVar(value=True)
        self.var_exportar_imagenes_csv = tk.BooleanVar(value=True)
        self._ultima_carpeta_salida = None

        self._candidatos = [
            {'nombre': "Agua", 'alias': "Agu", 'archivo': "agua.s1p", 'modelo': "agua", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Alcohol etilico", 'alias': "Eti", 'archivo': "etanol.s1p", 'modelo': "alcohol_etilico", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Alcohol isopropilico", 'alias': "Iso", 'archivo': "isopropilico.s1p", 'modelo': "alcohol_isopropilico", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Butanol", 'alias': "But", 'archivo': "butanol.s1p", 'modelo': "butanol", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Ciclohexano", 'alias': "Cic", 'archivo': "ciclohexano.s1p", 'modelo': "ciclohexano", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Acetona", 'alias': "Ace", 'archivo': "acetona.s1p", 'modelo': "acetona", 'temperatura': 25.0, 'habilitado': True},
        ]

        self._construir_estilos()
        self._construir_interfaz()
        self._iniciar_poll_cola()

    def _construir_estilos(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("Titulo.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Subtitulo.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Ayuda.TLabel", font=("Segoe UI", 9, "italic"), foreground="#444444")
        style.configure("Ejecutar.TButton", font=("Segoe UI", 10, "bold"), foreground="#004d40")

    def _construir_interfaz(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_config = ttk.Frame(self.notebook, padding=10)
        self.tab_ranking = ttk.Frame(self.notebook, padding=10)
        self.tab_combinacion = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_config, text="  1. Configuración y Ejecución  ")
        self.notebook.add(self.tab_ranking, text="  2. Ranking por Muestra  ")
        self.notebook.add(self.tab_combinacion, text="  3. Resultados por Combinación  ")

        self._construir_tab_config()
        self._construir_tab_ranking()
        self._construir_tab_combinacion()

    # -----------------------------------------------------------------------
    # PESTAÑA 1: Configuración y Ejecución
    # -----------------------------------------------------------------------
    def _construir_tab_config(self):
        f = self.tab_config
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(5, weight=1)

        # Panel Superior Izquierdo: Archivos Base y Frecuencias
        lf_base = ttk.LabelFrame(f, text="Archivos Base y Frecuencias", padding=8)
        lf_base.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        lf_base.columnconfigure(1, weight=1)

        ttk.Label(lf_base, text="Carpeta de datos:").grid(row=0, column=0, sticky="w", pady=3)
        self.var_carpeta_datos = tk.StringVar(value=".")
        ttk.Entry(lf_base, textvariable=self.var_carpeta_datos).grid(row=0, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Button(lf_base, text="Examinar...", command=self._examinar_carpeta_datos).grid(row=0, column=3, padx=(4, 0))
        self.var_carpeta_datos.trace_add("write", lambda *_: self._refrescar_tabla_candidatos())

        ttk.Label(lf_base, text="Cortocircuito (.s1p):").grid(row=1, column=0, sticky="w", pady=3)
        self.var_archivo_corto = tk.StringVar(value="corto.s1p")
        ttk.Entry(lf_base, textvariable=self.var_archivo_corto).grid(row=1, column=1, sticky="ew", pady=3)
        _crear_indicador_archivo(lf_base, self.var_archivo_corto, self.var_carpeta_datos).grid(row=1, column=2, padx=(2, 0))
        ttk.Button(lf_base, text="Examinar...", command=lambda: self._examinar_archivo_base(self.var_archivo_corto)).grid(row=1, column=3, padx=(4, 0))

        ttk.Label(lf_base, text="Circuito abierto (.s1p):").grid(row=2, column=0, sticky="w", pady=3)
        self.var_archivo_aire = tk.StringVar(value="aire.s1p")
        ttk.Entry(lf_base, textvariable=self.var_archivo_aire).grid(row=2, column=1, sticky="ew", pady=3)
        _crear_indicador_archivo(lf_base, self.var_archivo_aire, self.var_carpeta_datos).grid(row=2, column=2, padx=(2, 0))
        ttk.Button(lf_base, text="Examinar...", command=lambda: self._examinar_archivo_base(self.var_archivo_aire)).grid(row=2, column=3, padx=(4, 0))

        frm_frecs = ttk.Frame(lf_base)
        frm_frecs.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 2))
        ttk.Label(frm_frecs, text="F min (GHz):").pack(side="left")
        self.var_f_min = tk.DoubleVar(value=0.5)
        ttk.Spinbox(frm_frecs, from_=0.001, to=50, increment=0.1, textvariable=self.var_f_min, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(frm_frecs, text="F max (GHz):").pack(side="left")
        self.var_f_max = tk.DoubleVar(value=6.0)
        ttk.Spinbox(frm_frecs, from_=0.001, to=50, increment=0.1, textvariable=self.var_f_max, width=7).pack(side="left", padx=(4, 12))

        self.var_escala_log = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_frecs, text="Escala log en f", variable=self.var_escala_log).pack(side="left")

        frm_ints = ttk.Frame(lf_base)
        frm_ints.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        ttk.Label(frm_ints, text="F interm 1 (GHz):").pack(side="left")
        self.var_f_int1 = tk.DoubleVar(value=2.0)
        ttk.Spinbox(frm_ints, from_=0.001, to=50, increment=0.1, textvariable=self.var_f_int1, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(frm_ints, text="F interm 2 (GHz):").pack(side="left")
        self.var_f_int2 = tk.DoubleVar(value=4.0)
        ttk.Spinbox(frm_ints, from_=0.001, to=50, increment=0.1, textvariable=self.var_f_int2, width=7).pack(side="left", padx=(4, 0))
        _tooltip(frm_ints, "Define 3 intervalos contiguos [F_min, F_int1], [F_int1, F_int2] y [F_int2, F_max] para evaluar el error ponderado.")

        ttk.Label(lf_base, text="Carpeta de salida:").grid(row=5, column=0, sticky="w", pady=(6, 3))
        ttk.Entry(lf_base, textvariable=self.var_carpeta_salida).grid(row=5, column=1, columnspan=2, sticky="ew", pady=(6, 3))
        ttk.Button(lf_base, text="Examinar...", command=self._examinar_carpeta_salida).grid(row=5, column=3, padx=(4, 0), pady=(6, 3))

        frm_salida_opts = ttk.Frame(lf_base)
        frm_salida_opts.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        ttk.Checkbutton(frm_salida_opts, text=r"Guardar en ..\salidas\fechayhora",
                        variable=self.var_guardar_fechayhora).pack(side="left")
        ttk.Checkbutton(frm_salida_opts, text="Exportar figuras y .csv",
                        variable=self.var_exportar_imagenes_csv).pack(side="left", padx=(10, 0))

        # Panel Superior Derecho: Métodos y Acondicionamiento
        lf_metodos = ttk.LabelFrame(f, text="Métodos de Cálculo y Acondicionamiento Gn", padding=8)
        lf_metodos.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        self.var_metodo_comp_sin = tk.BooleanVar(value=True)
        self.var_metodo_comp_con = tk.BooleanVar(value=True)
        self.var_metodo_mc_sin = tk.BooleanVar(value=False)
        self.var_metodo_mc_con = tk.BooleanVar(value=False)
        self.var_metodo_simpl = tk.BooleanVar(value=False)

        ttk.Checkbutton(lf_metodos, text="Método completo (4 patrones) — Gn Original",
                        variable=self.var_metodo_comp_sin).pack(anchor="w", pady=1)
        ttk.Checkbutton(lf_metodos, text="Método completo (4 patrones) — Con Acondicionamiento de Gn",
                        variable=self.var_metodo_comp_con, command=self._al_cambiar_acond).pack(anchor="w", pady=1)
        ttk.Checkbutton(lf_metodos, text="Cuadrados mínimos (≥5 patrones) — Gn Original",
                        variable=self.var_metodo_mc_sin, command=self._al_cambiar_acond).pack(anchor="w", pady=1)
        ttk.Checkbutton(lf_metodos, text="Cuadrados mínimos (≥5 patrones) — Con Acondicionamiento de Gn",
                        variable=self.var_metodo_mc_con, command=self._al_cambiar_acond).pack(anchor="w", pady=1)
        ttk.Checkbutton(lf_metodos, text="Método simplificado (3 patrones — sólo P3)",
                        variable=self.var_metodo_simpl).pack(anchor="w", pady=1)

        frm_mc_k = ttk.Frame(lf_metodos)
        frm_mc_k.pack(fill="x", pady=(3, 1))
        ttk.Label(frm_mc_k, text="Líquidos patrón MC (k):").pack(side="left")
        self.var_mc_k_liquidos = tk.StringVar(value="Todos (3 a M-1)")
        self.combo_mc_k = ttk.Combobox(
            frm_mc_k, textvariable=self.var_mc_k_liquidos, state="disabled",
            values=["Todos (3 a M-1)", "3 líquidos (5 patrones)", "4 líquidos (6 patrones)", "5 líquidos (7 patrones)", "Máximo / Leave-one-out (M-1)"],
            width=28
        )
        self.combo_mc_k.pack(side="left", padx=(6, 0))
        _tooltip(
            self.combo_mc_k,
            "Cantidad k de líquidos candidatos a usar como patrones en cada combinación de "
            "cuadrados mínimos (además de Corto y Aire). Los M - k líquidos restantes se evalúan como muestras."
        )

        frm_semilla = ttk.Frame(lf_metodos)
        frm_semilla.pack(fill="x", pady=(4, 2))
        ttk.Label(frm_semilla, text="Arranque marcha Gn:").pack(side="left")
        self.var_estrategia_gn = tk.StringVar(value="minimo_gn")
        ttk.Radiobutton(frm_semilla, text="Automático (mín |Gn|)", variable=self.var_estrategia_gn,
                        value="minimo_gn").pack(side="left", padx=(6, 6))
        ttk.Radiobutton(frm_semilla, text="Clásico (f máx)", variable=self.var_estrategia_gn,
                        value="frecuencia_maxima").pack(side="left")

        self.frm_sub_acond = ttk.Frame(lf_metodos, padding=(16, 2, 4, 2))
        self.frm_sub_acond.pack(fill="x", expand=True)

        ttk.Label(self.frm_sub_acond, text="Tipo:").grid(row=0, column=0, sticky="w", pady=2)
        self.var_acond_tipo = tk.StringVar(value="Transicion cuadratica (f^2)")
        self.combo_acond_tipo = ttk.Combobox(
            self.frm_sub_acond, textvariable=self.var_acond_tipo, state="readonly",
            values=list(_TIPOS_ACOND_MAP.keys()), width=24)
        self.combo_acond_tipo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        self.combo_acond_tipo.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_acond_modo())

        ttk.Label(self.frm_sub_acond, text="Modo:").grid(row=1, column=0, sticky="w", pady=2)
        self.var_acond_modo = tk.StringVar(value="automatico")
        frm_radio = ttk.Frame(self.frm_sub_acond)
        frm_radio.grid(row=1, column=1, columnspan=3, sticky="w", pady=2)
        ttk.Radiobutton(frm_radio, text="Automático", variable=self.var_acond_modo,
                        value="automatico", command=self._al_cambiar_acond_modo).pack(side="left")
        ttk.Radiobutton(frm_radio, text="Manual", variable=self.var_acond_modo,
                        value="manual", command=self._al_cambiar_acond_modo).pack(side="left", padx=8)

        self.lbl_umbral = ttk.Label(self.frm_sub_acond, text="Umbral (%):")
        self.lbl_umbral.grid(row=2, column=0, sticky="w", pady=2)
        self.var_acond_umbral = tk.DoubleVar(value=30.0)
        self.spin_umbral = ttk.Spinbox(self.frm_sub_acond, from_=1.0, to=200.0, increment=5.0,
                                       textvariable=self.var_acond_umbral, width=8)
        self.spin_umbral.grid(row=2, column=1, sticky="w", pady=2)

        self.lbl_f1 = ttk.Label(self.frm_sub_acond, text="f1 (GHz):")
        self.var_acond_f1 = tk.DoubleVar(value=0.0)
        self.spin_f1 = ttk.Spinbox(self.frm_sub_acond, from_=0.0, to=20.0, increment=0.05,
                                   textvariable=self.var_acond_f1, width=7)

        self.lbl_f2 = ttk.Label(self.frm_sub_acond, text="f2 (GHz):")
        self.var_acond_f2 = tk.DoubleVar(value=0.5)
        self.spin_f2 = ttk.Spinbox(self.frm_sub_acond, from_=0.01, to=20.0, increment=0.05,
                                   textvariable=self.var_acond_f2, width=7)

        # Panel Central: Tabla de Candidatos a Patrones
        lf_cand = ttk.LabelFrame(f, text="Líquidos Candidatos a Patrones (Pool de Evaluación)", padding=8)
        lf_cand.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=6)
        lf_cand.columnconfigure(0, weight=1)

        cols = ("hab", "alias", "nombre", "archivo", "existe", "modelo", "temp")
        self.tree_cand = ttk.Treeview(lf_cand, columns=cols, show="headings", height=6, selectmode="browse")
        self.tree_cand.heading("hab", text="Activo")
        self.tree_cand.heading("alias", text="Alias")
        self.tree_cand.heading("nombre", text="Nombre")
        self.tree_cand.heading("archivo", text="Archivo .s1p")
        self.tree_cand.heading("existe", text="Estado")
        self.tree_cand.heading("modelo", text="Modelo Teórico")
        self.tree_cand.heading("temp", text="T (°C)")

        self.tree_cand.column("hab", width=50, anchor="center")
        self.tree_cand.column("alias", width=55, anchor="center")
        self.tree_cand.column("nombre", width=155, anchor="w")
        self.tree_cand.column("archivo", width=220, anchor="w")
        self.tree_cand.column("existe", width=60, anchor="center")
        self.tree_cand.column("modelo", width=190, anchor="w")
        self.tree_cand.column("temp", width=65, anchor="center")

        self.tree_cand.grid(row=0, column=0, sticky="nsew")
        sc_cand = ttk.Scrollbar(lf_cand, orient="vertical", command=self.tree_cand.yview)
        sc_cand.grid(row=0, column=1, sticky="ns")
        self.tree_cand.configure(yscrollcommand=sc_cand.set)
        self.tree_cand.bind("<Double-1>", lambda _e: self._editar_candidato())

        frm_btn_cand = ttk.Frame(lf_cand)
        frm_btn_cand.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(frm_btn_cand, text="Agregar...", command=self._agregar_candidato).pack(side="left")
        ttk.Button(frm_btn_cand, text="Editar...", command=self._editar_candidato).pack(side="left", padx=4)
        ttk.Button(frm_btn_cand, text="Eliminar", command=self._eliminar_candidato).pack(side="left", padx=4)
        ttk.Button(frm_btn_cand, text="Toggle Activo", command=self._toggle_candidato).pack(side="left", padx=4)
        ttk.Button(frm_btn_cand, text="Cargar 6 candidatos estándar", command=self._cargar_candidatos_default).pack(side="right")

        self._refrescar_tabla_candidatos()

        # Botones de Acción y Progreso
        frm_ctrl = ttk.Frame(f)
        frm_ctrl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        self.btn_ejecutar = ttk.Button(frm_ctrl, text="▶  Ejecutar Combinatoria",
                                       style="Ejecutar.TButton", command=self._ejecutar_analisis)
        self.btn_ejecutar.pack(side="left", padx=(0, 8))
        self.btn_cancelar = ttk.Button(frm_ctrl, text="■  Cancelar", state="disabled", command=self._cancelar_analisis)
        self.btn_cancelar.pack(side="left", padx=(0, 14))

        ttk.Separator(frm_ctrl, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(frm_ctrl, text="Intervalo Informe PDF:").pack(side="left", padx=(4, 4))
        self.combo_criterio_tab1 = ttk.Combobox(
            frm_ctrl, textvariable=self.var_criterio_orden, state="readonly",
            values=["Promedio Global", "Intervalo 1", "Intervalo 2", "Intervalo 3"], width=24)
        self.combo_criterio_tab1.pack(side="left", padx=(0, 8))
        self.combo_criterio_tab1.bind("<<ComboboxSelected>>", self._al_cambiar_criterio_ranking)
        self.btn_pdf_tab1 = ttk.Button(
            frm_ctrl, text="📄 Generar Informe PDF (Top 3)...",
            state="disabled", command=self._generar_informe_pdf)
        self.btn_pdf_tab1.pack(side="left")
        ttk.Button(frm_ctrl, text="📂 Abrir salida", command=self._abrir_carpeta_salida).pack(side="left", padx=(4, 6))
        ttk.Checkbutton(frm_ctrl, text="Omitir redundantes (*)", variable=self.var_omitir_redundantes).pack(side="left", padx=(4, 4))
        ttk.Checkbutton(frm_ctrl, text="Excluir patrones en reporte", variable=self.var_excluir_patrones_pdf).pack(side="left", padx=(4, 4))

        ttk.Button(frm_ctrl, text="Guardar Config...", command=self._guardar_config).pack(side="right")
        ttk.Button(frm_ctrl, text="Cargar Config...", command=self._cargar_config).pack(side="right", padx=6)

        self.var_paso_actual = tk.StringVar(value="Listo.")
        ttk.Label(f, textvariable=self.var_paso_actual, style="Ayuda.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.progreso = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.progreso.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 6))

        # Consola de ejecución
        lf_consola = ttk.LabelFrame(f, text="Consola de Registro", padding=6)
        lf_consola.grid(row=5, column=0, columnspan=2, sticky="nsew")
        f.rowconfigure(5, weight=1)
        lf_consola.columnconfigure(0, weight=1)
        lf_consola.rowconfigure(0, weight=1)

        self.txt_consola = tk.Text(lf_consola, height=7, wrap="word", state="disabled", font=("Courier New", 9))
        self.txt_consola.grid(row=0, column=0, sticky="nsew")
        sc_consola = ttk.Scrollbar(lf_consola, orient="vertical", command=self.txt_consola.yview)
        sc_consola.grid(row=0, column=1, sticky="ns")
        self.txt_consola.configure(yscrollcommand=sc_consola.set)

    def _al_cambiar_acond(self):
        activo = self.var_metodo_comp_con.get() or self.var_metodo_mc_con.get()
        estado = "normal" if activo else "disabled"
        self.combo_acond_tipo.configure(state="readonly" if activo else "disabled")
        for w in self.frm_sub_acond.winfo_children():
            try:
                if w is not self.combo_acond_tipo:
                    w.configure(state=estado)
            except Exception:
                pass
        activo_mc = self.var_metodo_mc_sin.get() or self.var_metodo_mc_con.get()
        if hasattr(self, 'combo_mc_k'):
            self.combo_mc_k.configure(state="readonly" if activo_mc else "disabled")

    def _actualizar_opciones_mc_k(self):
        if not hasattr(self, 'combo_mc_k'):
            return
        n_activos = sum(1 for c in self._candidatos if c.get('habilitado', True))
        max_k = max(3, n_activos)
        opciones = ["Todos (3 a M)"]
        for k in range(3, max_k + 1):
            if k == n_activos:
                opciones.append(f"{k} líquidos ({k + 2} patrones - todos)")
            else:
                opciones.append(f"{k} líquidos ({k + 2} patrones)")
        opciones.append("Todos los líquidos (M)")
        if n_activos > 3:
            opciones.append("Leave-one-out (M-1)")
        self.combo_mc_k.configure(values=opciones)
        if self.var_mc_k_liquidos.get() not in opciones:
            self.var_mc_k_liquidos.set(opciones[0])

    def _obtener_mc_k_config(self):
        val = (self.var_mc_k_liquidos.get() or "").strip()
        if val.startswith("Todos ("):
            return "todos"
        if val.startswith("Todos los líquidos") or val.startswith("Todos los liquidos"):
            return "todos_m"
        if "leave-one-out" in val.lower():
            return "leave_one_out"
        try:
            return int(val.split()[0])
        except (ValueError, IndexError):
            return "todos"

    def _setear_mc_k_desde_config(self, val):
        self._actualizar_opciones_mc_k()
        s = str(val).strip().lower()
        if s in ("todos", "all", ""):
            self.var_mc_k_liquidos.set("Todos (3 a M)")
        elif s in ("todos_m", "todos_los_liquidos", "m"):
            self.var_mc_k_liquidos.set("Todos los líquidos (M)")
        elif s in ("max", "leave_one_out", "m-1", "n-1"):
            self.var_mc_k_liquidos.set("Leave-one-out (M-1)")
        else:
            try:
                k_int = int(s.split()[0])
                n_activos = sum(1 for c in self._candidatos if c.get('habilitado', True))
                if k_int == n_activos:
                    etiqueta = f"{k_int} líquidos ({k_int + 2} patrones - todos)"
                else:
                    etiqueta = f"{k_int} líquidos ({k_int + 2} patrones)"
                if etiqueta in self.combo_mc_k.cget("values"):
                    self.var_mc_k_liquidos.set(etiqueta)
                else:
                    self.var_mc_k_liquidos.set("Todos (3 a M)")
            except (ValueError, IndexError):
                self.var_mc_k_liquidos.set("Todos (3 a M)")

    def _al_cambiar_acond_modo(self):
        tipo_clave = _TIPOS_ACOND_MAP.get(self.var_acond_tipo.get(), "transicion_polinomica_cero")
        if self.var_acond_modo.get() == "automatico":
            self.lbl_f1.grid_remove()
            self.spin_f1.grid_remove()
            self.lbl_f2.grid_remove()
            self.spin_f2.grid_remove()
            self.lbl_umbral.grid(row=2, column=0, sticky="w", pady=2)
            self.spin_umbral.grid(row=2, column=1, sticky="w", pady=2)
        else:
            self.lbl_umbral.grid_remove()
            self.spin_umbral.grid_remove()
            if tipo_clave == "cero_clamping":
                self.lbl_f1.grid_remove()
                self.spin_f1.grid_remove()
                self.lbl_f2.configure(text="f corte (GHz):")
                self.lbl_f2.grid(row=2, column=0, sticky="w", pady=2)
                self.spin_f2.grid(row=2, column=1, sticky="w", pady=2)
            else:
                self.lbl_f1.configure(text="f1 (GHz):")
                self.lbl_f1.grid(row=2, column=0, sticky="w", pady=2)
                self.spin_f1.grid(row=2, column=1, sticky="w", pady=2)
                self.lbl_f2.configure(text="f2 (GHz):")
                self.lbl_f2.grid(row=2, column=2, sticky="w", padx=(6, 0), pady=2)
                self.spin_f2.grid(row=2, column=3, sticky="w", pady=2)

    def _examinar_carpeta_datos(self):
        d = filedialog.askdirectory(title="Carpeta de datos Touchstone", initialdir=self.var_carpeta_datos.get(), parent=self)
        if d:
            self.var_carpeta_datos.set(d)

    def _examinar_archivo_base(self, var_target):
        carpeta = self.var_carpeta_datos.get().strip()
        d = carpeta if os.path.isdir(carpeta) else "."
        r = filedialog.askopenfilename(title="Seleccionar .s1p", initialdir=d, parent=self,
                                       filetypes=[("Archivos Touchstone S1P", "*.s1p"), ("Todos", "*.*")])
        if r:
            var_target.set(_normalizar_ruta_s1p(r, carpeta))

    def _refrescar_tabla_candidatos(self):
        if not hasattr(self, 'tree_cand'):
            return
        for item in self.tree_cand.get_children():
            self.tree_cand.delete(item)
        carpeta = self.var_carpeta_datos.get().strip() if hasattr(self, 'var_carpeta_datos') else ""
        for i, cand in enumerate(self._candidatos):
            hab_str = "✓" if cand.get('habilitado', True) else "✗"
            alias_str = cand.get('alias') or generar_alias_defecto(cand['nombre'])
            archivo = cand.get('archivo', '')
            ruta_completa = os.path.join(carpeta, archivo) if archivo else ""
            existe_str = "✓ OK" if (archivo and os.path.isfile(ruta_completa)) else "✗ No hallado"
            et_mod = gf.etiqueta_de_modelo(cand.get('modelo'))
            self.tree_cand.insert("", "end", iid=str(i), values=(
                hab_str, alias_str, cand['nombre'], archivo, existe_str, et_mod, f"{cand.get('temperatura', 25.0):.1f}"
            ))
        self._actualizar_opciones_mc_k()
        self._actualizar_opciones_filtro_liquidos()

    def _obtener_alias(self, nombre: str) -> str:
        """Devuelve el alias de máximo 3 letras asociado a un nombre de líquido."""
        if not nombre or nombre == "—":
            return "—"
        for c in self._candidatos:
            if c.get('nombre') == nombre:
                return c.get('alias') or generar_alias_defecto(nombre)
        return generar_alias_defecto(nombre)

    def _formatear_patrones_alias(self, c_o_item) -> str:
        """Formatea la lista de patrones de calibración usando los alias de los líquidos."""
        if c_o_item.get('es_mc'):
            pat_str = c_o_item.get('patrones_str') or f"{c_o_item.get('p3', '')} + {c_o_item.get('p4', '')}"
            partes = [p.strip() for p in pat_str.split('+')]
            partes_alias = [self._obtener_alias(p) for p in partes]
            return " + ".join(partes_alias)
        else:
            p3_alias = self._obtener_alias(c_o_item.get('p3', ''))
            p4_val = c_o_item.get('p4')
            if p4_val and p4_val != '—':
                p4_alias = self._obtener_alias(p4_val)
                return f"P3: {p3_alias} | P4: {p4_alias}"
            return f"P3: {p3_alias}"

    def _actualizar_opciones_filtro_liquidos(self):
        """Actualiza el combobox de filtro de líquidos en Tab 3."""
        if not hasattr(self, 'combo_filtro_liquido'):
            return
        vals = ["Todos"]
        for c in self._candidatos:
            if c.get('habilitado', True):
                alias = c.get('alias') or generar_alias_defecto(c.get('nombre', ''))
                vals.append(f"{c['nombre']} ({alias})")
        self.combo_filtro_liquido.configure(values=vals)
        if self.var_filtro_liquido_combo.get() not in vals:
            self.var_filtro_liquido_combo.set("Todos")

    def _agregar_candidato(self):
        d = DialogoCandidato(self, carpeta_datos=self.var_carpeta_datos.get())
        if d.resultado:
            self._candidatos.append(d.resultado)
            self._refrescar_tabla_candidatos()

    def _editar_candidato(self):
        sel = self.tree_cand.selection()
        if not sel:
            return
        idx = int(sel[0])
        cand = self._candidatos[idx]
        d = DialogoCandidato(self, candidato=cand, carpeta_datos=self.var_carpeta_datos.get())
        if d.resultado:
            self._candidatos[idx] = d.resultado
            self._refrescar_tabla_candidatos()

    def _eliminar_candidato(self):
        sel = self.tree_cand.selection()
        if not sel:
            return
        idx = int(sel[0])
        del self._candidatos[idx]
        self._refrescar_tabla_candidatos()

    def _toggle_candidato(self):
        sel = self.tree_cand.selection()
        if not sel:
            return
        idx = int(sel[0])
        self._candidatos[idx]['habilitado'] = not self._candidatos[idx].get('habilitado', True)
        self._refrescar_tabla_candidatos()

    def _cargar_candidatos_default(self):
        self._candidatos = [
            {'nombre': "Agua", 'alias': "Agu", 'archivo': "agua.s1p", 'modelo': "agua", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Alcohol etilico", 'alias': "Eti", 'archivo': "etanol.s1p", 'modelo': "alcohol_etilico", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Alcohol isopropilico", 'alias': "Iso", 'archivo': "isopropilico.s1p", 'modelo': "alcohol_isopropilico", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Butanol", 'alias': "But", 'archivo': "butanol.s1p", 'modelo': "butanol", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Ciclohexano", 'alias': "Cic", 'archivo': "ciclohexano.s1p", 'modelo': "ciclohexano", 'temperatura': 25.0, 'habilitado': True},
            {'nombre': "Acetona", 'alias': "Ace", 'archivo': "acetona.s1p", 'modelo': "acetona", 'temperatura': 25.0, 'habilitado': True},
        ]
        self._refrescar_tabla_candidatos()

    # -----------------------------------------------------------------------
    # PESTAÑA 2: Ranking por Muestra
    # -----------------------------------------------------------------------
    def _construir_tab_ranking(self):
        f = self.tab_ranking
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)

        # Barra superior de filtros
        frm_top = ttk.Frame(f)
        frm_top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frm_top, text="Seleccionar Muestra / Líquido evaluado:", style="Subtitulo.TLabel").pack(side="left")
        self.var_muestra_ranking = tk.StringVar()
        self.combo_muestra_ranking = ttk.Combobox(frm_top, textvariable=self.var_muestra_ranking, state="readonly", width=22)
        self.combo_muestra_ranking.pack(side="left", padx=(8, 16))
        self.combo_muestra_ranking.bind("<<ComboboxSelected>>", self._al_cambiar_muestra_ranking)

        ttk.Label(frm_top, text="Rol:").pack(side="left")
        self.var_filtro_rol = tk.StringVar(value="Todos")
        self.combo_filtro_rol = ttk.Combobox(
            frm_top, textvariable=self.var_filtro_rol, state="readonly",
            values=["Todos", "Sólo Muestras", "Sólo Patrones"], width=15)
        self.combo_filtro_rol.pack(side="left", padx=(4, 14))
        self.combo_filtro_rol.bind("<<ComboboxSelected>>", self._al_cambiar_criterio_ranking)

        ttk.Label(frm_top, text="Intervalo / Ordenar por:").pack(side="left")
        self.combo_criterio = ttk.Combobox(
            frm_top, textvariable=self.var_criterio_orden, state="readonly",
            values=["Promedio Global", "Intervalo 1", "Intervalo 2", "Intervalo 3"], width=22)
        self.combo_criterio.pack(side="left", padx=(6, 10))
        self.combo_criterio.bind("<<ComboboxSelected>>", self._al_cambiar_criterio_ranking)

        self.var_orden_asc = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_top, text="Menor error primero", variable=self.var_orden_asc,
                        command=self._al_cambiar_criterio_ranking).pack(side="left")
        ttk.Checkbutton(frm_top, text="Omitir redundantes (*)", variable=self.var_omitir_redundantes,
                        command=self._al_cambiar_criterio_ranking).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(frm_top, text="Excluir patrones en reporte", variable=self.var_excluir_patrones_pdf,
                        command=self._al_cambiar_criterio_ranking).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(frm_top, text="Limitar eje Y", variable=self.var_limitar_y_tab2,
                        command=self._al_cambiar_limitar_y_tab2).pack(side="left", padx=(10, 0))

        self.btn_pdf_tab2 = ttk.Button(
            frm_top, text="📄 Generar Informe PDF (Top 3)...",
            state="disabled", command=self._generar_informe_pdf)
        self.btn_pdf_tab2.pack(side="right", padx=(8, 0))

        # Panel izquierdo: Treeview con ranking
        lf_tabla = ttk.LabelFrame(f, text="Tabla Clasificatoria de Combinaciones", padding=6)
        lf_tabla.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        lf_tabla.columnconfigure(0, weight=1)
        lf_tabla.rowconfigure(0, weight=1)

        cols_r = ("rank", "rol", "p3", "p4", "metodo", "int1", "int2", "int3", "global")
        self.tree_ranking = ttk.Treeview(lf_tabla, columns=cols_r, show="headings", height=18, selectmode="browse")
        self.tree_ranking.heading("rank", text="#")
        self.tree_ranking.heading("rol", text="Rol")
        self.tree_ranking.heading("p3", text="Patrón 3 (P3)")
        self.tree_ranking.heading("p4", text="Patrón 4 / Adic. MC")
        self.tree_ranking.heading("metodo", text="Método")
        self.tree_ranking.heading("int1", text="Int 1")
        self.tree_ranking.heading("int2", text="Int 2")
        self.tree_ranking.heading("int3", text="Int 3")
        self.tree_ranking.heading("global", text="Score Prom")

        self.tree_ranking.column("rank", width=35, anchor="center")
        self.tree_ranking.column("rol", width=65, anchor="center")
        self.tree_ranking.column("p3", width=105, anchor="w")
        self.tree_ranking.column("p4", width=140, anchor="w")
        self.tree_ranking.column("metodo", width=120, anchor="w")
        self.tree_ranking.column("int1", width=60, anchor="center")
        self.tree_ranking.column("int2", width=60, anchor="center")
        self.tree_ranking.column("int3", width=60, anchor="center")
        self.tree_ranking.column("global", width=70, anchor="center")

        self.tree_ranking.grid(row=0, column=0, sticky="nsew")
        sc_r = ttk.Scrollbar(lf_tabla, orient="vertical", command=self.tree_ranking.yview)
        sc_r.grid(row=0, column=1, sticky="ns")
        self.tree_ranking.configure(yscrollcommand=sc_r.set)
        self.tree_ranking.bind("<<TreeviewSelect>>", self._al_seleccionar_fila_ranking)

        # Panel derecho: Gráfico interactivo de la combinación seleccionada
        lf_graf = ttk.LabelFrame(f, text="Curvas de Permitividad (Medido vs Teórico)", padding=6)
        lf_graf.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        lf_graf.columnconfigure(0, weight=1)
        lf_graf.rowconfigure(0, weight=1)

        self.visor_graf_ranking = VisorFigura(lf_graf)
        self.visor_graf_ranking.grid(row=0, column=0, sticky="nsew")

    # -----------------------------------------------------------------------
    # PESTAÑA 3: Resultados por Combinación
    # -----------------------------------------------------------------------
    def _construir_tab_combinacion(self):
        f = self.tab_combinacion
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=1)

        # Barra superior: Selector de combinación
        frm_top = ttk.Frame(f)
        frm_top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        frm_fila1 = ttk.Frame(frm_top)
        frm_fila1.pack(side="top", fill="x", pady=(0, 3))

        ttk.Label(frm_fila1, text="Combinación A:", style="Subtitulo.TLabel").pack(side="left")
        self.var_combo_seleccionada = tk.StringVar()
        self.combo_combinacion = ttk.Combobox(frm_fila1, textvariable=self.var_combo_seleccionada, state="readonly", width=50)
        self.combo_combinacion.pack(side="left", padx=(6, 12))
        self.combo_combinacion.bind("<<ComboboxSelected>>", self._al_cambiar_combinacion)

        ttk.Label(frm_fila1, text="Ordenar:").pack(side="left")
        self.combo_orden_combos = ttk.Combobox(
            frm_fila1, textvariable=self.var_orden_combos, state="readonly",
            values=["Menor error (mejor primero)", "Mayor error (peor primero)", "Orden de cálculo"],
            width=22
        )
        self.combo_orden_combos.pack(side="left", padx=(4, 12))
        self.combo_orden_combos.bind("<<ComboboxSelected>>", self._al_cambiar_orden_combos)

        self.btn_pdf_combo = ttk.Button(
            frm_fila1, text="📄 Imprimir Informe de Combinación...",
            state="disabled", command=self._generar_informe_combinacion_pdf
        )
        self.btn_pdf_combo.pack(side="right")

        ttk.Button(frm_fila1, text="📂 Abrir salida", command=self._abrir_carpeta_salida).pack(side="right", padx=(0, 6))

        frm_fila2 = ttk.Frame(frm_top)
        frm_fila2.pack(side="top", fill="x", pady=(2, 3))

        ttk.Label(frm_fila2, text="Filtrar texto:").pack(side="left")
        self.entry_filtro_combo = ttk.Entry(frm_fila2, textvariable=self.var_filtro_palabra_combo, width=13)
        self.entry_filtro_combo.pack(side="left", padx=(4, 8))
        self.var_filtro_palabra_combo.trace_add("write", lambda *_: self._filtrar_lista_combinaciones())

        ttk.Label(frm_fila2, text="Líquido patrón:").pack(side="left")
        self.combo_filtro_liquido = ttk.Combobox(frm_fila2, textvariable=self.var_filtro_liquido_combo, state="readonly", width=16)
        self.combo_filtro_liquido.pack(side="left", padx=(4, 8))
        self.combo_filtro_liquido.bind("<<ComboboxSelected>>", lambda *_: self._filtrar_lista_combinaciones())

        ttk.Checkbutton(frm_fila2, text="Limitar eje Y", variable=self.var_limitar_y_tab3,
                        command=self._al_cambiar_limitar_y_tab3).pack(side="left", padx=(4, 8))

        ttk.Checkbutton(frm_fila2, text="Comparar con B:", variable=self.var_comparar_activo,
                        command=self._al_toggle_comparar).pack(side="left", padx=(8, 4))

        self.combo_combinacion_b = ttk.Combobox(frm_fila2, textvariable=self.var_combo_b_seleccionada, state="disabled", width=42)
        self.combo_combinacion_b.pack(side="left", padx=(2, 4))
        self.combo_combinacion_b.bind("<<ComboboxSelected>>", self._al_cambiar_combinacion_b)

        frm_fila3 = ttk.Frame(frm_top)
        frm_fila3.pack(side="top", fill="x", pady=(1, 0))
        self.lbl_score_combo = ttk.Label(frm_fila3, text="", font=("Segoe UI", 9, "bold"), foreground="#004d40")
        self.lbl_score_combo.pack(side="left")
        self.lbl_score_combo_b = ttk.Label(frm_fila3, text="", font=("Segoe UI", 9, "bold"), foreground="#6c3483")
        self.lbl_score_combo_b.pack(side="left", padx=(24, 0))

        # Fila 1 izquierda: Gráfico de Gn(f) + etiqueta de transición
        lf_gn = ttk.LabelFrame(f, text="Diagnóstico de Conductancia Normalizada Gn(f)", padding=6)
        lf_gn.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        lf_gn.columnconfigure(0, weight=1)
        lf_gn.rowconfigure(0, weight=1)
        self.visor_gn = VisorFigura(lf_gn)
        self.visor_gn.grid(row=0, column=0, sticky="nsew")

        self.var_info_acond_gn = tk.StringVar(value="")
        self.lbl_info_acond_gn = ttk.Label(
            lf_gn, textvariable=self.var_info_acond_gn,
            font=("Segoe UI", 9, "bold"), foreground="#1b4f72", justify="left")
        self.lbl_info_acond_gn.grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Fila 1 derecha: Tabla con las muestras evaluadas bajo esa combinación
        lf_muestras = ttk.LabelFrame(f, text="Muestras Evaluadas con esta Calibración", padding=6)
        lf_muestras.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        lf_muestras.columnconfigure(0, weight=1)
        lf_muestras.rowconfigure(0, weight=1)

        cols_m = ("muestra", "rol", "int1", "int2", "int3", "global")
        self.tree_combo_muestras = ttk.Treeview(lf_muestras, columns=cols_m, show="headings", height=8, selectmode="browse")
        self.tree_combo_muestras.heading("muestra", text="Líquido / Muestra")
        self.tree_combo_muestras.heading("rol", text="Rol")
        self.tree_combo_muestras.heading("int1", text="Error Int 1")
        self.tree_combo_muestras.heading("int2", text="Error Int 2")
        self.tree_combo_muestras.heading("int3", text="Error Int 3")
        self.tree_combo_muestras.heading("global", text="Score Global")

        self.tree_combo_muestras.column("muestra", width=130, anchor="w")
        self.tree_combo_muestras.column("rol", width=70, anchor="center")
        self.tree_combo_muestras.column("int1", width=70, anchor="center")
        self.tree_combo_muestras.column("int2", width=70, anchor="center")
        self.tree_combo_muestras.column("int3", width=70, anchor="center")
        self.tree_combo_muestras.column("global", width=80, anchor="center")

        self.tree_combo_muestras.grid(row=0, column=0, sticky="nsew")
        sc_m = ttk.Scrollbar(lf_muestras, orient="vertical", command=self.tree_combo_muestras.yview)
        sc_m.grid(row=0, column=1, sticky="ns")
        self.tree_combo_muestras.configure(yscrollcommand=sc_m.set)
        self.tree_combo_muestras.bind("<<TreeviewSelect>>", self._al_seleccionar_muestra_en_combo)

        # Fila 2 (abajo completo): Gráfico de permitividad comparativa
        lf_curvas = ttk.LabelFrame(f, text="Permitividad de la Muestra Seleccionada", padding=6)
        lf_curvas.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        lf_curvas.columnconfigure(0, weight=1)
        lf_curvas.rowconfigure(0, weight=1)
        self.visor_curvas_combo = VisorFigura(lf_curvas)
        self.visor_curvas_combo.grid(row=0, column=0, sticky="nsew")

    # -----------------------------------------------------------------------
    # Hilo de Ejecución y Comunicación
    # -----------------------------------------------------------------------
    def _iniciar_poll_cola(self):
        try:
            while True:
                msg_tipo, *args = self._cola_hilo.get_nowait()
                if msg_tipo == 'log':
                    texto = args[0]
                    self.txt_consola.configure(state="normal")
                    self.txt_consola.insert("end", texto + "\n")
                    self.txt_consola.see("end")
                    self.txt_consola.configure(state="disabled")
                elif msg_tipo == 'progreso':
                    paso, total, etiq = args
                    if total > 0:
                        self.progreso.configure(maximum=total, value=paso)
                    if etiq:
                        self.var_paso_actual.set(etiq)
                elif msg_tipo == 'ok':
                    res = args[0]
                    self._al_terminar_analisis(res)
                elif msg_tipo == 'cancelado':
                    self.var_paso_actual.set("Análisis cancelado.")
                    self.btn_ejecutar.configure(state="normal")
                    self.btn_cancelar.configure(state="disabled")
                elif msg_tipo == 'error':
                    err_msg, detalle = args
                    self.var_paso_actual.set("Error en el análisis.")
                    self.btn_ejecutar.configure(state="normal")
                    self.btn_cancelar.configure(state="disabled")
                    messagebox.showerror("Error en análisis", f"{err_msg}\n\nDetalle:\n{detalle}", parent=self)
        except queue.Empty:
            pass
        self.after(100, self._iniciar_poll_cola)

    def _ejecutar_analisis(self):
        # Armar config
        try:
            f_min = float(self.var_f_min.get())
            f_max = float(self.var_f_max.get())
            f_int1 = float(self.var_f_int1.get())
            f_int2 = float(self.var_f_int2.get())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Frecuencias inválidas", "Las frecuencias ingresadas deben ser números válidos.", parent=self)
            return

        cfg = {
            'carpeta_datos': self.var_carpeta_datos.get().strip(),
            'archivo_corto': self.var_archivo_corto.get().strip(),
            'archivo_aire': self.var_archivo_aire.get().strip(),
            'f_min_ghz': f_min,
            'f_max_ghz': f_max,
            'f_err_int1_ghz': f_int1,
            'f_err_int2_ghz': f_int2,
            'escala_log_frecuencia': bool(self.var_escala_log.get()),
            'metodos_calculo': {
                'simplificado': bool(self.var_metodo_simpl.get()),
                'completo_sin_acond': bool(self.var_metodo_comp_sin.get()),
                'completo_con_acond': bool(self.var_metodo_comp_con.get()),
                'mc_sin_acond': bool(self.var_metodo_mc_sin.get()),
                'mc_con_acond': bool(self.var_metodo_mc_con.get()),
            },
            'mc_k_liquidos': self._obtener_mc_k_config(),
            'estrategia_gn': self.var_estrategia_gn.get(),
            'acondicionar_gn': {
                'tipo': _TIPOS_ACOND_MAP.get(self.var_acond_tipo.get(), 'transicion_polinomica_cero'),
                'modo': self.var_acond_modo.get(),
                'umbral_pct': float(self.var_acond_umbral.get()),
                'f1_ghz': float(self.var_acond_f1.get()),
                'f2_ghz': float(self.var_acond_f2.get()),
            },
            'candidatos': copy.deepcopy(self._candidatos),
        }
        self._ultima_config = copy.deepcopy(cfg)

        self.btn_ejecutar.configure(state="disabled")
        self.btn_cancelar.configure(state="normal")
        self.progreso.configure(value=0)
        self.txt_consola.configure(state="normal")
        self.txt_consola.delete("1.0", "end")
        self.txt_consola.configure(state="disabled")

        self._evento_cancelar.clear()

        def _worker():
            def log_cb(m):
                self._cola_hilo.put(('log', str(m)))
            def prog_cb(p, t, m):
                self._cola_hilo.put(('progreso', p, t, m))
            try:
                res = ejecutar_analisis_combinatoria(
                    cfg, callback_progreso=prog_cb, callback_log=log_cb,
                    fn_cancelado=self._evento_cancelar.is_set
                )
                if res.get('cancelado'):
                    self._cola_hilo.put(('cancelado', res))
                else:
                    self._cola_hilo.put(('ok', res))
            except Exception as exc:
                self._cola_hilo.put(('error', str(exc), traceback.format_exc()))

        self._hilo_ejecucion = threading.Thread(target=_worker, daemon=True)
        self._hilo_ejecucion.start()

    def _cancelar_analisis(self):
        self._evento_cancelar.set()
        self.btn_cancelar.configure(state="disabled")
        self.var_paso_actual.set("Cancelando...")

    def _al_terminar_analisis(self, resultados):
        self._datos_resultados = resultados
        self.btn_ejecutar.configure(state="normal")
        self.btn_cancelar.configure(state="disabled")
        self.btn_pdf_tab1.configure(state="normal")
        self.btn_pdf_tab2.configure(state="normal")
        self.var_paso_actual.set("Análisis completado con éxito.")

        # Actualizar opciones de intervalos con sus rangos en GHz
        ints_str = resultados.get('intervalos_str', [])
        opciones_crit = ["Promedio Global"]
        for i in range(3):
            rango_txt = f" ({ints_str[i]})" if i < len(ints_str) else ""
            opciones_crit.append(f"Intervalo {i + 1}{rango_txt}")
        self.combo_criterio.configure(values=opciones_crit)
        self.combo_criterio_tab1.configure(values=opciones_crit)

        crit_actual = self.var_criterio_orden.get()
        if crit_actual.startswith("Intervalo 1"):
            self.var_criterio_orden.set(opciones_crit[1])
        elif crit_actual.startswith("Intervalo 2"):
            self.var_criterio_orden.set(opciones_crit[2])
        elif crit_actual.startswith("Intervalo 3"):
            self.var_criterio_orden.set(opciones_crit[3])
        else:
            self.var_criterio_orden.set(opciones_crit[0])

        # Actualizar pestañas de resultados
        muestras_disp = list(resultados.get('resultados_por_muestra', {}).keys())

        self.combo_muestra_ranking.configure(values=muestras_disp)
        if muestras_disp:
            self.combo_muestra_ranking.set(muestras_disp[0])
            self._al_cambiar_muestra_ranking()

        self._refrescar_lista_combinaciones(mantener_seleccion=False)

        # Cambiar de pestaña al ranking automáticamente
        self.notebook.select(self.tab_ranking)

    def _examinar_carpeta_salida(self):
        actual = self.var_carpeta_salida.get().strip() or "."
        d = filedialog.askdirectory(title="Seleccionar carpeta de salida base", initialdir=actual, parent=self)
        if d:
            self.var_carpeta_salida.set(d)

    def _abrir_carpeta_salida(self):
        if hasattr(self, '_ultima_carpeta_salida') and self._ultima_carpeta_salida and os.path.isdir(self._ultima_carpeta_salida):
            gf.abrir_carpeta(self._ultima_carpeta_salida)
            return
        base = self.var_carpeta_salida.get().strip() or r"..\salidas"
        if os.path.isdir(base):
            gf.abrir_carpeta(base)
        else:
            messagebox.showinfo(
                "Carpeta de salida",
                "La carpeta de salida todavía no existe. Generá un informe primero, o creala en disco.",
                parent=self
            )

    def _resolver_carpeta_y_archivo(self, tipo_informe="top3", nombre_sufijo=""):
        usar_fyh = bool(self.var_guardar_fechayhora.get())
        carpeta_base = self.var_carpeta_salida.get().strip() or r"..\salidas"

        if tipo_informe == "top3":
            nombre_pdf = f"informe_combinatoria_top3_{nombre_sufijo}.pdf"
            titulo_dialogo = "Guardar Informe PDF de Combinatoria (Top 3 por líquido)"
        else:
            nombre_pdf = f"informe_combinacion_{nombre_sufijo}.pdf"
            titulo_dialogo = "Guardar Informe PDF de la Combinación"

        if usar_fyh:
            ahora = datetime.now()
            carpeta_dia = ahora.strftime("%Y-%m-%d")
            carpeta_hora = ahora.strftime("%H-%M-%S")
            carpeta_run = os.path.join(carpeta_base, carpeta_dia, carpeta_hora)
            os.makedirs(carpeta_run, exist_ok=True)
            ruta_pdf = os.path.join(carpeta_run, nombre_pdf)
            return ruta_pdf, carpeta_run
        else:
            ruta_pdf = filedialog.asksaveasfilename(
                title=titulo_dialogo,
                initialfile=nombre_pdf,
                defaultextension=".pdf",
                parent=self,
                filetypes=[("Documento PDF", "*.pdf"), ("Todos los archivos", "*.*")]
            )
            if not ruta_pdf:
                return None, None
            carpeta_run = os.path.dirname(os.path.abspath(ruta_pdf))
            return ruta_pdf, carpeta_run

    def _generar_informe_pdf(self):
        if not self._datos_resultados or not self._ultima_config:
            messagebox.showwarning(
                "Sin resultados",
                "Ejecutá primero el análisis combinatorio antes de generar el informe PDF.",
                parent=self
            )
            return

        criterio = self.var_criterio_orden.get()
        sufijo = "global"
        if criterio.startswith("Intervalo 1"):
            sufijo = "int1"
        elif criterio.startswith("Intervalo 2"):
            sufijo = "int2"
        elif criterio.startswith("Intervalo 3"):
            sufijo = "int3"

        ruta_pdf, carpeta_run = self._resolver_carpeta_y_archivo("top3", sufijo)
        if not ruta_pdf:
            return

        try:
            self.var_paso_actual.set("Generando informe PDF...")
            self.update_idletasks()
            omitir_red = getattr(self, 'var_omitir_redundantes', None)
            omitir_val = bool(omitir_red.get()) if omitir_red is not None else True
            excluir_pat = bool(self.var_excluir_patrones_pdf.get()) if hasattr(self, 'var_excluir_patrones_pdf') else True
            exportar = carpeta_run if self.var_exportar_imagenes_csv.get() else None

            generar_reporte_combinatoria_pdf(
                ruta_pdf,
                self._ultima_config,
                self._datos_resultados,
                criterio_orden=criterio,
                top_n=3,
                omitir_redundantes=omitir_val,
                excluir_patrones=excluir_pat,
                carpeta_exportar=exportar,
                limitar_eje_y=self.var_limitar_y_tab2.get(),
            )
            self._ultima_carpeta_salida = carpeta_run
            self.var_paso_actual.set(f"Informe generado en: {os.path.basename(ruta_pdf)}")

            msg = f"Se generó el informe PDF con las 3 mejores combinaciones por líquido\n(criterio: {criterio}):\n\n{ruta_pdf}"
            if exportar:
                msg += f"\n\nSe exportaron las figuras (.png) y tablas (.csv) en:\n{os.path.abspath(carpeta_run)}"
            msg += "\n\n¿Querés abrir el PDF ahora?"

            if messagebox.askyesno("Informe PDF generado", msg, parent=self):
                try:
                    os.startfile(ruta_pdf)
                except Exception:
                    pass
        except Exception as exc:
            self.var_paso_actual.set("Error al generar el informe PDF.")
            messagebox.showerror(
                "Error al generar PDF",
                f"No se pudo generar el informe PDF:\n{exc}\n\n{traceback.format_exc()}",
                parent=self
            )

    # -----------------------------------------------------------------------
    # Métodos de la Pestaña 2: Ranking por Muestra
    # -----------------------------------------------------------------------
    def _al_cambiar_muestra_ranking(self, _event=None):
        if not self._datos_resultados:
            return
        muestra = self.var_muestra_ranking.get()
        items = self._datos_resultados.get('resultados_por_muestra', {}).get(muestra, [])

        filtro_rol = getattr(self, 'var_filtro_rol', None)
        if filtro_rol is not None:
            f_val = filtro_rol.get()
            if f_val == "Sólo Muestras":
                items = [x for x in items if not x.get('es_patron')]
            elif f_val == "Sólo Patrones":
                items = [x for x in items if x.get('es_patron')]

        crit = self.var_criterio_orden.get()
        if crit.startswith("Intervalo 1"):
            clave_sort = 'puntaje_int1'
        elif crit.startswith("Intervalo 2"):
            clave_sort = 'puntaje_int2'
        elif crit.startswith("Intervalo 3"):
            clave_sort = 'puntaje_int3'
        else:
            clave_sort = 'puntaje_global'

        asc = self.var_orden_asc.get()
        items_ord = sorted(
            items,
            key=lambda x: (np.isnan(x[clave_sort]), x[clave_sort] if asc else -x[clave_sort])
        )

        omitir_red = getattr(self, 'var_omitir_redundantes', None) and self.var_omitir_redundantes.get()
        if omitir_red:
            items_ord = filtrar_combinaciones_redundantes(
                items_ord, clave_sort=clave_sort, lista_completa=items
            )

        for it in self.tree_ranking.get_children():
            self.tree_ranking.delete(it)

        self._items_ranking_actual = items_ord

        for i, it in enumerate(items_ord, 1):
            rol_txt = it.get('rol', "Muestra")
            ast = "*" if it.get('orden_indistinto') else ""
            rank_txt = f"{i}{ast}"
            s_int1 = f"{it['puntaje_int1']:.4f}" if not np.isnan(it['puntaje_int1']) else "n/a"
            s_int2 = f"{it['puntaje_int2']:.4f}" if not np.isnan(it['puntaje_int2']) else "n/a"
            s_int3 = f"{it['puntaje_int3']:.4f}" if not np.isnan(it['puntaje_int3']) else "n/a"
            s_glob = f"{it['puntaje_global']:.4f}" if not np.isnan(it['puntaje_global']) else "n/a"

            p3_disp = self._obtener_alias(it.get('p3', ''))
            if it.get('es_mc'):
                p4_disp = self._formatear_patrones_alias(it)
            else:
                p4_disp = self._obtener_alias(it.get('p4', '')) if it.get('p4') and it['p4'] != "—" else "—"

            self.tree_ranking.insert("", "end", iid=str(i - 1), values=(
                rank_txt, rol_txt, p3_disp, p4_disp, it['metodo'], s_int1, s_int2, s_int3, s_glob
            ))

        if items_ord:
            self.tree_ranking.selection_set("0")
            self._mostrar_grafico_muestra(items_ord[0], muestra)
        else:
            self.visor_graf_ranking.limpiar("Sin combinaciones para el filtro seleccionado.")

    def _al_cambiar_criterio_ranking(self, _event=None):
        self._al_cambiar_muestra_ranking()

    def _al_cambiar_limitar_y_tab2(self):
        sel = self.tree_ranking.selection()
        if sel and hasattr(self, '_items_ranking_actual'):
            idx = int(sel[0])
            item = self._items_ranking_actual[idx]
            muestra = self.var_muestra_ranking.get()
            self._mostrar_grafico_muestra(item, muestra)

    def _al_seleccionar_fila_ranking(self, _event=None):
        sel = self.tree_ranking.selection()
        if not sel or not hasattr(self, '_items_ranking_actual'):
            return
        idx = int(sel[0])
        item = self._items_ranking_actual[idx]
        muestra = self.var_muestra_ranking.get()
        self._mostrar_grafico_muestra(item, muestra)

    def _mostrar_grafico_muestra(self, item, muestra_nom):
        frecs_r = item.get('frecs_r', self._datos_resultados['frecs_r'])
        er_med = item['er_medido']
        er_teo = item['er_teorico']
        log_x = self._datos_resultados.get('log_x', True)
        etiq_teo = item.get('etiqueta_teorico', 'Teórico')
        rol_txt = f" [{item.get('rol')}]" if item.get('rol') else ""
        ast_txt = " (* orden indistinto)" if item.get('orden_indistinto') else ""
        if item.get('es_mc'):
            pat_s = item.get('patrones_str') or f"{item.get('p3', '')} + {item.get('p4', '')}"
            titulo = f"{muestra_nom}{rol_txt} — Calibración MC: {pat_s}{ast_txt}"
        else:
            titulo = f"{muestra_nom}{rol_txt} — Calibración P3: {item['p3']} | P4: {item['p4'] or '—'}{ast_txt}"

        # Usamos exactamente `graficar_comparacion` de `analisis_permitividad.py`
        fig = graficar_comparacion(
            frecs_r,
            {f"Medido ({item['metodo']})": er_med},
            er_teo,
            titulo=titulo,
            archivo_salida=None,
            log_x=log_x,
            etiqueta_teorico=etiq_teo,
            limitar_eje_y=self.var_limitar_y_tab2.get(),
        )
        # Marcar los límites de los intervalos de evaluación
        for _, fb in self._datos_resultados['intervalos'][:-1]:
            for ax in fig.axes:
                ax.axvline(fb / 1e9, color='#888888', linestyle=':', lw=1.1)

        self.visor_graf_ranking.mostrar(fig)

    # -----------------------------------------------------------------------
    # Métodos de la Pestaña 3: Resultados por Combinación
    # -----------------------------------------------------------------------
    def _refrescar_lista_combinaciones(self, mantener_seleccion=True):
        if not self._datos_resultados:
            self.combo_combinacion.configure(values=[])
            self.combo_combinacion_b.configure(values=[])
            self.var_combo_seleccionada.set("")
            self.var_combo_b_seleccionada.set("")
            self.btn_pdf_combo.configure(state="disabled")
            return

        res_combos = self._datos_resultados.get('resultados_por_combinacion', {})
        if not res_combos:
            self.combo_combinacion.configure(values=[])
            self.combo_combinacion_b.configure(values=[])
            self.var_combo_seleccionada.set("")
            self.var_combo_b_seleccionada.set("")
            self.btn_pdf_combo.configure(state="disabled")
            return

        orden = self.var_orden_combos.get()
        items_combo = list(res_combos.items())

        if orden == "Menor error (mejor primero)":
            def _clave_score_asc(it):
                sc = it[1].get('puntaje_promedio_total')
                return (sc is None or np.isnan(sc), sc if sc is not None else float('inf'))
            items_combo.sort(key=_clave_score_asc)
        elif orden == "Mayor error (peor primero)":
            def _clave_score_desc(it):
                sc = it[1].get('puntaje_promedio_total')
                return (sc is None or np.isnan(sc), -sc if sc is not None else -float('inf'))
            items_combo.sort(key=_clave_score_desc)

        self._mapa_display_a_combo_id.clear()
        self._todas_opciones_combos = []

        for idx, (cid, c) in enumerate(items_combo, 1):
            sc = c.get('puntaje_promedio_total')
            sc_str = f"{sc:.4f}" if (sc is not None and not np.isnan(sc)) else "n/a"
            met = c.get('metodo', '')
            pat_alias = self._formatear_patrones_alias(c)
            disp = f"#{idx} [Score: {sc_str}] {met} — {pat_alias}"

            if disp in self._mapa_display_a_combo_id:
                disp = f"{disp} ({cid})"

            self._mapa_display_a_combo_id[disp] = cid
            self._todas_opciones_combos.append({
                'disp': disp,
                'cid': cid,
                'datos': c,
                'metodo': met,
                'patrones_str': c.get('patrones_str', ''),
                'p3': c.get('p3', ''),
                'p4': c.get('p4', ''),
                'pat_alias': pat_alias,
            })

        self._filtrar_lista_combinaciones(mantener_seleccion=mantener_seleccion)

    def _filtrar_lista_combinaciones(self, mantener_seleccion=True):
        if not hasattr(self, '_todas_opciones_combos') or not self._todas_opciones_combos:
            return

        palabra = self.var_filtro_palabra_combo.get().strip().lower()
        liq_filtro = self.var_filtro_liquido_combo.get()
        liq_nom = liq_filtro.split(" (")[0].strip() if (liq_filtro and liq_filtro != "Todos") else None

        filtradas = []
        for item in self._todas_opciones_combos:
            if palabra:
                texto_busq = f"{item['disp']} {item['cid']} {item['metodo']} {item['patrones_str']} {item['p3']} {item['p4']} {item['pat_alias']}".lower()
                if palabra not in texto_busq:
                    continue

            if liq_nom:
                p3_nom = item['p3']
                p4_nom = item['p4']
                pat_str = item['patrones_str']
                c_datos = item['datos']
                en_combo = (p3_nom == liq_nom) or (p4_nom == liq_nom) or (liq_nom in pat_str) or (liq_nom in c_datos.get('patrones', []))
                if not en_combo:
                    continue

            filtradas.append(item['disp'])

        self.combo_combinacion.configure(values=filtradas)
        self.combo_combinacion_b.configure(values=filtradas)
        self.btn_pdf_combo.configure(state="normal" if filtradas else "disabled")

        combo_id_previo_a = getattr(self, '_combo_id_activo', None)
        target_a = None
        if mantener_seleccion and combo_id_previo_a:
            for disp in filtradas:
                if self._mapa_display_a_combo_id.get(disp) == combo_id_previo_a:
                    target_a = disp
                    break

        if not target_a and filtradas:
            target_a = filtradas[0]

        if target_a:
            self.var_combo_seleccionada.set(target_a)
            self._combo_id_activo = self._mapa_display_a_combo_id[target_a]
            self._al_cambiar_combinacion()
        else:
            self.var_combo_seleccionada.set("")
            self._combo_id_activo = None
            self.lbl_score_combo.configure(text="Sin combinaciones coincidentes")
            self.visor_curvas_combo.limpiar("Sin combinaciones para el filtro actual.")

        if self.var_comparar_activo.get():
            combo_id_previo_b = getattr(self, '_combo_id_b_activo', None)
            target_b = None
            if mantener_seleccion and combo_id_previo_b:
                for disp in filtradas:
                    if self._mapa_display_a_combo_id.get(disp) == combo_id_previo_b:
                        target_b = disp
                        break
            if not target_b and filtradas:
                target_b = filtradas[1] if len(filtradas) > 1 else filtradas[0]

            if target_b:
                self.var_combo_b_seleccionada.set(target_b)
                self._combo_id_b_activo = self._mapa_display_a_combo_id[target_b]
                self._al_cambiar_combinacion_b()
            else:
                self.var_combo_b_seleccionada.set("")
                self._combo_id_b_activo = None
                self.lbl_score_combo_b.configure(text="")

    def _al_cambiar_orden_combos(self, _event=None):
        self._refrescar_lista_combinaciones(mantener_seleccion=True)

    def _al_toggle_comparar(self):
        if self.var_comparar_activo.get():
            self.combo_combinacion_b.configure(state="readonly")
            self.btn_pdf_combo.configure(text="📄 Imprimir Comparativa (A vs B)...")
            vals = list(self.combo_combinacion_b['values'])
            if not self.var_combo_b_seleccionada.get() and vals:
                sel_b = vals[1] if len(vals) > 1 else vals[0]
                self.var_combo_b_seleccionada.set(sel_b)
                self._combo_id_b_activo = self._mapa_display_a_combo_id.get(sel_b, sel_b)
            self._al_cambiar_combinacion_b()
        else:
            self.combo_combinacion_b.configure(state="disabled")
            self.btn_pdf_combo.configure(text="📄 Imprimir Informe de Combinación...")
            self.lbl_score_combo_b.configure(text="")
            self._al_refrescar_grafico_combo_actual()

    def _al_cambiar_combinacion_b(self, _event=None):
        disp_b = self.var_combo_b_seleccionada.get()
        self._combo_id_b_activo = self._mapa_display_a_combo_id.get(disp_b, disp_b)
        datos_b = self._datos_resultados.get('resultados_por_combinacion', {}).get(self._combo_id_b_activo, {})
        sc = datos_b.get('puntaje_promedio_total')
        sc_str = f"{sc:.4f}" if (sc is not None and not np.isnan(sc)) else "n/a"
        met = datos_b.get('metodo', '')
        self.lbl_score_combo_b.configure(text=f"Combo B: [Score: {sc_str}] {met}")
        self._al_refrescar_grafico_combo_actual()

    def _al_cambiar_limitar_y_tab3(self):
        self._al_refrescar_grafico_combo_actual()

    def _al_refrescar_grafico_combo_actual(self):
        sel = self.tree_combo_muestras.selection()
        if sel and hasattr(self, '_muestras_combo_actual'):
            idx = int(sel[0])
            m_dat = list(self._muestras_combo_actual.values())[idx]
            combo_id = getattr(self, '_combo_id_activo', None)
            datos_combo = self._datos_resultados.get('resultados_por_combinacion', {}).get(combo_id, {})
            self._mostrar_grafico_curvas_combo(m_dat, datos_combo)

    def _al_cambiar_combinacion(self, _event=None):
        if not self._datos_resultados:
            return
        disp_sel = self.var_combo_seleccionada.get()
        combo_id = self._mapa_display_a_combo_id.get(disp_sel, disp_sel)
        self._combo_id_activo = combo_id

        datos_combo = self._datos_resultados.get('resultados_por_combinacion', {}).get(combo_id)
        if not datos_combo:
            return

        score_tot = datos_combo.get('puntaje_promedio_total')
        score_pat = datos_combo.get('puntaje_promedio_patrones')
        score_res = datos_combo.get('puntaje_promedio_restantes')
        partes = []
        if score_tot is not None and not np.isnan(score_tot):
            partes.append(f"Global: {score_tot:.4f}")
        if score_pat is not None and not np.isnan(score_pat):
            partes.append(f"Patrones (ajuste): {score_pat:.4f}")
        if score_res is not None and not np.isnan(score_res):
            partes.append(f"Muestras (validación): {score_res:.4f}")
        self.lbl_score_combo.configure(text="Combo A:  " + ("   |   ".join(partes) if partes else "n/a"))

        # 1. Graficar Gn(f) usando `graficar_Gn` de `analisis_permitividad.py`
        frecs_r = datos_combo.get('frecs_r', self._datos_resultados['frecs_r'])
        Gn_orig = datos_combo.get('Gn_original')
        Gn_acond = datos_combo.get('Gn_acondicionado')
        info_ac = datos_combo.get('info_acond')
        marcas_inicio = datos_combo.get('marcas_inicio_gn')
        log_x = self._datos_resultados.get('log_x', True)

        if Gn_orig is not None:
            fig_gn = graficar_Gn(
                frecs_r,
                Gn_orig,
                archivo_salida=None,
                log_x=log_x,
                marcas_inicio=marcas_inicio,
                Gn_acondicionado=Gn_acond,
                info_acond=info_ac,
            )
            self.visor_gn.mostrar(fig_gn)

            if info_ac and info_ac.get('disparo_detectado'):
                f1_ap = info_ac.get('f1_aplicada')
                f2_ap = info_ac.get('f2_aplicada')
                fc_ap = info_ac.get('f_corte_aplicada')
                if f2_ap is not None:
                    f1_ghz = (f1_ap or 0.0) / 1e9
                    f2_ghz = f2_ap / 1e9
                    self.var_info_acond_gn.set(
                        f"Frecuencias de transición Gn: f1 = {f1_ghz:.3f} GHz  →  f2 = {f2_ghz:.3f} GHz"
                    )
                elif fc_ap is not None:
                    self.var_info_acond_gn.set(f"Frecuencia de corte Gn: {fc_ap/1e9:.3f} GHz")
                else:
                    self.var_info_acond_gn.set("")
            elif info_ac is not None:
                self.var_info_acond_gn.set("Acondicionamiento de Gn: no se detectó disparo abrupto.")
            else:
                self.var_info_acond_gn.set("")
        else:
            self.visor_gn.limpiar("Método simplificado: Gn(f) no se utiliza.")
            self.var_info_acond_gn.set("")

        # 2. Llenar tabla de muestras bajo esta combinación
        for it in self.tree_combo_muestras.get_children():
            self.tree_combo_muestras.delete(it)

        muestras_dict = datos_combo.get('muestras', {})
        self._muestras_combo_actual = muestras_dict

        for i, (m_nom, m_dat) in enumerate(muestras_dict.items()):
            rol_txt = m_dat.get('rol', "Muestra")
            s_i1 = f"{m_dat['puntaje_int1']:.4f}" if not np.isnan(m_dat['puntaje_int1']) else "n/a"
            s_i2 = f"{m_dat['puntaje_int2']:.4f}" if not np.isnan(m_dat['puntaje_int2']) else "n/a"
            s_i3 = f"{m_dat['puntaje_int3']:.4f}" if not np.isnan(m_dat['puntaje_int3']) else "n/a"
            s_gl = f"{m_dat['puntaje_global']:.4f}" if not np.isnan(m_dat['puntaje_global']) else "n/a"

            self.tree_combo_muestras.insert("", "end", iid=str(i), values=(
                m_nom, rol_txt, s_i1, s_i2, s_i3, s_gl
            ))

        if muestras_dict:
            self.tree_combo_muestras.selection_set("0")
            primera_m = list(muestras_dict.values())[0]
            self._mostrar_grafico_curvas_combo(primera_m, datos_combo)
        else:
            self.visor_curvas_combo.limpiar("No hay muestras evaluadas para esta combinación.")

    def _al_seleccionar_muestra_en_combo(self, _event=None):
        sel = self.tree_combo_muestras.selection()
        if not sel or not hasattr(self, '_muestras_combo_actual'):
            return
        idx = int(sel[0])
        m_dat = list(self._muestras_combo_actual.values())[idx]
        combo_id = getattr(self, '_combo_id_activo', None) or self.var_combo_seleccionada.get()
        combo_id = self._mapa_display_a_combo_id.get(combo_id, combo_id)
        datos_combo = self._datos_resultados.get('resultados_por_combinacion', {}).get(combo_id, {})
        self._mostrar_grafico_curvas_combo(m_dat, datos_combo)

    def _generar_informe_combinacion_pdf(self):
        if not self._datos_resultados or not self._ultima_config:
            messagebox.showwarning(
                "Sin resultados",
                "Ejecutá primero el análisis combinatorio antes de generar el informe.",
                parent=self
            )
            return

        combo_id = getattr(self, '_combo_id_activo', None)
        if not combo_id:
            disp_sel = self.var_combo_seleccionada.get()
            combo_id = self._mapa_display_a_combo_id.get(disp_sel, disp_sel)

        if not combo_id or combo_id not in self._datos_resultados.get('resultados_por_combinacion', {}):
            messagebox.showwarning(
                "Combinación no válida",
                "Seleccioná una combinación válida para generar su informe.",
                parent=self
            )
            return

        comparando = bool(self.var_comparar_activo.get())
        combo_b_id = None
        if comparando:
            combo_b_id = getattr(self, '_combo_id_b_activo', None)
            if not combo_b_id:
                disp_b = self.var_combo_b_seleccionada.get()
                combo_b_id = self._mapa_display_a_combo_id.get(disp_b, disp_b)
            if not combo_b_id or combo_b_id not in self._datos_resultados.get('resultados_por_combinacion', {}):
                messagebox.showwarning(
                    "Combinación B no válida",
                    "Seleccioná una combinación B válida para comparar.",
                    parent=self
                )
                return
            nombre_sufijo = f"comp_{_slug(combo_id)[:30].rstrip('_')}_{_slug(combo_b_id)[:30].rstrip('_')}"
        else:
            nombre_sufijo = _slug(combo_id)[:50].rstrip('_')

        ruta_pdf, carpeta_run = self._resolver_carpeta_y_archivo("combo", nombre_sufijo)
        if not ruta_pdf:
            return

        try:
            texto_paso = "Generando informe PDF comparativo..." if comparando else "Generando informe PDF de la combinación..."
            self.var_paso_actual.set(texto_paso)
            self.update_idletasks()
            exportar = carpeta_run if self.var_exportar_imagenes_csv.get() else None

            generar_reporte_combinacion_pdf(
                ruta_pdf,
                self._ultima_config,
                self._datos_resultados,
                combo_id=combo_id,
                combo_id_comparar=combo_b_id,
                carpeta_exportar=exportar,
                limitar_eje_y=self.var_limitar_y_tab3.get(),
            )
            self._ultima_carpeta_salida = carpeta_run
            self.var_paso_actual.set(f"Informe generado en: {os.path.basename(ruta_pdf)}")

            if comparando and combo_b_id:
                msg = f"Se generó el informe PDF comparativo entre:\n'{combo_id}'\ny\n'{combo_b_id}':\n\n{ruta_pdf}"
            else:
                msg = f"Se generó el informe PDF para la combinación:\n'{combo_id}'\n\n{ruta_pdf}"

            if exportar:
                msg += f"\n\nSe exportaron las figuras (.png) y tablas (.csv) en:\n{os.path.abspath(carpeta_run)}"
            msg += "\n\n¿Querés abrir el PDF ahora?"

            if messagebox.askyesno("Informe PDF generado", msg, parent=self):
                try:
                    os.startfile(ruta_pdf)
                except Exception:
                    pass
        except Exception as exc:
            self.var_paso_actual.set("Error al generar el informe PDF de la combinación.")
            messagebox.showerror(
                "Error al generar PDF",
                f"No se pudo generar el informe PDF:\n{exc}\n\n{traceback.format_exc()}",
                parent=self
            )

    def _mostrar_grafico_curvas_combo(self, m_dat, datos_combo=None):
        frecs_r = m_dat.get('frecs_r', self._datos_resultados['frecs_r'])
        er_med = m_dat['er_medido']
        er_teo = m_dat['er_teorico']
        m_nom = m_dat['muestra']
        log_x = self._datos_resultados.get('log_x', True)
        etiq_teo = m_dat.get('etiqueta_teorico', 'Teórico')
        comparando = self.var_comparar_activo.get() and bool(getattr(self, '_combo_id_b_activo', None))

        curvas = {}
        if comparando:
            cid_a = getattr(self, '_combo_id_activo', None)
            cid_b = getattr(self, '_combo_id_b_activo', None)
            combos = self._datos_resultados.get('resultados_por_combinacion', {})
            datos_a = datos_combo or combos.get(cid_a, {})
            datos_b = combos.get(cid_b, {})
            m_dat_a = datos_a.get('muestras', {}).get(m_nom, m_dat)
            m_dat_b = datos_b.get('muestras', {}).get(m_nom)

            pat_a_alias = self._formatear_patrones_alias(datos_a)
            pat_b_alias = self._formatear_patrones_alias(datos_b)
            met_a = datos_a.get('metodo', 'Medido')
            met_b = datos_b.get('metodo', 'Medido')

            curvas[f"A ({met_a}): {pat_a_alias}"] = m_dat_a.get('er_medido')
            if m_dat_b and m_dat_b.get('er_medido') is not None:
                curvas[f"B ({met_b}): {pat_b_alias}"] = m_dat_b.get('er_medido')
            titulo = f"{m_nom} — Comparación Medido A vs Medido B vs Teórico"
        else:
            metodo_str = datos_combo.get('metodo', 'Medido') if datos_combo else 'Medido'
            rol_txt = f" [{m_dat.get('rol')}]" if m_dat.get('rol') else ""
            curvas[f"Medido ({metodo_str})"] = er_med
            titulo = f"{m_nom}{rol_txt}: medido vs. teórico"

        fig = graficar_comparacion(
            frecs_r,
            curvas,
            er_teo,
            titulo=titulo,
            archivo_salida=None,
            log_x=log_x,
            etiqueta_teorico=etiq_teo,
            limitar_eje_y=self.var_limitar_y_tab3.get(),
        )
        for _, fb in self._datos_resultados['intervalos'][:-1]:
            for ax in fig.axes:
                ax.axvline(fb / 1e9, color='#888888', linestyle=':', lw=1.1)

        self.visor_curvas_combo.mostrar(fig)

    # -----------------------------------------------------------------------
    # Guardar / Cargar Configuración JSON
    # -----------------------------------------------------------------------
    def _guardar_config(self):
        ruta = filedialog.asksaveasfilename(
            title="Guardar Configuración de Combinatoria",
            defaultextension=".json", parent=self,
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return
        cfg = {
            'carpeta_datos': self.var_carpeta_datos.get().strip(),
            'archivo_corto': self.var_archivo_corto.get().strip(),
            'archivo_aire': self.var_archivo_aire.get().strip(),
            'f_min_ghz': float(self.var_f_min.get()),
            'f_max_ghz': float(self.var_f_max.get()),
            'f_err_int1_ghz': float(self.var_f_int1.get()),
            'f_err_int2_ghz': float(self.var_f_int2.get()),
            'escala_log_frecuencia': bool(self.var_escala_log.get()),
            'estrategia_gn': self.var_estrategia_gn.get(),
            'metodos_calculo': {
                'simplificado': bool(self.var_metodo_simpl.get()),
                'completo_sin_acond': bool(self.var_metodo_comp_sin.get()),
                'completo_con_acond': bool(self.var_metodo_comp_con.get()),
                'mc_sin_acond': bool(self.var_metodo_mc_sin.get()),
                'mc_con_acond': bool(self.var_metodo_mc_con.get()),
            },
            'mc_k_liquidos': self._obtener_mc_k_config(),
            'acondicionar_gn': {
                'tipo': _TIPOS_ACOND_MAP.get(self.var_acond_tipo.get(), 'transicion_polinomica_cero'),
                'modo': self.var_acond_modo.get(),
                'umbral_pct': float(self.var_acond_umbral.get()),
                'f1_ghz': float(self.var_acond_f1.get()),
                'f2_ghz': float(self.var_acond_f2.get()),
            },
            'carpeta_salida': self.var_carpeta_salida.get().strip(),
            'guardar_subcarpeta_fechayhora': bool(self.var_guardar_fechayhora.get()),
            'exportar_imagenes_csv': bool(self.var_exportar_imagenes_csv.get()),
            'candidatos': copy.deepcopy(self._candidatos),
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Configuración guardada", f"Se guardó la configuración en:\n{ruta}", parent=self)

    def _cargar_config(self):
        ruta = filedialog.askopenfilename(
            title="Cargar Configuración de Combinatoria",
            defaultextension=".json", parent=self,
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.var_carpeta_datos.set(cfg.get('carpeta_datos', '.'))
            self.var_archivo_corto.set(cfg.get('archivo_corto', 'corto.s1p'))
            self.var_archivo_aire.set(cfg.get('archivo_aire', 'aire.s1p'))
            self.var_f_min.set(float(cfg.get('f_min_ghz', 0.5)))
            self.var_f_max.set(float(cfg.get('f_max_ghz', 6.0)))
            self.var_f_int1.set(float(cfg.get('f_err_int1_ghz', 2.0)))
            self.var_f_int2.set(float(cfg.get('f_err_int2_ghz', 4.0)))
            self.var_escala_log.set(bool(cfg.get('escala_log_frecuencia', True)))
            self.var_estrategia_gn.set(cfg.get('estrategia_gn', 'minimo_gn'))

            m = cfg.get('metodos_calculo', {})
            self.var_metodo_simpl.set(bool(m.get('simplificado', False)))
            self.var_metodo_comp_sin.set(bool(m.get('completo_sin_acond', True)))
            self.var_metodo_comp_con.set(bool(m.get('completo_con_acond', False)))
            self.var_metodo_mc_sin.set(bool(m.get('mc_sin_acond', False)))
            self.var_metodo_mc_con.set(bool(m.get('mc_con_acond', False)))

            ac = cfg.get('acondicionar_gn', {})
            tipo_key = ac.get('tipo', 'transicion_polinomica_cero')
            self.var_acond_tipo.set(_TIPOS_ACOND_INV.get(tipo_key, "Transicion cuadratica (f^2)"))
            self.var_acond_modo.set(ac.get('modo', 'automatico'))
            self.var_acond_umbral.set(float(ac.get('umbral_pct', 30.0)))
            self.var_acond_f1.set(float(ac.get('f1_ghz', 0.0)))
            self.var_acond_f2.set(float(ac.get('f2_ghz', 0.5)))

            self.var_carpeta_salida.set(cfg.get('carpeta_salida', r"..\salidas"))
            self.var_guardar_fechayhora.set(bool(cfg.get('guardar_subcarpeta_fechayhora', True)))
            self.var_exportar_imagenes_csv.set(bool(cfg.get('exportar_imagenes_csv', True)))

            self._candidatos = copy.deepcopy(cfg.get('candidatos', []))
            for c in self._candidatos:
                if 'alias' not in c or not c['alias']:
                    c['alias'] = generar_alias_defecto(c.get('nombre', ''))
            self._refrescar_tabla_candidatos()
            self._setear_mc_k_desde_config(cfg.get('mc_k_liquidos', 'todos'))
            self._al_cambiar_acond()
            self._al_cambiar_acond_modo()
            messagebox.showinfo("Configuración cargada", f"Se cargó la configuración de:\n{ruta}", parent=self)
        except Exception as exc:
            messagebox.showerror("Error al cargar", f"No se pudo cargar la configuración:\n{exc}", parent=self)


if __name__ == "__main__":
    app = AppCombinatoria()
    app.mainloop()
