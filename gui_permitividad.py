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
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import gui_funciones as gf
import analisis_permitividad as ap
import Patrones as P

# matplotlib ya queda forzado al backend 'Agg' por gui_funciones.py (se
# importa arriba, y ese modulo lo hace ANTES de importar analisis_
# permitividad) -- eso es lo que usa el pipeline de analisis en el hilo de
# fondo para guardar los .png a disco. Para la vista previa EN VIVO dentro
# de la ventana (con zoom, pan, "home" y lectura de coordenadas bajo el
# cursor -- la misma interaccion que da el backend Qt de matplotlib, via
# la misma clase base NavigationToolbar2) se usa por separado
# FigureCanvasTkAgg/NavigationToolbar2Tk (ver clase VisorFigura mas
# abajo), que embebe una Figure directamente en un widget de Tkinter sin
# pasar por el registro global de pyplot -- por eso es seguro usarlo
# desde el hilo principal (Tkinter) mientras el hilo de fondo genera sus
# propias figuras con pyplot/Agg por su lado, sin que se pisen entre si.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from Touchstone import leer_s1p, graficar_s11_mag_fase, graficar_smith
from funciones import filtrar_s11_savgol, filtrar_s11_pasabajos, filtrar_s11_mediana, filtrar_s11_splines, aplicar_filtro_s11


TITULO_APP = "Analizador de Permitividad - Sonda Coaxial (NPL MAT 23)"

# Etiquetas legibles para los patrones de calibracion, para la pestaña de
# "Vista previa S11" (analogo a _ETIQUETAS_CALIBRACION_FIJAS de
# gui_funciones.py, pero esa es privada y esta la necesitamos aca). Solo
# corto/aire tienen nombre fijo -- patron3/patron4 muestran el liquido
# elegido en la pestaña de Calibracion (ver _refrescar_lista_preview_s11).
_ETIQUETAS_CALIBRACION_PREVIEW = {
    'corto': "Cortocircuito",
    'aire': "Aire",
}
_NOMBRES_PATRON_CALIBRACION = {'patron3': "Patron 3", 'patron4': "Patron 4"}

_TIPOS_ACOND_MAP = {
    "Transicion orden 5 (f^5)": "transicion_polinomica_cero",
    "Filtro Pasabajos (Butterworth)": "pasabajos",
    "Filtro Savitzky-Golay": "savitzky_golay",
    "Filtro Mediana móvil": "mediana_movil",
    "Splines de suavizado": "splines",
    "Transicion orden 5 + Pasabajos": "transicion_pasabajos",
    "Transicion orden 5 + Savitzky-Golay": "transicion_savgol",
    "Transicion cuadratica (f^2)": "transicion_polinomica_cero",  # retrocompatibilidad
    "Transicion polinomica (cubica)": "transicion_polinomica_cero",
    "Cero clamping (corte duro)": "cero_clamping",
    "Tapering (ventana suave)": "tapering",
}
_TIPOS_ACOND_INV = {
    "transicion_polinomica_cero": "Transicion orden 5 (f^5)",
    "pasabajos": "Filtro Pasabajos (Butterworth)",
    "savitzky_golay": "Filtro Savitzky-Golay",
    "mediana_movil": "Filtro Mediana móvil",
    "splines": "Splines de suavizado",
    "transicion_pasabajos": "Transicion orden 5 + Pasabajos",
    "transicion_savgol": "Transicion orden 5 + Savitzky-Golay",
    "cero_clamping": "Cero clamping (corte duro)",
    "tapering": "Tapering (ventana suave)",
}


def _generar_preview_s11(ruta_s1p, log_x=True, filtrar_s11=False, tipo_filtro='savitzky_golay',
                         sg_window=15, sg_polyorder=3, pb_orden=2, pb_fcorte=0.10,
                         med_window=15, spline_s=0.10, modo_rango='todo', f1_ghz=0.5, f2_ghz=6.0,
                         filtrar_sg=None):
    """
    Lee un archivo .s1p y arma dos Figures de matplotlib EN VIVO (para
    embeber con VisorFigura, sin guardar ningun .png a disco) con el S11
    medido: modulo/fase y diagrama de Smith. Si `filtrar_s11` (o `filtrar_sg`) es True,
    grafica superpuestas la curva original y la curva con el filtro configurado.

    Devuelve (fig_modulo_fase, fig_smith, datos).
    """
    if filtrar_sg is not None:
        filtrar_s11 = bool(filtrar_sg)

    datos = leer_s1p(ruta_s1p)
    nombre = datos.get('Nombre') or os.path.splitext(os.path.basename(ruta_s1p))[0]

    if filtrar_s11:
        s11_orig = datos['Complex']
        frecs = datos['Frec']
        cfg_f = {
            'tipo_filtro': tipo_filtro,
            'pb_orden': pb_orden,
            'pb_fcorte': pb_fcorte,
            'sg_window': sg_window,
            'sg_polyorder': sg_polyorder,
            'med_window': med_window,
            'spline_s': spline_s,
            'modo_rango': modo_rango,
            'f1_ghz': f1_ghz,
            'f2_ghz': f2_ghz,
        }
        s11_filt = aplicar_filtro_s11(frecs, s11_orig, cfg_f)

        nombres_tipo = {
            'pasabajos': "Pasabajos",
            'savitzky_golay': "Savitzky-Golay",
            'mediana_movil': "Mediana móvil",
            'splines': "Splines",
        }
        nom_f = nombres_tipo.get(tipo_filtro, tipo_filtro)
        etiqueta_filt = f"{nombre} (filtrado {nom_f})"
        titulo_mf = f"{nombre}: S11 original vs filtrado {nom_f} (modulo y fase)"
        titulo_sm = f"{nombre}: S11 original vs filtrado {nom_f} (diagrama de Smith)"

        datos_orig = {'Frec': frecs, 'Complex': s11_orig}
        datos_filt = {'Frec': frecs, 'Complex': s11_filt}
        datos_dict = {
            f"{nombre} (original)": datos_orig,
            etiqueta_filt: datos_filt,
        }
        fig_mf = graficar_s11_mag_fase(datos_dict, titulo=titulo_mf, log_x=log_x)
        fig_sm = graficar_smith(datos_dict, titulo=titulo_sm)
        return fig_mf, fig_sm, datos
    else:
        datos_dict = {nombre: datos}
        fig_mf = graficar_s11_mag_fase(
            datos_dict, titulo=f"{nombre}: S11 medido (modulo y fase)", log_x=log_x)
        fig_sm = graficar_smith(datos_dict, titulo=f"{nombre}: S11 medido (diagrama de Smith)")
        return fig_mf, fig_sm, datos

def _generar_preview_gn(frecs, Gn, Gn_acondicionado=None, log_x=True, info_acond=None):
    """
    Genera dos Figures de matplotlib (parte real y parte imaginaria) para la vista previa
    interactiva de Gn(f) en la pestaña 'Vista previa S11'.
    """
    from matplotlib.figure import Figure

    fig_re = Figure(figsize=(6, 5))
    ax_re = fig_re.subplots(1, 1)

    fig_im = Figure(figsize=(6, 5))
    ax_im = fig_im.subplots(1, 1)

    re_Gn = np.real(Gn)
    im_Gn = np.imag(Gn)
    tiene_acond = Gn_acondicionado is not None

    if tiene_acond:
        re_Gn_acond = np.real(Gn_acondicionado)
        im_Gn_acond = np.imag(Gn_acondicionado)

        ax_re.plot(frecs / 1e9, re_Gn, linewidth=1.4, color='#777777', linestyle='--',
                   label="Re(Gn) Original", zorder=3)
        ax_re.plot(frecs / 1e9, re_Gn_acond, linewidth=1.8, color='tab:purple',
                   label="Re(Gn) Acondicionado", zorder=4)

        ax_im.plot(frecs / 1e9, im_Gn, linewidth=1.4, color='#777777', linestyle='--',
                   label="Im(Gn) Original", zorder=3)
        ax_im.plot(frecs / 1e9, im_Gn_acond, linewidth=1.8, color='tab:purple',
                   label="Im(Gn) Acondicionado", zorder=4)
    else:
        ax_re.plot(frecs / 1e9, re_Gn, linewidth=1.6, color='tab:purple',
                   label="Re(Gn)", zorder=3)
        ax_im.plot(frecs / 1e9, im_Gn, linewidth=1.6, color='tab:purple',
                   label="Im(Gn)", zorder=3)

    if info_acond and info_acond.get('disparo_detectado'):
        f_disp = info_acond.get('f2_aplicada') or info_acond.get('f_corte_aplicada')
        if f_disp is not None and f_disp > 0:
            f_disp_ghz = f_disp / 1e9
            lbl_trans = f"Empalme/Corte: {f_disp_ghz:.3f} GHz"
            ax_re.axvline(f_disp_ghz, color='#8856a7', linestyle=':', linewidth=1.4,
                          label=lbl_trans, zorder=2)
            ax_im.axvline(f_disp_ghz, color='#8856a7', linestyle=':', linewidth=1.4,
                          label=lbl_trans, zorder=2)

    if log_x:
        ax_re.set_xscale('log')
        ax_im.set_xscale('log')

    ax_re.set_xlabel("Frecuencia (GHz)")
    ax_re.set_ylabel("Re(Gn)")
    ax_re.set_title("Conductancia normalizada Re(Gn)")
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_re.legend(loc='best', framealpha=0.9, fontsize=8)

    ax_im.set_xlabel("Frecuencia (GHz)")
    ax_im.set_ylabel("Im(Gn)")
    ax_im.set_title("Conductancia normalizada Im(Gn)")
    ax_im.minorticks_on()
    ax_im.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_im.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_im.legend(loc='best', framealpha=0.9, fontsize=8)

    fig_re.tight_layout()
    fig_im.tight_layout()
    return fig_re, fig_im


# ===========================================================================
# Visor de figuras interactivo: zoom (rectangulo o rueda), pan (arrastrar),
# "home" para volver a la vista original, y lectura de las coordenadas del
# dato bajo el cursor en la esquina inferior derecha -- la misma
# interaccion que ofrece el backend Qt de matplotlib (ambos backends
# heredan de la misma clase base NavigationToolbar2, con el mismo set de
# herramientas: Home/Back/Forward, Pan, Zoom, Configurar subplots,
# Guardar). Reemplaza al mecanismo anterior (imagen PNG estatica via PIL,
# reescalada a mano en cada <Configure> de la ventana): ahora la figura
# es una Figure de matplotlib real embebida con FigureCanvasTkAgg, que ya
# se redimensiona sola con el panel que la contiene.
# ===========================================================================
def _crear_indicador_archivo(parent, var_archivo, var_carpeta=None):
    """
    Crea (sin colocar con grid/pack -- eso lo decide el llamador) y
    devuelve un ttk.Label chico que muestra \u2713 (verde) si el archivo
    referenciado por `var_archivo` existe en disco, o \u2717 (rojo) si no.
    Si `var_carpeta` se pasa, la ruta a chequear es la combinacion de
    ambas (`os.path.join`, que ya maneja bien el caso de que `var_archivo`
    sea una ruta absoluta -- entonces `var_carpeta` se ignora, como
    corresponde); si no se pasa, se asume que `var_archivo` ya tiene la
    ruta completa (p.ej. los campos que se cargan con
    `filedialog.askopenfilename`, que siempre devuelve rutas absolutas).

    El indicador se actualiza solo con cada tecla (via `trace_add` sobre
    el/los StringVar), sin depender de ningun boton de "Verificar
    archivos" -- se usa tanto en la pestaña de Calibracion como en los
    dialogos de Material y de curvas (pestañas 5 y 6).
    """
    label = ttk.Label(parent, text="", width=2, anchor="center")

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


class CartelAdvertencia(ttk.Frame):
    """
    Cartel de advertencia reusable (un recuadro de color con titulo y
    texto) para avisar de las limitaciones de un modelo teorico.

    Existe sobre todo por los modelos de permitividad ESTATICA que trae
    `Patrones.py` -- acetona, ciclohexano y fluido de silicona --, que NO
    tienen ecuacion de Debye ajustada y devuelven una permitividad
    constante en frecuencia. Elegir uno de esos sin darse cuenta llevaria
    a comparar contra una referencia que no significa lo que uno cree
    (sobre todo en er'', que en esos modelos es 0 por no estar modelado),
    asi que la advertencia tiene que verse en la GUI, no solo en el PDF.

    Uso:
        cartel = CartelAdvertencia(parent)
        cartel.grid(...)            # o .pack(...)
        cartel.mostrar_modelo('acetona')   # se llena y se hace visible
        cartel.ocultar()                   # se esconde (no ocupa lugar)

    El propio widget se oculta solo (`grid_remove`/`pack_forget`) cuando
    el modelo no tiene advertencia, asi no deja un hueco vacio en el
    layout.
    """

    # (color del borde/titulo, color de fondo) por nivel, en la misma
    # linea que la paleta del informe PDF (reporte_pdf._COLORES_ADVERTENCIA)
    # para que el cartel en pantalla y la caja del PDF se vean coherentes.
    _COLORES = {
        'info': ("#2c5d8f", "#eaf2fa"),
        'aviso': ("#8a6100", "#fdf4e0"),
        'critico': ("#a32020", "#fdeaea"),
    }

    def __init__(self, master, wraplength=560, **kwargs):
        super().__init__(master, **kwargs)
        self._wraplength = wraplength
        self._visible = False
        self._geometria = None  # 'grid' | 'pack', segun como lo ubique el padre

        self._marco = tk.Frame(self, bd=1, relief="solid")
        self._marco.pack(fill="both", expand=True)

        self._lbl_titulo = tk.Label(
            self._marco, anchor="w", justify="left", font=("", 9, "bold"),
            wraplength=wraplength, padx=8, pady=(0))
        self._lbl_titulo.pack(fill="x", padx=2, pady=(6, 0))

        self._lbl_texto = tk.Label(
            self._marco, anchor="w", justify="left", font=("", 8),
            wraplength=wraplength, padx=8)
        self._lbl_texto.pack(fill="x", padx=2, pady=(3, 7))

    # -- API --------------------------------------------------------
    def mostrar_modelo(self, clave_modelo):
        """Muestra la advertencia del modelo `clave_modelo` (clave de
        Patrones.PATRONES_TEORICOS), o se oculta si ese modelo no tiene
        ninguna. Devuelve True si quedo visible."""
        texto = P.advertencia_patron(clave_modelo) if clave_modelo else None
        if not texto:
            self.ocultar()
            return False
        etiqueta = gf.etiqueta_de_modelo(clave_modelo)
        nivel = P.nivel_advertencia_patron(clave_modelo)
        # El encabezado dice de que se trata la advertencia, que no
        # siempre es una limitacion grave: para los liquidos no polares
        # es simplemente que no hay (ni puede haber) relajacion, y para
        # la acetona es de donde salio el modelo.
        if P.es_modelo_estatico(clave_modelo):
            titulo = f"Sin ecuación de Debye — {etiqueta}"
        elif nivel == 'info':
            titulo = f"Sobre este modelo — {etiqueta}"
        else:
            titulo = f"Atención — {etiqueta}"
        self.mostrar(titulo, texto, nivel)
        return True

    def mostrar(self, titulo, texto, nivel='aviso'):
        """Muestra un cartel con texto libre (para avisos que no salen de
        un modelo en particular, p.ej. reparos de calibracion)."""
        color_borde, color_fondo = self._COLORES.get(nivel, self._COLORES['aviso'])
        simbolo = "ⓘ" if nivel == 'info' else "⚠"
        self._marco.configure(background=color_fondo, highlightbackground=color_borde,
                               highlightcolor=color_borde, highlightthickness=2)
        self._lbl_titulo.configure(text=f"{simbolo}  {titulo}",
                                    background=color_fondo, foreground=color_borde)
        self._lbl_texto.configure(text=str(texto).strip(), background=color_fondo,
                                   foreground="#222222")
        self._revelar()

    def ocultar(self):
        if not self._visible:
            return
        if self._geometria == 'grid':
            self.grid_remove()
        elif self._geometria == 'pack':
            self.pack_forget()
        self._visible = False

    def ajustar_wraplength(self, ancho_px):
        """Reajusta el ancho de wrap (util si el panel que lo contiene
        cambia de tamaño)."""
        ancho = max(int(ancho_px), 200)
        self._wraplength = ancho
        self._lbl_titulo.configure(wraplength=ancho)
        self._lbl_texto.configure(wraplength=ancho)

    # -- interno ----------------------------------------------------
    def grid(self, **kwargs):
        self._geometria = 'grid'
        self._grid_kwargs = kwargs
        super().grid(**kwargs)
        self._visible = True

    def pack(self, **kwargs):
        self._geometria = 'pack'
        self._pack_kwargs = kwargs
        super().pack(**kwargs)
        self._visible = True

    def _revelar(self):
        if self._visible:
            return
        if self._geometria == 'grid':
            super().grid(**getattr(self, '_grid_kwargs', {}))
        elif self._geometria == 'pack':
            super().pack(**getattr(self, '_pack_kwargs', {}))
        self._visible = True


class ScrollableFrame(ttk.Frame):
    """Contenedor con barras de desplazamiento horizontal y vertical integradas."""

    def __init__(self, master, padding=12, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_content = ttk.Frame(self.canvas, padding=padding)

        self.scrollable_content.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<Configure>", self._al_configurar_canvas)
        self.canvas.bind("<Enter>", self._enlazar_rueda)
        self.canvas.bind("<Leave>", self._desenlazar_rueda)

    def _al_configurar_canvas(self, event):
        req_w = self.scrollable_content.winfo_reqwidth()
        if event.width > req_w:
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        else:
            self.canvas.itemconfig(self.canvas_window, width=req_w)

    def _enlazar_rueda(self, _event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _desenlazar_rueda(self, _event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if not self.winfo_exists() or not self.canvas.winfo_ismapped():
            return
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        elif getattr(event, 'delta', 0):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class VisorFigura(ttk.Frame):
    """Panel reusable que embebe una Figure de matplotlib con su barra de
    herramientas interactiva. Uso: `visor.mostrar(fig)` para mostrar una
    Figure nueva (reemplaza lo que hubiera antes), `visor.limpiar(msg)`
    para dejarlo vacio con un mensaje."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._canvas = None
        self._toolbar = None
        self._label_vacio = ttk.Label(self, anchor="center", justify="left",
                                       style="Ayuda.TLabel")
        self._label_vacio.grid(row=0, column=0, sticky="nsew")

    def mostrar(self, fig):
        """Reemplaza el contenido del panel por `fig` (matplotlib
        Figure), con su barra de herramientas interactiva."""
        self._limpiar_widgets()
        self._label_vacio.grid_remove()

        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.draw()
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # pack_toolbar=False para poder ubicarla nosotros con grid (la
        # barra de NavigationToolbar2Tk usa pack() internamente por
        # default, que no se puede mezclar con grid() en el mismo padre
        # sin este parametro).
        self._toolbar = NavigationToolbar2Tk(self._canvas, self, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.grid(row=1, column=0, sticky="ew")

    def limpiar(self, mensaje=""):
        """Vacia el panel y muestra `mensaje` centrado (p.ej. cuando
        todavia no hay nada que mostrar, o fallo la generacion)."""
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
class DialogoCurvaComparar(tk.Toplevel):
    """Ventana modal para agregar o editar una curva de la pestaña
    'Comparar mediciones'. El resultado queda en `self.resultado` (dict
    con 'etiqueta', 'archivo', 'columna') si se confirma con "Guardar", o
    en None si se cancela."""

    def __init__(self, master, curva=None):
        super().__init__(master)
        self.resultado = None
        self._columnas_disponibles = []
        curva = curva or {'etiqueta': '', 'archivo': '', 'columna': ''}

        self.title("Agregar curva" if not curva.get('archivo') else f"Editar: {curva['etiqueta']}")
        self.resizable(False, False)
        self.transient(master)

        frm = ttk.Frame(self, padding=14)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Archivo CSV:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_archivo = tk.StringVar(value=curva['archivo'])
        ttk.Entry(frm, textvariable=self.var_archivo, width=40).grid(
            row=0, column=1, sticky="ew", pady=4)
        _crear_indicador_archivo(frm, self.var_archivo).grid(row=0, column=2, padx=(4, 0))
        ttk.Button(frm, text="Examinar...", command=self._examinar_archivo).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Label(frm, text="Columna a graficar:").grid(row=1, column=0, sticky="w", pady=4)
        self.var_columna = tk.StringVar(value=curva['columna'])
        self.combo_columna = ttk.Combobox(
            frm, textvariable=self.var_columna, state="readonly", width=37)
        self.combo_columna.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        self.combo_columna.bind("<<ComboboxSelected>>", self._al_cambiar_columna)

        ttk.Label(frm, text="Etiqueta:").grid(row=2, column=0, sticky="w", pady=4)
        self.var_etiqueta = tk.StringVar(value=curva['etiqueta'])
        ttk.Entry(frm, textvariable=self.var_etiqueta, width=40).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=4)
        _tooltip(self.combo_columna,
                 "Un mismo CSV puede tener varias columnas (por ejemplo\n"
                 "'teorico', 'Medido (simplificado, 3 patrones)', 'Medido\n"
                 "(completo, 4 patrones)') -- elegi cual mostrar en el grafico.")

        ttk.Separator(frm).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)

        frm_botones = ttk.Frame(frm)
        frm_botones.grid(row=4, column=0, columnspan=3, sticky="e")
        ttk.Button(frm_botones, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(frm_botones, text="Guardar", command=self._guardar).pack(side="right")

        if curva.get('archivo') and os.path.isfile(curva['archivo']):
            self._cargar_columnas(curva['archivo'], seleccionar=curva.get('columna'))

        self.bind("<Return>", lambda _e: self._guardar())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_idletasks()
        self.grab_set()
        self.focus_set()

    def _examinar_archivo(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar CSV de resultados", parent=self,
            filetypes=[("CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if ruta:
            self.var_archivo.set(ruta)
            self._cargar_columnas(ruta)

    def _cargar_columnas(self, ruta, seleccionar=None):
        """Lee el CSV (via ap.leer_csv_resultado) y puebla el combobox de
        columnas disponibles. Si falla (no es un CSV valido de este
        programa), avisa y deja el combobox vacio."""
        try:
            _, columnas = ap.leer_csv_resultado(ruta)
        except Exception as exc:
            messagebox.showwarning(
                "No se pudo leer el CSV",
                f"No se pudieron detectar columnas en este archivo:\n{exc}", parent=self)
            self._columnas_disponibles = []
            self.combo_columna.configure(values=[])
            self.var_columna.set("")
            return

        self._columnas_disponibles = list(columnas.keys())
        self.combo_columna.configure(values=self._columnas_disponibles)
        if seleccionar in self._columnas_disponibles:
            self.var_columna.set(seleccionar)
        elif self._columnas_disponibles:
            # Preferir 'Medido (completo, 4 patrones)' como default si
            # esta presente (el metodo mas preciso); si no, la primera.
            preferida = next(
                (c for c in self._columnas_disponibles if 'completo' in c.lower()),
                self._columnas_disponibles[0])
            self.var_columna.set(preferida)
        self._al_cambiar_columna()

    def _al_cambiar_columna(self, _event=None):
        if not self.var_etiqueta.get().strip() and self.var_archivo.get() and self.var_columna.get():
            base = os.path.splitext(os.path.basename(self.var_archivo.get()))[0]
            self.var_etiqueta.set(f"{base} \u2014 {self.var_columna.get()}")

    def _guardar(self):
        archivo = self.var_archivo.get().strip()
        columna = self.var_columna.get().strip()
        etiqueta = self.var_etiqueta.get().strip()
        if not archivo:
            messagebox.showwarning("Falta el archivo", "Elegi el archivo CSV.", parent=self)
            return
        if not columna:
            messagebox.showwarning("Falta la columna", "Elegi que columna graficar.", parent=self)
            return
        if not etiqueta:
            etiqueta = f"{os.path.splitext(os.path.basename(archivo))[0]} \u2014 {columna}"
        self.resultado = {'etiqueta': etiqueta, 'archivo': archivo, 'columna': columna}
        self.destroy()


class DialogoModeloTeorico(tk.Toplevel):
    """Ventana modal para agregar o editar una curva de la pestaña
    '6. Modelos teoricos'. El resultado queda en `self.resultado` (dict
    con 'etiqueta', 'modelo', 'temperatura', 'f_min_ghz', 'f_max_ghz',
    'n_puntos') si se confirma con "Guardar", o en None si se cancela."""

    def __init__(self, master, curva=None):
        super().__init__(master)
        self.resultado = None
        curva = curva or {
            'etiqueta': '', 'modelo': gf.SIN_MODELO, 'temperatura': 25.0,
            'f_min_ghz': 0.5, 'f_max_ghz': 6.0, 'n_puntos': 200,
        }

        self.title("Agregar modelo teorico" if not curva.get('modelo') else
                   f"Editar: {curva.get('etiqueta') or curva['modelo']}")
        self.resizable(False, False)
        self.transient(master)

        frm = ttk.Frame(self, padding=14)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Modelo teorico:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_modelo_etiqueta = tk.StringVar(
            value=gf.etiqueta_de_modelo(curva.get('modelo')) if curva.get('modelo') else "")
        self.combo_modelo = ttk.Combobox(
            frm, textvariable=self.var_modelo_etiqueta, state="readonly",
            values=[et for _, et in gf.listar_modelos_teoricos_calibracion()], width=33)
        self.combo_modelo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        self.combo_modelo.bind("<<ComboboxSelected>>", self._al_cambiar_algo)

        ttk.Label(frm, text="Temperatura (°C):").grid(row=1, column=0, sticky="w", pady=4)
        self.var_temperatura = tk.DoubleVar(value=float(curva.get('temperatura', 25.0)))
        spin_t = ttk.Spinbox(frm, from_=-20, to=150, increment=0.5,
                              textvariable=self.var_temperatura, width=10)
        spin_t.grid(row=1, column=1, sticky="w", pady=4)
        _tooltip(spin_t, "El modelo teorico usa la temperatura tabulada mas cercana\n"
                          "(pasos de 5°C, NPL MAT 23). No hace falta que coincida\n"
                          "exactamente con un escalon de la tabla.")

        frm_rango = ttk.Frame(frm)
        frm_rango.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(frm_rango, text="Frecuencia:").pack(side="left")
        self.var_f_min = tk.DoubleVar(value=float(curva.get('f_min_ghz', 0.5)))
        ttk.Label(frm_rango, text="  desde").pack(side="left")
        ttk.Spinbox(frm_rango, from_=0.0001, to=1000, increment=0.1,
                    textvariable=self.var_f_min, width=8).pack(side="left", padx=(4, 0))
        self.var_f_max = tk.DoubleVar(value=float(curva.get('f_max_ghz', 6.0)))
        ttk.Label(frm_rango, text="  hasta").pack(side="left")
        ttk.Spinbox(frm_rango, from_=0.0001, to=1000, increment=0.1,
                    textvariable=self.var_f_max, width=8).pack(side="left", padx=(4, 0))
        ttk.Label(frm_rango, text="GHz").pack(side="left", padx=(4, 0))

        ttk.Label(frm, text="Cantidad de puntos:").grid(row=3, column=0, sticky="w", pady=4)
        self.var_n_puntos = tk.IntVar(value=int(curva.get('n_puntos', 200)))
        ttk.Spinbox(frm, from_=2, to=5000, increment=10,
                    textvariable=self.var_n_puntos, width=10).grid(
            row=3, column=1, sticky="w", pady=4)

        ttk.Label(frm, text="Etiqueta:").grid(row=4, column=0, sticky="w", pady=4)
        self.var_etiqueta = tk.StringVar(value=curva.get('etiqueta', ''))
        ttk.Entry(frm, textvariable=self.var_etiqueta, width=40).grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Separator(frm).grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        # Mismo cartel que en el dialogo de Material: avisa si el modelo
        # elegido no tiene ecuacion de relajacion (permitividad constante).
        self.cartel_modelo = CartelAdvertencia(frm, wraplength=430)
        self.cartel_modelo.grid(row=6, column=0, columnspan=3, sticky="ew",
                                 pady=(0, 8))
        self.cartel_modelo.ocultar()

        frm_botones = ttk.Frame(frm)
        frm_botones.grid(row=7, column=0, columnspan=3, sticky="e")
        ttk.Button(frm_botones, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(frm_botones, text="Guardar", command=self._guardar).pack(side="right")

        self._al_cambiar_algo()
        self.bind("<Return>", lambda _e: self._guardar())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_idletasks()
        self.grab_set()
        self.focus_set()

    def _al_cambiar_algo(self, _event=None):
        if not self.var_etiqueta.get().strip() and self.var_modelo_etiqueta.get():
            self.var_etiqueta.set(
                f"{self.var_modelo_etiqueta.get()} a {self.var_temperatura.get():.1f}\u00b0C")
        # Cartel de advertencia del modelo elegido (se oculta solo si el
        # modelo no tiene ninguna).
        self.cartel_modelo.mostrar_modelo(
            gf.clave_de_etiqueta(self.var_modelo_etiqueta.get()))
        self.update_idletasks()

    def _guardar(self):
        modelo = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get())
        if not modelo:
            messagebox.showwarning("Falta el modelo", "Elegi que modelo teorico graficar.", parent=self)
            return
        try:
            f_min = float(self.var_f_min.get())
            f_max = float(self.var_f_max.get())
        except (tk.TclError, ValueError):
            messagebox.showwarning("Rango invalido", "La frecuencia minima/maxima no es un numero valido.",
                                    parent=self)
            return
        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            messagebox.showwarning(
                "Rango invalido",
                "La frecuencia minima tiene que ser mayor que 0 y menor que la maxima.", parent=self)
            return
        try:
            temperatura = float(self.var_temperatura.get())
        except (tk.TclError, ValueError):
            temperatura = 25.0
        try:
            n_puntos = max(int(self.var_n_puntos.get()), 2)
        except (tk.TclError, ValueError):
            n_puntos = 200

        etiqueta = self.var_etiqueta.get().strip() or f"{self.var_modelo_etiqueta.get()} a {temperatura:.1f}\u00b0C"
        self.resultado = {
            'etiqueta': etiqueta, 'modelo': modelo, 'temperatura': temperatura,
            'f_min_ghz': f_min, 'f_max_ghz': f_max, 'n_puntos': n_puntos,
        }
        self.destroy()


class DialogoPatronAdicional(tk.Toplevel):
    """Ventana modal para agregar o editar un patron liquido adicional de
    la calibracion por cuadrados minimos (pestaña 1). El resultado queda
    en `self.resultado` (dict con 'archivo', 'modelo', 'temperatura_c')
    si se confirma con "Guardar", o en None si se cancela."""

    def __init__(self, master, patron=None):
        super().__init__(master)
        self.resultado = None
        patron = patron or {'archivo': '', 'modelo': 'alcohol_etilico', 'temperatura_c': 25.0}

        self.title("Agregar patron adicional" if not patron.get('archivo') else "Editar patron adicional")
        self.resizable(False, False)
        self.transient(master)

        frm = ttk.Frame(self, padding=14)
        frm.grid(sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Archivo .s1p:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_archivo = tk.StringVar(value=patron.get('archivo', ''))
        ttk.Entry(frm, textvariable=self.var_archivo, width=38).grid(
            row=0, column=1, sticky="ew", pady=4)
        _crear_indicador_archivo(
            frm, self.var_archivo, getattr(master, 'var_carpeta_datos', None)).grid(
            row=0, column=2, padx=(4, 0))
        ttk.Button(frm, text="Examinar...", command=self._examinar_archivo).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Label(frm, text="Liquido:").grid(row=1, column=0, sticky="w", pady=4)
        self.var_modelo_etiqueta = tk.StringVar(
            value=gf.etiqueta_de_modelo(patron.get('modelo') or 'alcohol_etilico'))
        modelos = [et for _, et in gf.listar_modelos_teoricos_calibracion()]
        self.combo_modelo = ttk.Combobox(
            frm, textvariable=self.var_modelo_etiqueta, state="readonly",
            values=modelos, width=35)
        self.combo_modelo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=4)
        self.combo_modelo.bind("<<ComboboxSelected>>", self._al_cambiar_modelo)

        ttk.Label(frm, text="Temperatura (°C):").grid(row=2, column=0, sticky="w", pady=4)
        self.var_temperatura = tk.DoubleVar(value=float(patron.get('temperatura_c', 25.0) or 25.0))
        spin_t = ttk.Spinbox(frm, from_=-20, to=150, increment=0.5,
                              textvariable=self.var_temperatura, width=10)
        spin_t.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Separator(frm).grid(row=3, column=0, columnspan=4, sticky="ew", pady=10)

        self.cartel_modelo = CartelAdvertencia(frm, wraplength=430)
        self.cartel_modelo.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        self.cartel_modelo.ocultar()

        frm_botones = ttk.Frame(frm)
        frm_botones.grid(row=5, column=0, columnspan=4, sticky="e")
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
            title="Seleccionar archivo .s1p del patron", parent=self,
            filetypes=[("Touchstone .s1p", "*.s1p"), ("Todos los archivos", "*.*")])
        if ruta:
            self.var_archivo.set(ruta)
            sugerido = gf.liquido_sugerido_por_nombre(ruta)
            if sugerido:
                self.var_modelo_etiqueta.set(gf.etiqueta_de_modelo(sugerido))
                self._al_cambiar_modelo()

    def _al_cambiar_modelo(self, _event=None):
        clave = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get())
        self.cartel_modelo.mostrar_modelo(clave)
        self.update_idletasks()

    def _guardar(self):
        archivo = self.var_archivo.get().strip()
        if not archivo:
            messagebox.showwarning("Falta el archivo", "Elegi el archivo .s1p de este patron.", parent=self)
            return
        modelo = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get())
        if not modelo:
            messagebox.showwarning("Falta el liquido", "Elegi que liquido es este patron.", parent=self)
            return
        try:
            temperatura = float(self.var_temperatura.get())
        except (tk.TclError, ValueError):
            temperatura = 25.0
        self.resultado = {'archivo': archivo, 'modelo': modelo, 'temperatura_c': temperatura}
        self.destroy()


class DialogoMaterial(tk.Toplevel):
    """Ventana modal para agregar o editar una fila de la lista de
    materiales. El resultado queda en `self.resultado` (dict) si se
    confirma con "Guardar", o en None si se cancela."""

    def __init__(self, master, material=None):
        super().__init__(master)
        self.resultado = None
        var_mc_global = getattr(master, 'var_usar_mc', None)
        mc_habilitado = bool(var_mc_global is not None and var_mc_global.get())
        metodos_default = ['simplificado', 'completo']
        if mc_habilitado:
            metodos_default.append('minimos_cuadrados')
        material = material or {
            'nombre': '', 'archivo': '', 'modelo': gf.SIN_MODELO,
            'temperatura': 25.0, 'metodos': metodos_default,
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
        _crear_indicador_archivo(
            frm, self.var_archivo, getattr(master, 'var_carpeta_datos', None)).grid(
            row=1, column=2, padx=(4, 0))
        ttk.Button(frm, text="Examinar...", command=self._examinar_archivo).grid(
            row=1, column=3, padx=(6, 0))

        # Checkbox y configuracion del filtro sobre el S11 (Touchstone):
        frm_filtro_box = ttk.Frame(frm)
        frm_filtro_box.grid(row=2, column=1, columnspan=3, sticky="w", pady=(2, 6))

        frm_filtro_f1 = ttk.Frame(frm_filtro_box)
        frm_filtro_f1.pack(anchor="w")

        filtrar_init = bool(material.get('filtrar_s11') or material.get('filtrar_sg', False))
        tipo_init = material.get('tipo_filtro', 'pasabajos' if material.get('pb_fcorte') else 'savitzky_golay')
        tipo_combo_map = {
            'pasabajos': "Pasabajos (Butterworth)",
            'savitzky_golay': "Savitzky-Golay",
            'mediana_movil': "Mediana móvil",
            'splines': "Splines de suavizado",
        }
        tipo_combo_val = tipo_combo_map.get(tipo_init, "Savitzky-Golay")

        self.var_filtrar_s11 = tk.BooleanVar(value=filtrar_init)
        self.var_filtrar_sg = self.var_filtrar_s11  # retrocompatibilidad
        self.chk_filtrar_s11 = ttk.Checkbutton(
            frm_filtro_f1, text="Aplicar filtro sobre este S11",
            variable=self.var_filtrar_s11,
            command=self._al_cambiar_filtrar_s11)
        self.chk_filtrar_s11.pack(side="left")

        self.lbl_filtro_tipo = ttk.Label(frm_filtro_f1, text="  Tipo:")
        self.lbl_filtro_tipo.pack(side="left")
        self.var_filtro_tipo = tk.StringVar(value=tipo_combo_val)
        self.combo_filtro_tipo = ttk.Combobox(
            frm_filtro_f1, textvariable=self.var_filtro_tipo, state="readonly", width=22,
            values=["Pasabajos (Butterworth)", "Savitzky-Golay", "Mediana móvil", "Splines de suavizado"])
        self.combo_filtro_tipo.pack(side="left", padx=(4, 6))
        self.combo_filtro_tipo.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_filtrar_s11())

        frm_filtro_f2 = ttk.Frame(frm_filtro_box)
        frm_filtro_f2.pack(anchor="w", pady=(2, 0))

        # Parametros Pasabajos
        self.lbl_pb_ord = ttk.Label(frm_filtro_f2, text="   Orden PB:")
        self.lbl_pb_ord.pack(side="left")
        self.var_pb_ord = tk.IntVar(value=int(material.get('pb_orden', 2) or 2))
        self.spin_pb_ord = ttk.Spinbox(
            frm_filtro_f2, from_=1, to=5, increment=1, textvariable=self.var_pb_ord, width=3)
        self.spin_pb_ord.pack(side="left", padx=(2, 6))

        self.lbl_pb_fc = ttk.Label(frm_filtro_f2, text="Corte PB (fc/fn):")
        self.lbl_pb_fc.pack(side="left")
        self.var_pb_fc = tk.DoubleVar(value=float(material.get('pb_fcorte', 0.10) or 0.10))
        self.spin_pb_fc = ttk.Spinbox(
            frm_filtro_f2, from_=0.01, to=0.50, increment=0.01, textvariable=self.var_pb_fc, width=5)
        self.spin_pb_fc.pack(side="left", padx=(2, 6))

        # Parametros Savitzky-Golay
        self.lbl_sg_win = ttk.Label(frm_filtro_f2, text="   Ventana SG:")
        self.lbl_sg_win.pack(side="left")
        self.var_sg_win = tk.IntVar(value=int(material.get('sg_window', 15) or 15))
        self.spin_sg_win = ttk.Spinbox(
            frm_filtro_f2, from_=3, to=101, increment=2, textvariable=self.var_sg_win, width=4)
        self.spin_sg_win.pack(side="left", padx=(2, 6))

        self.lbl_sg_ord = ttk.Label(frm_filtro_f2, text="Orden SG:")
        self.lbl_sg_ord.pack(side="left")
        self.var_sg_ord = tk.IntVar(value=int(material.get('sg_polyorder', 3) or 3))
        self.spin_sg_ord = ttk.Spinbox(
            frm_filtro_f2, from_=1, to=5, increment=1, textvariable=self.var_sg_ord, width=3)
        self.spin_sg_ord.pack(side="left", padx=(2, 6))

        # Parametros Mediana móvil
        self.lbl_med_win = ttk.Label(frm_filtro_f2, text="   Ventana Med:")
        self.var_med_win = tk.IntVar(value=int(material.get('med_window', 15) or 15))
        self.spin_med_win = ttk.Spinbox(
            frm_filtro_f2, from_=3, to=101, increment=2, textvariable=self.var_med_win, width=4)

        # Parametros Splines
        self.lbl_spline_s = ttk.Label(frm_filtro_f2, text="   Factor s:")
        self.var_spline_s = tk.DoubleVar(value=float(material.get('spline_s', 0.10) or 0.10))
        self.spin_spline_s = ttk.Spinbox(
            frm_filtro_f2, from_=0.0001, to=50.0, increment=0.01, textvariable=self.var_spline_s, width=6)

        # Rango
        self.lbl_mat_rango = ttk.Label(frm_filtro_f2, text="   Rango:")
        self.lbl_mat_rango.pack(side="left")
        self.var_mat_rango = tk.StringVar(value=material.get('modo_rango', 'todo'))
        self.radio_mat_rango_todo = ttk.Radiobutton(
            frm_filtro_f2, text="Todo", value="todo", variable=self.var_mat_rango,
            command=self._al_cambiar_filtrar_s11)
        self.radio_mat_rango_todo.pack(side="left", padx=(2, 4))
        self.radio_mat_rango_sub = ttk.Radiobutton(
            frm_filtro_f2, text="[f1, f2]:", value="rango", variable=self.var_mat_rango,
            command=self._al_cambiar_filtrar_s11)
        self.radio_mat_rango_sub.pack(side="left", padx=(2, 6))

        self.lbl_mat_f1 = ttk.Label(frm_filtro_f2, text="f1 (GHz):")
        self.var_mat_f1 = tk.DoubleVar(value=float(material.get('f1_ghz', 0.5) or 0.5))
        self.spin_mat_f1 = ttk.Spinbox(
            frm_filtro_f2, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_mat_f1, width=6)
        self.lbl_mat_f2 = ttk.Label(frm_filtro_f2, text="f2 (GHz):")
        self.var_mat_f2 = tk.DoubleVar(value=float(material.get('f2_ghz', 6.0) or 6.0))
        self.spin_mat_f2 = ttk.Spinbox(
            frm_filtro_f2, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_mat_f2, width=6)

        _tooltip(frm_filtro_box,
                 "Aplica un filtro digital (Pasabajos Butterworth o Savitzky-Golay) de fase cero\n"
                 "a Re(S11) e Im(S11) para atenuar ruido y rizado sin distorsionar la fase.")

        ttk.Label(frm, text="Modelo teorico:").grid(row=3, column=0, sticky="w", pady=4)
        self.var_modelo_etiqueta = tk.StringVar(value=gf.etiqueta_de_modelo(material.get('modelo')))
        self.combo_modelo = ttk.Combobox(
            frm, textvariable=self.var_modelo_etiqueta, state="readonly",
            values=[et for _, et in gf.listar_modelos_teoricos()], width=33)
        self.combo_modelo.grid(row=3, column=1, columnspan=3, sticky="ew", pady=4)
        self.combo_modelo.bind("<<ComboboxSelected>>", self._al_cambiar_modelo)

        ttk.Label(frm, text="Temperatura (°C):").grid(row=4, column=0, sticky="w", pady=4)
        self.var_temperatura = tk.DoubleVar(value=float(material.get('temperatura') or 25.0))
        self.spin_temperatura = ttk.Spinbox(
            frm, from_=-20, to=150, increment=0.5, textvariable=self.var_temperatura, width=10)
        self.spin_temperatura.grid(row=4, column=1, sticky="w", pady=4)
        _tooltip(self.spin_temperatura,
                 "El modelo teorico usa la temperatura tabulada mas cercana\n"
                 "(pasos de 5°C, NPL MAT 23). No hace falta que coincida\n"
                 "exactamente con un escalon de la tabla.")

        # Si el metodo completo esta deshabilitado globalmente (pestaña de
        # Calibracion), no tiene sentido ofrecer "Completo" aca: se
        # deshabilita el checkbox y se fuerza a False, con una nota.
        var_usar_completo_global = getattr(master, 'var_usar_metodo_completo', None)
        completo_deshabilitado = (
            var_usar_completo_global is not None and not var_usar_completo_global.get())
        mc_deshabilitado = not mc_habilitado

        ttk.Label(frm, text="Metodos a calcular:").grid(row=5, column=0, sticky="nw", pady=4)
        metodos = material.get('metodos') or ['simplificado', 'completo']
        self.var_simplificado = tk.BooleanVar(value='simplificado' in metodos)
        self.var_completo = tk.BooleanVar(
            value=('completo' in metodos) and not completo_deshabilitado)
        self.var_minimos_cuadrados = tk.BooleanVar(
            value=('minimos_cuadrados' in metodos) and not mc_deshabilitado)
        frm_metodos = ttk.Frame(frm)
        frm_metodos.grid(row=5, column=1, columnspan=3, sticky="w")
        ttk.Checkbutton(frm_metodos, text="Simplificado (corto + aire + patron 3)",
                        variable=self.var_simplificado).pack(anchor="w")
        chk_completo = ttk.Checkbutton(frm_metodos, text="Completo (+ patron 4)",
                                        variable=self.var_completo)
        chk_completo.pack(anchor="w")
        if completo_deshabilitado:
            chk_completo.configure(state="disabled")
            ttk.Label(frm_metodos,
                      text="(deshabilitado: el metodo completo est\u00e1 apagado en "
                           "la pesta\u00f1a de Calibracion)",
                      style="Ayuda.TLabel").pack(anchor="w")
        chk_mc = ttk.Checkbutton(
            frm_metodos, text="Cuadrados minimos (todos los patrones)",
            variable=self.var_minimos_cuadrados)
        chk_mc.pack(anchor="w")
        if mc_deshabilitado:
            chk_mc.configure(state="disabled")
            ttk.Label(frm_metodos,
                      text="(deshabilitado: activalo en la pesta\u00f1a de Calibracion)",
                      style="Ayuda.TLabel").pack(anchor="w")

        ttk.Separator(frm).grid(row=6, column=0, columnspan=4, sticky="ew", pady=10)

        # Cartel de advertencia del modelo teorico elegido: se llena solo
        # al cambiar el combobox (ver _al_cambiar_modelo). Solo aparece si
        # el modelo tiene advertencia -- p.ej. acetona/ciclohexano/
        # silicona, que no tienen ecuacion de Debye.
        self.cartel_modelo = CartelAdvertencia(frm, wraplength=430)
        self.cartel_modelo.grid(row=7, column=0, columnspan=4, sticky="ew",
                                 pady=(0, 8))
        self.cartel_modelo.ocultar()

        frm_botones = ttk.Frame(frm)
        frm_botones.grid(row=8, column=0, columnspan=4, sticky="e")
        ttk.Button(frm_botones, text="Cancelar", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(frm_botones, text="Guardar", command=self._guardar).pack(side="right")

        self._al_cambiar_filtrar_s11()
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
        clave = gf.clave_de_etiqueta(self.var_modelo_etiqueta.get())
        activo = clave != gf.SIN_MODELO
        self.spin_temperatura.configure(state=("normal" if activo else "disabled"))
        # El cartel de advertencia se actualiza junto con el modelo: si el
        # modelo elegido no tiene advertencia, se oculta solo.
        self.cartel_modelo.mostrar_modelo(clave if activo else None)
        self.update_idletasks()

    def _al_cambiar_filtrar_s11(self):
        activo = bool(self.var_filtrar_s11.get())
        estado = "normal" if activo else "disabled"
        self.lbl_filtro_tipo.configure(state=estado)
        self.combo_filtro_tipo.configure(state=("readonly" if activo else "disabled"))

        tipo = self.var_filtro_tipo.get()
        for w in (self.lbl_pb_ord, self.spin_pb_ord, self.lbl_pb_fc, self.spin_pb_fc,
                  self.lbl_sg_win, self.spin_sg_win, self.lbl_sg_ord, self.spin_sg_ord,
                  self.lbl_med_win, self.spin_med_win, self.lbl_spline_s, self.spin_spline_s):
            w.pack_forget()

        if activo:
            if tipo == "Pasabajos (Butterworth)":
                self.lbl_pb_ord.pack(side="left", before=self.lbl_mat_rango)
                self.spin_pb_ord.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)
                self.lbl_pb_fc.pack(side="left", before=self.lbl_mat_rango)
                self.spin_pb_fc.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)
            elif tipo == "Mediana móvil":
                self.lbl_med_win.pack(side="left", before=self.lbl_mat_rango)
                self.spin_med_win.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)
            elif tipo == "Splines de suavizado":
                self.lbl_spline_s.pack(side="left", before=self.lbl_mat_rango)
                self.spin_spline_s.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)
            else:  # Savitzky-Golay
                self.lbl_sg_win.pack(side="left", before=self.lbl_mat_rango)
                self.spin_sg_win.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)
                self.lbl_sg_ord.pack(side="left", before=self.lbl_mat_rango)
                self.spin_sg_ord.pack(side="left", padx=(2, 6), before=self.lbl_mat_rango)

        self.lbl_mat_rango.configure(state=estado)
        self.radio_mat_rango_todo.configure(state=estado)
        self.radio_mat_rango_sub.configure(state=estado)

        es_rango = (self.var_mat_rango.get() == "rango")
        if activo and es_rango:
            self.lbl_mat_f1.pack(side="left")
            self.spin_mat_f1.pack(side="left", padx=(2, 6))
            self.lbl_mat_f2.pack(side="left")
            self.spin_mat_f2.pack(side="left", padx=(2, 6))
        else:
            self.lbl_mat_f1.pack_forget()
            self.spin_mat_f1.pack_forget()
            self.lbl_mat_f2.pack_forget()
            self.spin_mat_f2.pack_forget()

    def _al_cambiar_filtrar_sg(self):
        self._al_cambiar_filtrar_s11()

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
        if self.var_minimos_cuadrados.get():
            metodos.append('minimos_cuadrados')
        if not metodos:
            messagebox.showwarning("Sin metodo", "Marca al menos un metodo.", parent=self)
            return
        try:
            temperatura = float(self.var_temperatura.get())
        except (tk.TclError, ValueError):
            temperatura = 25.0
        tipo_map = {
            "Pasabajos (Butterworth)": "pasabajos",
            "Savitzky-Golay": "savitzky_golay",
            "Mediana móvil": "mediana_movil",
            "Splines de suavizado": "splines",
        }
        tipo_f = tipo_map.get(self.var_filtro_tipo.get(), 'savitzky_golay')
        self.resultado = {
            'nombre': nombre,
            'archivo': self.var_archivo.get().strip(),
            'modelo': gf.clave_de_etiqueta(self.var_modelo_etiqueta.get()),
            'temperatura': temperatura,
            'metodos': metodos,
            'filtrar_s11': bool(self.var_filtrar_s11.get()),
            'filtrar_sg': bool(self.var_filtrar_s11.get() and tipo_f == 'savitzky_golay'),
            'tipo_filtro': tipo_f,
            'sg_window': int(self.var_sg_win.get() or 15),
            'sg_polyorder': int(self.var_sg_ord.get() or 3),
            'pb_orden': int(self.var_pb_ord.get() or 2),
            'pb_fcorte': float(self.var_pb_fc.get() or 0.10),
            'med_window': int(self.var_med_win.get() or 15),
            'spline_s': float(self.var_spline_s.get() or 0.10),
            'modo_rango': self.var_mat_rango.get(),
            'f1_ghz': float(self.var_mat_f1.get() or 0.5),
            'f2_ghz': float(self.var_mat_f2.get() or 6.0),
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
        self._patrones_adicionales = []  # lista de dicts {archivo, modelo, temperatura_c}, cuadrados minimos
        self._cola_analisis = None
        self._analisis_corriendo = False
        self._evento_cancelar = None
        self._ultima_carpeta_run = None  # carpeta especifica (fecha/hora) de la ultima corrida
        self._graficos_disponibles = {}  # {etiqueta: funcion() -> Figure}, se llena tras correr
        self._rutas_preview_s11 = {}  # {etiqueta: ruta_completa}, ver _refrescar_lista_preview_s11
        self._materiales_preview_s11 = {}  # {etiqueta: dict_material} si es un material
        self._patrones_preview_s11 = {}  # {etiqueta: clave_patron} si es un patron de calibracion
        self._curvas_comparar = []  # lista de dicts {etiqueta, archivo, columna}, pestaña 5
        self._curvas_modelos = []  # lista de dicts {etiqueta, modelo, temperatura, f_min_ghz, f_max_ghz, n_puntos}, pestaña 6

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

        self.tab_calibracion = ttk.Frame(self.notebook, padding=0)
        self.tab_materiales = ttk.Frame(self.notebook, padding=12)
        self.tab_preview_s11 = ttk.Frame(self.notebook, padding=12)
        self.tab_salida = ttk.Frame(self.notebook, padding=12)
        self.tab_comparar = ttk.Frame(self.notebook, padding=12)
        self.tab_modelos = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.tab_calibracion, text="  1. Calibracion  ")
        self.notebook.add(self.tab_materiales, text="  2. Materiales  ")
        self.notebook.add(self.tab_preview_s11, text="  3. Vista previa S11  ")
        self.notebook.add(self.tab_salida, text="  4. Salida y ejecucion  ")
        self.notebook.add(self.tab_comparar, text="  5. Comparar mediciones  ")
        self.notebook.add(self.tab_modelos, text="  6. Modelos teoricos  ")

        self._construir_tab_calibracion()
        self._construir_tab_materiales()
        self._construir_tab_preview_s11()
        self._construir_tab_salida()
        self._construir_tab_comparar()
        self._construir_tab_modelos()

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
        self.scroll_calibracion = ScrollableFrame(self.tab_calibracion, padding=12)
        self.scroll_calibracion.pack(fill="both", expand=True)
        f = self.scroll_calibracion.scrollable_content
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Patrones de calibracion", style="Titulo.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(
            f, text="El metodo simplificado usa corto + aire + patron 3; el metodo "
                    "completo ademas usa patron 4 (se puede deshabilitar si no interesa "
                    "el metodo completo). Patron 3 y patron 4 pueden ser cualquier par "
                    "de liquidos con modelo teorico conocido (por defecto agua y "
                    "alcohol isopropilico, pero no hace falta calibrar si o si con "
                    "esos dos).",
            style="Ayuda.TLabel", wraplength=800, justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(f, text="Carpeta base (opcional):").grid(row=2, column=0, sticky="w", pady=3)
        self.var_carpeta_datos = tk.StringVar()
        entry_carpeta = ttk.Entry(f, textvariable=self.var_carpeta_datos)
        entry_carpeta.grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Button(f, text="Examinar...", command=self._examinar_carpeta_datos).grid(
            row=2, column=3, padx=(6, 0), pady=3)
        _tooltip(entry_carpeta,
                 "Opcional. Si elegis los archivos con 'Examinar...' (rutas\n"
                 "absolutas), podes dejar esto vacio.")

        # Indicador ✓/✗ de archivo (feedback inmediato, sin depender del
        # boton "Verificar archivos"): se actualiza solo con cada tecla,
        # via trace_add sobre el StringVar del campo (y de la carpeta
        # base, ya que la ruta completa depende de ambos).
        self._widgets_patron4 = []  # para habilitar/deshabilitar en bloque

        def _indicador_archivo(parent, var_archivo, col, fila_):
            label = _crear_indicador_archivo(parent, var_archivo, self.var_carpeta_datos)
            label.grid(row=fila_, column=col, padx=(4, 0))
            return label

        # Cortocircuito y aire: son siempre los mismos 2 (no son liquidos
        # con modelo de Debye, y el algoritmo los necesita especificamente
        # a ellos), asi que no tienen selector de modelo ni temperatura.
        self.vars_calibracion = {}
        fila = 3
        for clave, etiqueta in [('corto', "Cortocircuito:"), ('aire', "Aire:")]:
            ttk.Label(f, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
            var = tk.StringVar()
            self.vars_calibracion[clave] = var
            ttk.Entry(f, textvariable=var).grid(row=fila, column=1, sticky="ew", pady=3)
            _indicador_archivo(f, var, 2, fila)
            ttk.Button(f, text="Examinar...",
                       command=lambda c=clave: self._examinar_archivo_calibracion(c)).grid(
                row=fila, column=3, padx=(6, 0), pady=3)
            fila += 1

        # Patron 3 y patron 4: aca es donde entra la flexibilidad -- cada
        # uno con su propio archivo, MODELO TEORICO elegible (combobox,
        # misma lista que usa un material en la pestaña 2) y temperatura,
        # en vez de estar fijos a agua/alcohol isopropilico como antes.
        self.vars_modelo_calibracion = {}
        self.vars_temp_calibracion = {}
        modelos_calibracion = [et for _, et in gf.listar_modelos_teoricos_calibracion()]
        tooltips_patron = {
            'patron3': "Temperatura real de este liquido durante la calibracion.\n"
                       "Afecta la precision de los dos metodos de conversion.",
            'patron4': "Temperatura real de este liquido durante la calibracion.\n"
                       "Solo lo usa el metodo completo, pero afecta el resultado de\n"
                       "TODOS los materiales analizados con ese metodo (entra en el\n"
                       "calculo de la conductancia Gn), no solo el de este patron.",
        }
        for clave, etiqueta in [('patron3', "Patron 3:"), ('patron4', "Patron 4:")]:
            ttk.Label(f, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3)
            var_archivo = tk.StringVar()
            self.vars_calibracion[clave] = var_archivo
            entry_p = ttk.Entry(f, textvariable=var_archivo)
            entry_p.grid(row=fila, column=1, sticky="ew", pady=3)
            _indicador_archivo(f, var_archivo, 2, fila)
            btn_examinar = ttk.Button(
                f, text="Examinar...",
                command=lambda c=clave: self._examinar_archivo_calibracion(c))
            btn_examinar.grid(row=fila, column=3, padx=(6, 0), pady=3)

            lbl_liquido = ttk.Label(f, text="  Liquido:")
            lbl_liquido.grid(row=fila, column=4, sticky="w")
            var_modelo = tk.StringVar()
            self.vars_modelo_calibracion[clave] = var_modelo
            combo = ttk.Combobox(f, textvariable=var_modelo, state="readonly",
                                  values=modelos_calibracion, width=24)
            combo.grid(row=fila, column=5, sticky="w", padx=(4, 0), pady=3)
            # Al cambiar el liquido de un patron se refresca el cartel de
            # advertencia de calibracion (ver _refrescar_cartel_calibracion).
            combo.bind("<<ComboboxSelected>>",
                       lambda _e: self._refrescar_cartel_calibracion())

            lbl_temp = ttk.Label(f, text="  T (°C):")
            lbl_temp.grid(row=fila, column=6, sticky="w")
            var_temp = tk.DoubleVar(value=25.0)
            self.vars_temp_calibracion[clave] = var_temp
            spin = ttk.Spinbox(f, from_=-20, to=150, increment=0.5,
                                textvariable=var_temp, width=8)
            spin.grid(row=fila, column=7, sticky="w", padx=(2, 0), pady=3)
            _tooltip(spin, tooltips_patron[clave])

            if clave == 'patron4':
                self._widgets_patron4 = [entry_p, btn_examinar, lbl_liquido, combo,
                                          lbl_temp, spin]
            fila += 1

        # --- Filtro sobre patrones de calibracion ---
        frm_filtro_pat_box = ttk.Frame(f)
        frm_filtro_pat_box.grid(row=fila, column=0, columnspan=8, sticky="w", pady=(8, 2))

        frm_filtro_pat_f1 = ttk.Frame(frm_filtro_pat_box)
        frm_filtro_pat_f1.pack(anchor="w")

        self.var_filtrar_patrones_sg = tk.BooleanVar(value=False)
        self.chk_filtrar_patrones_sg = ttk.Checkbutton(
            frm_filtro_pat_f1, text="Aplicar filtro sobre patrones de calibración",
            variable=self.var_filtrar_patrones_sg,
            command=self._al_cambiar_filtrar_patrones_sg)
        self.chk_filtrar_patrones_sg.pack(side="left")

        self.lbl_pat_filtro_tipo = ttk.Label(frm_filtro_pat_f1, text="  Tipo:")
        self.lbl_pat_filtro_tipo.pack(side="left")
        self.var_patrones_tipo_filtro = tk.StringVar(value="Pasabajos (Butterworth)")
        self.combo_patrones_tipo_filtro = ttk.Combobox(
            frm_filtro_pat_f1, textvariable=self.var_patrones_tipo_filtro, state="readonly", width=22,
            values=["Pasabajos (Butterworth)", "Savitzky-Golay", "Mediana móvil", "Splines de suavizado"])
        self.combo_patrones_tipo_filtro.pack(side="left", padx=(4, 10))
        self.combo_patrones_tipo_filtro.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_filtrar_patrones_sg())

        # Parametros Pasabajos
        self.lbl_patrones_pb_ord = ttk.Label(frm_filtro_pat_f1, text="Orden PB:")
        self.lbl_patrones_pb_ord.pack(side="left")
        self.var_patrones_pb_ord = tk.IntVar(value=2)
        self.spin_patrones_pb_ord = ttk.Spinbox(
            frm_filtro_pat_f1, from_=1, to=5, increment=1,
            textvariable=self.var_patrones_pb_ord, width=3)
        self.spin_patrones_pb_ord.pack(side="left", padx=(2, 6))

        self.lbl_patrones_pb_fc = ttk.Label(frm_filtro_pat_f1, text="Corte PB (fc/fn):")
        self.lbl_patrones_pb_fc.pack(side="left")
        self.var_patrones_pb_fc = tk.DoubleVar(value=0.10)
        self.spin_patrones_pb_fc = ttk.Spinbox(
            frm_filtro_pat_f1, from_=0.01, to=0.50, increment=0.01,
            textvariable=self.var_patrones_pb_fc, width=5)
        self.spin_patrones_pb_fc.pack(side="left", padx=(2, 10))

        # Parametros Savitzky-Golay
        self.lbl_patrones_sg_win = ttk.Label(frm_filtro_pat_f1, text="Ventana SG:")
        self.lbl_patrones_sg_win.pack(side="left")
        self.var_patrones_sg_win = tk.IntVar(value=15)
        self.spin_patrones_sg_win = ttk.Spinbox(
            frm_filtro_pat_f1, from_=3, to=101, increment=2,
            textvariable=self.var_patrones_sg_win, width=4)
        self.spin_patrones_sg_win.pack(side="left", padx=(2, 6))

        self.lbl_patrones_sg_ord = ttk.Label(frm_filtro_pat_f1, text="Orden SG:")
        self.lbl_patrones_sg_ord.pack(side="left")
        self.var_patrones_sg_ord = tk.IntVar(value=3)
        self.spin_patrones_sg_ord = ttk.Spinbox(
            frm_filtro_pat_f1, from_=1, to=5, increment=1,
            textvariable=self.var_patrones_sg_ord, width=3)
        self.spin_patrones_sg_ord.pack(side="left", padx=(2, 10))

        # Parametros Mediana móvil
        self.lbl_patrones_med_win = ttk.Label(frm_filtro_pat_f1, text="Ventana Med:")
        self.var_patrones_med_win = tk.IntVar(value=15)
        self.spin_patrones_med_win = ttk.Spinbox(
            frm_filtro_pat_f1, from_=3, to=101, increment=2,
            textvariable=self.var_patrones_med_win, width=4)

        # Parametros Splines
        self.lbl_patrones_spline_s = ttk.Label(frm_filtro_pat_f1, text="Factor s:")
        self.var_patrones_spline_s = tk.DoubleVar(value=0.10)
        self.spin_patrones_spline_s = ttk.Spinbox(
            frm_filtro_pat_f1, from_=0.0001, to=50.0, increment=0.01,
            textvariable=self.var_patrones_spline_s, width=6)

        # Rango
        self.lbl_patrones_rango = ttk.Label(frm_filtro_pat_f1, text="  Rango:")
        self.lbl_patrones_rango.pack(side="left")
        self.var_patrones_rango = tk.StringVar(value="todo")
        self.radio_pat_rango_todo = ttk.Radiobutton(
            frm_filtro_pat_f1, text="Todo", value="todo", variable=self.var_patrones_rango,
            command=self._al_cambiar_filtrar_patrones_sg)
        self.radio_pat_rango_todo.pack(side="left", padx=(2, 4))
        self.radio_pat_rango_sub = ttk.Radiobutton(
            frm_filtro_pat_f1, text="[f1, f2]:", value="rango", variable=self.var_patrones_rango,
            command=self._al_cambiar_filtrar_patrones_sg)
        self.radio_pat_rango_sub.pack(side="left", padx=(2, 6))

        self.lbl_pat_f1 = ttk.Label(frm_filtro_pat_f1, text="f1 (GHz):")
        self.var_pat_f1 = tk.DoubleVar(value=0.5)
        self.spin_pat_f1 = ttk.Spinbox(
            frm_filtro_pat_f1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_pat_f1, width=6)
        self.lbl_pat_f2 = ttk.Label(frm_filtro_pat_f1, text="f2 (GHz):")
        self.var_pat_f2 = tk.DoubleVar(value=6.0)
        self.spin_pat_f2 = ttk.Spinbox(
            frm_filtro_pat_f1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_pat_f2, width=6)

        # Fila 2: Aplicar en
        frm_filtro_pat_f2 = ttk.Frame(frm_filtro_pat_box)
        frm_filtro_pat_f2.pack(anchor="w", pady=(3, 0))

        self.lbl_patrones_sg_en = ttk.Label(frm_filtro_pat_f2, text="   Aplicar en:")
        self.lbl_patrones_sg_en.pack(side="left")

        self.var_sg_pat_corto = tk.BooleanVar(value=True)
        self.chk_sg_pat_corto = ttk.Checkbutton(frm_filtro_pat_f2, text="Corto", variable=self.var_sg_pat_corto)
        self.chk_sg_pat_corto.pack(side="left", padx=(4, 2))

        self.var_sg_pat_aire = tk.BooleanVar(value=True)
        self.chk_sg_pat_aire = ttk.Checkbutton(frm_filtro_pat_f2, text="Aire", variable=self.var_sg_pat_aire)
        self.chk_sg_pat_aire.pack(side="left", padx=(2, 2))

        self.var_sg_pat_p3 = tk.BooleanVar(value=True)
        self.chk_sg_pat_p3 = ttk.Checkbutton(frm_filtro_pat_f2, text="Patrón 3", variable=self.var_sg_pat_p3)
        self.chk_sg_pat_p3.pack(side="left", padx=(2, 2))

        self.var_sg_pat_p4 = tk.BooleanVar(value=True)
        self.chk_sg_pat_p4 = ttk.Checkbutton(frm_filtro_pat_f2, text="Patrón 4", variable=self.var_sg_pat_p4)
        self.chk_sg_pat_p4.pack(side="left", padx=(2, 2))

        self.var_sg_pat_adic = tk.BooleanVar(value=True)
        self.chk_sg_pat_adic = ttk.Checkbutton(frm_filtro_pat_f2, text="Adicionales", variable=self.var_sg_pat_adic)
        self.chk_sg_pat_adic.pack(side="left", padx=(2, 2))

        _tooltip(frm_filtro_pat_box,
                 "Aplica un filtro digital (Pasabajos Butterworth o Savitzky-Golay) de fase cero a Re(S11) e Im(S11)\n"
                 "de los patrones seleccionados para atenuar ruido y rizado antes de calcular la calibración y Gn(f).")
        fila += 1

        # --- Habilitar/deshabilitar el metodo completo por completo ---
        frm_completo = ttk.Frame(f)
        frm_completo.grid(row=fila, column=0, columnspan=8, sticky="w", pady=(10, 0))
        self.var_usar_metodo_completo = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frm_completo, text="Habilitar metodo completo (usa Patron 4 y calcula Gn)",
            variable=self.var_usar_metodo_completo,
            command=self._al_cambiar_usar_metodo_completo,
        ).pack(side="left")
        fila += 1

        frm_estrategia = ttk.Frame(f)
        frm_estrategia.grid(row=fila, column=0, columnspan=8, sticky="w", pady=(4, 0))
        ttk.Label(frm_estrategia, text="   Punto de arranque de Gn:").pack(side="left")
        self.var_estrategia_gn = tk.StringVar(value="minimo_gn")
        self.radio_estrategia_minimo = ttk.Radiobutton(
            frm_estrategia, text="Autom\u00e1tico (m\u00ednimo de |Gn|)",
            value="minimo_gn", variable=self.var_estrategia_gn)
        self.radio_estrategia_minimo.pack(side="left", padx=(6, 0))
        self.radio_estrategia_max = ttk.Radiobutton(
            frm_estrategia, text="Frecuencia m\u00e1s alta (cl\u00e1sico)",
            value="frecuencia_maxima", variable=self.var_estrategia_gn)
        self.radio_estrategia_max.pack(side="left", padx=(10, 0))
        self.radio_estrategia_comparar = ttk.Radiobutton(
            frm_estrategia, text="Comparar ambas",
            value="comparar_ambas", variable=self.var_estrategia_gn)
        self.radio_estrategia_comparar.pack(side="left", padx=(10, 0))
        _tooltip(frm_estrategia,
                 "Donde arranca la 'marcha en frecuencia' del metodo completo:\n"
                 "- Automatico: en el punto donde |Gn(f)| es realmente minimo en\n"
                 "  todo el barrido (mas confiable, sirve para cualquier forma de\n"
                 "  Gn(f), incluso si el minimo no cae en un extremo).\n"
                 "- Clasico: siempre en la frecuencia mas alta. Valido solo si Gn\n"
                 "  decrece en forma monotona con la frecuencia; se deja disponible\n"
                 "  para comparar contra el automatico.\n"
                 "- Comparar ambas: corre las dos y las muestra juntas (patrones 3/4\n"
                 "  y cada material con metodo completo), para ver de un vistazo si\n"
                 "  la eleccion cambia algo con tus datos, sin correr el analisis dos\n"
                 "  veces a mano.")
        fila += 1

        # --- Acondicionamiento de Gn ---
        frm_acond_chk = ttk.Frame(f)
        frm_acond_chk.grid(row=fila, column=0, columnspan=8, sticky="w", pady=(8, 0))
        self.var_acondicionar_gn = tk.BooleanVar(value=False)
        self.chk_acondicionar_gn = ttk.Checkbutton(
            frm_acond_chk, text="Acondicionar Gn (mitigar ruido / divergencia a bajas frecuencias)",
            variable=self.var_acondicionar_gn,
            command=self._al_cambiar_acondicionar_gn,
        )
        self.chk_acondicionar_gn.pack(side="left")
        _tooltip(frm_acond_chk,
                 "Permite modificar la conductancia Gn(f) a bajas frecuencias o filtrarla:\n"
                 "- Transicion orden 5 (f^5): empalme suave de orden 5 entre 0 y Gn original sin sobrepicos.\n"
                 "- Filtro Savitzky-Golay: suavizado de fase cero en todo el espectro de Gn.\n"
                 "- Transicion orden 5 + Savitzky-Golay: transicion en bajas frecuencias y suavizado en altas.\n"
                 "- Cero clamping: anula Gn a 0 por debajo de una frecuencia.\n"
                 "- Tapering: desvanece suavemente a 0 con ventana de coseno alzado.")
        fila += 1

        self.frm_acond_opciones = ttk.Frame(f)
        self.frm_acond_opciones.grid(row=fila, column=0, columnspan=8, sticky="w", padx=(16, 0), pady=(3, 0))

        # Fila 1: Metodo, Modo y Comparar
        self.frm_acond_f1 = ttk.Frame(self.frm_acond_opciones)
        self.frm_acond_f1.pack(anchor="w", pady=(0, 4))

        ttk.Label(self.frm_acond_f1, text="Metodo:").pack(side="left")
        self.var_acond_tipo = tk.StringVar(value="Transicion orden 5 (f^5)")
        self.combo_acond_tipo = ttk.Combobox(
            self.frm_acond_f1, textvariable=self.var_acond_tipo, state="readonly", width=32,
            values=[
                "Transicion orden 5 (f^5)",
                "Filtro Pasabajos (Butterworth)",
                "Filtro Savitzky-Golay",
                "Filtro Mediana móvil",
                "Splines de suavizado",
                "Transicion orden 5 + Pasabajos",
                "Transicion orden 5 + Savitzky-Golay",
                "Cero clamping (corte duro)",
                "Tapering (ventana suave)"
            ],
        )
        self.combo_acond_tipo.pack(side="left", padx=(4, 10))
        self.combo_acond_tipo.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_acondicionar_gn())

        self.lbl_acond_modo = ttk.Label(self.frm_acond_f1, text="Modo:")
        self.lbl_acond_modo.pack(side="left")
        self.var_acond_modo = tk.StringVar(value="automatico")
        self.radio_acond_auto = ttk.Radiobutton(
            self.frm_acond_f1, text="Automatico", value="automatico",
            variable=self.var_acond_modo, command=self._al_cambiar_acondicionar_gn)
        self.radio_acond_auto.pack(side="left", padx=(2, 4))
        self.radio_acond_manual = ttk.Radiobutton(
            self.frm_acond_f1, text="Manual", value="manual",
            variable=self.var_acond_modo, command=self._al_cambiar_acondicionar_gn)
        self.radio_acond_manual.pack(side="left", padx=(0, 14))

        self.var_comparar_gn_original = tk.BooleanVar(value=True)
        self.chk_comparar_gn_original = ttk.Checkbutton(
            self.frm_acond_f1, text="Comparar con Gn original",
            variable=self.var_comparar_gn_original)
        self.chk_comparar_gn_original.pack(side="left", padx=(4, 0))
        _tooltip(self.chk_comparar_gn_original,
                 "Calcula y grafica en simultaneo la curva con Gn original y con\n"
                 "Gn acondicionado para comparar el impacto en la permitividad.")

        # Fila 2: Parametros dinamicos
        self.frm_acond_f2 = ttk.Frame(self.frm_acond_opciones)
        self.frm_acond_f2.pack(anchor="w", pady=(2, 0))

        self.lbl_acond_umbral = ttk.Label(self.frm_acond_f2, text="Umbral salto (%):")
        self.lbl_acond_umbral.pack(side="left")
        self.var_acond_umbral_pct = tk.DoubleVar(value=30.0)
        self.spin_acond_umbral = ttk.Spinbox(
            self.frm_acond_f2, from_=1.0, to=500.0, increment=5.0,
            textvariable=self.var_acond_umbral_pct, width=6)
        self.spin_acond_umbral.pack(side="left", padx=(4, 10))
        _tooltip(self.spin_acond_umbral, "Porcentaje de variacion relativa |Delta(Gn)| / |Gn| para disparar\n"
                                         "el corte/empalme buscando de frecuencias altas hacia bajas (def: 30%).")

        self.lbl_acond_f1 = ttk.Label(self.frm_acond_f2, text="f1 (GHz):")
        self.lbl_acond_f1.pack(side="left")
        self.var_acond_f1_ghz = tk.DoubleVar(value=0.0)
        self.spin_acond_f1 = ttk.Spinbox(
            self.frm_acond_f2, from_=0.0, to=100.0, increment=0.05,
            textvariable=self.var_acond_f1_ghz, width=6)
        self.spin_acond_f1.pack(side="left", padx=(4, 6))

        self.lbl_acond_f2 = ttk.Label(self.frm_acond_f2, text="f2 (GHz):")
        self.lbl_acond_f2.pack(side="left")
        self.var_acond_f2_ghz = tk.DoubleVar(value=0.5)
        self.spin_acond_f2 = ttk.Spinbox(
            self.frm_acond_f2, from_=0.001, to=100.0, increment=0.05,
            textvariable=self.var_acond_f2_ghz, width=6)
        self.spin_acond_f2.pack(side="left", padx=(4, 10))

        # Parametros Pasabajos Gn
        self.lbl_acond_pb_orden = ttk.Label(self.frm_acond_f2, text="Orden PB:")
        self.lbl_acond_pb_orden.pack(side="left")
        self.var_acond_pb_orden = tk.IntVar(value=2)
        self.spin_acond_pb_orden = ttk.Spinbox(
            self.frm_acond_f2, from_=1, to=5, increment=1,
            textvariable=self.var_acond_pb_orden, width=3)
        self.spin_acond_pb_orden.pack(side="left", padx=(4, 6))
        _tooltip(self.spin_acond_pb_orden, "Orden del filtro Butterworth pasabajos (def: 2).")

        self.lbl_acond_pb_fcorte = ttk.Label(self.frm_acond_f2, text="Corte PB (fc/fn):")
        self.lbl_acond_pb_fcorte.pack(side="left")
        self.var_acond_pb_fcorte = tk.DoubleVar(value=0.10)
        self.spin_acond_pb_fcorte = ttk.Spinbox(
            self.frm_acond_f2, from_=0.01, to=0.50, increment=0.01,
            textvariable=self.var_acond_pb_fcorte, width=5)
        self.spin_acond_pb_fcorte.pack(side="left", padx=(4, 10))
        _tooltip(self.spin_acond_pb_fcorte, "Frecuencia de corte normalizada a Nyquist fn (def: 0.10).")

        # Parametros Savitzky-Golay Gn
        self.lbl_acond_sg_window = ttk.Label(self.frm_acond_f2, text="Ventana SG:")
        self.lbl_acond_sg_window.pack(side="left")
        self.var_acond_sg_window = tk.IntVar(value=15)
        self.spin_acond_sg_window = ttk.Spinbox(
            self.frm_acond_f2, from_=3, to=101, increment=2,
            textvariable=self.var_acond_sg_window, width=4)
        self.spin_acond_sg_window.pack(side="left", padx=(4, 6))
        _tooltip(self.spin_acond_sg_window, "Longitud de ventana del filtro Savitzky-Golay (debe ser impar).")

        self.lbl_acond_sg_polyorder = ttk.Label(self.frm_acond_f2, text="Orden SG:")
        self.lbl_acond_sg_polyorder.pack(side="left")
        self.var_acond_sg_polyorder = tk.IntVar(value=3)
        self.spin_acond_sg_polyorder = ttk.Spinbox(
            self.frm_acond_f2, from_=1, to=5, increment=1,
            textvariable=self.var_acond_sg_polyorder, width=3)
        self.spin_acond_sg_polyorder.pack(side="left", padx=(4, 10))
        _tooltip(self.spin_acond_sg_polyorder, "Orden del polinomio local para el filtro Savitzky-Golay.")

        # Parametros Mediana móvil Gn
        self.lbl_acond_med_window = ttk.Label(self.frm_acond_f2, text="Ventana Med:")
        self.var_acond_med_window = tk.IntVar(value=15)
        self.spin_acond_med_window = ttk.Spinbox(
            self.frm_acond_f2, from_=3, to=101, increment=2,
            textvariable=self.var_acond_med_window, width=4)
        _tooltip(self.spin_acond_med_window, "Longitud de ventana del filtro de mediana móvil (debe ser impar).")

        # Parametros Splines Gn
        self.lbl_acond_spline_s = ttk.Label(self.frm_acond_f2, text="Factor s (Spline):")
        self.var_acond_spline_s = tk.DoubleVar(value=0.10)
        self.spin_acond_spline_s = ttk.Spinbox(
            self.frm_acond_f2, from_=0.0001, to=50.0, increment=0.01,
            textvariable=self.var_acond_spline_s, width=6)
        _tooltip(self.spin_acond_spline_s, "Factor de suavizado del spline regularizado (0 = interpolación, mayor = más suave).")
        fila += 1

        ttk.Separator(f).grid(row=fila, column=0, columnspan=8, sticky="ew", pady=10)
        fila += 1

        # --- Calibracion redundante por cuadrados minimos (5+ patrones) ---
        fila = self._construir_seccion_minimos_cuadrados(f, fila)

        ttk.Separator(f).grid(row=fila, column=0, columnspan=8, sticky="ew", pady=10)
        fila += 1

        ttk.Label(
            f, text="\u24d8 Patron 3 y patron 4 tienen que ser dos liquidos distintos "
                    "entre si. Si alguno de los dos tambien te interesa analizar como "
                    "material (para ver su curva completa), agregalo tambien en la "
                    "pestaña de Materiales -- ahi conviene dejar tildado solo "
                    "'Simplificado', porque comparar ese mismo liquido con el metodo "
                    "completo da una comparacion circular sin sentido.",
            style="Ayuda.TLabel", wraplength=800, justify="left",
        ).grid(row=fila, column=0, columnspan=8, sticky="w", pady=(2, 0))
        fila += 1

        # Cartel de advertencia de calibracion: aparece si alguno de los
        # dos patrones usa un modelo sin ecuacion de relajacion
        # (permitividad estatica constante). Aca importa mas que en un
        # material suelto, porque la permitividad del patron entra directo
        # en las formulas de conversion y afecta a TODOS los materiales.
        self.cartel_calibracion = CartelAdvertencia(f, wraplength=820)
        self.cartel_calibracion.grid(row=fila, column=0, columnspan=8,
                                      sticky="ew", pady=(10, 0))
        self.cartel_calibracion.ocultar()
        self._al_cambiar_filtrar_patrones_sg()

    def _al_cambiar_filtrar_patrones_sg(self):
        activo = bool(self.var_filtrar_patrones_sg.get())
        estado = "normal" if activo else "disabled"
        tipo_p = self.var_patrones_tipo_filtro.get() if hasattr(self, 'var_patrones_tipo_filtro') else ''

        if hasattr(self, 'combo_patrones_tipo_filtro'):
            self.combo_patrones_tipo_filtro.configure(state=("readonly" if activo else "disabled"))
        if hasattr(self, 'lbl_pat_filtro_tipo'):
            self.lbl_pat_filtro_tipo.configure(state=estado)

        for w in (self.lbl_patrones_pb_ord, self.spin_patrones_pb_ord, self.lbl_patrones_pb_fc, self.spin_patrones_pb_fc,
                  self.lbl_patrones_sg_win, self.spin_patrones_sg_win, self.lbl_patrones_sg_ord, self.spin_patrones_sg_ord,
                  self.lbl_patrones_med_win, self.spin_patrones_med_win, self.lbl_patrones_spline_s, self.spin_patrones_spline_s):
            w.pack_forget()

        if activo:
            if tipo_p == "Pasabajos (Butterworth)":
                self.lbl_patrones_pb_ord.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_pb_ord.pack(side="left", padx=(2, 6), before=self.lbl_patrones_rango)
                self.lbl_patrones_pb_fc.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_pb_fc.pack(side="left", padx=(2, 10), before=self.lbl_patrones_rango)
            elif tipo_p == "Mediana móvil":
                self.lbl_patrones_med_win.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_med_win.pack(side="left", padx=(2, 10), before=self.lbl_patrones_rango)
            elif tipo_p == "Splines de suavizado":
                self.lbl_patrones_spline_s.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_spline_s.pack(side="left", padx=(2, 10), before=self.lbl_patrones_rango)
            else:  # Savitzky-Golay
                self.lbl_patrones_sg_win.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_sg_win.pack(side="left", padx=(2, 6), before=self.lbl_patrones_rango)
                self.lbl_patrones_sg_ord.pack(side="left", before=self.lbl_patrones_rango)
                self.spin_patrones_sg_ord.pack(side="left", padx=(2, 10), before=self.lbl_patrones_rango)

        self.lbl_patrones_rango.configure(state=estado)
        self.radio_pat_rango_todo.configure(state=estado)
        self.radio_pat_rango_sub.configure(state=estado)

        es_rango = (self.var_patrones_rango.get() == "rango")
        if activo and es_rango:
            self.lbl_pat_f1.pack(side="left")
            self.spin_pat_f1.pack(side="left", padx=(2, 6))
            self.lbl_pat_f2.pack(side="left")
            self.spin_pat_f2.pack(side="left", padx=(2, 6))
        else:
            self.lbl_pat_f1.pack_forget()
            self.spin_pat_f1.pack_forget()
            self.lbl_pat_f2.pack_forget()
            self.spin_pat_f2.pack_forget()

        for w in (getattr(self, 'lbl_patrones_sg_en', None),
                  getattr(self, 'chk_sg_pat_corto', None),
                  getattr(self, 'chk_sg_pat_aire', None),
                  getattr(self, 'chk_sg_pat_p3', None),
                  getattr(self, 'chk_sg_pat_p4', None),
                  getattr(self, 'chk_sg_pat_adic', None)):
            if w is not None:
                try:
                    w.configure(state=estado)
                except (tk.TclError, AttributeError):
                    pass

    def _construir_seccion_minimos_cuadrados(self, f, fila):
        """Seccion en la pestaña 1 para activar la calibracion redundante
        por cuadrados minimos y cargar los patrones liquidos adicionales
        (etanol, butanol, acetona, ciclohexano, etc.)."""
        frm_mc = ttk.Frame(f)
        frm_mc.grid(row=fila, column=0, columnspan=8, sticky="w")
        self.var_usar_mc = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_mc,
            text="Habilitar calibracion por cuadrados minimos "
                 "(ajusta corto + aire + patron 3 + patron 4 + patrones adicionales)",
            variable=self.var_usar_mc,
            command=self._al_cambiar_usar_mc,
        ).pack(side="left")
        fila += 1

        ttk.Label(
            f,
            text="Ajusta los 4 parametros (a, b, c, Gn) minimizando el error de S11 sobre "
                 "todos los patrones a la vez, y calcula la validacion cruzada "
                 "(que pasa al sacar cada liquido de la calibracion y medirlo con el resto).",
            style="Ayuda.TLabel", wraplength=820, justify="left",
        ).grid(row=fila, column=0, columnspan=8, sticky="w", pady=(2, 4))
        fila += 1

        frm_tabla_mc = ttk.Frame(f)
        frm_tabla_mc.grid(row=fila, column=0, columnspan=8, sticky="ew")
        frm_tabla_mc.columnconfigure(0, weight=1)

        cols = ("archivo", "modelo", "temperatura")
        self.tabla_adicionales = ttk.Treeview(
            frm_tabla_mc, columns=cols, show="headings", selectmode="browse", height=3)
        self.tabla_adicionales.heading("archivo", text="Archivo .s1p del patron adicional")
        self.tabla_adicionales.heading("modelo", text="Liquido (modelo teorico)")
        self.tabla_adicionales.heading("temperatura", text="T (°C)")
        self.tabla_adicionales.column("archivo", width=360, anchor="w")
        self.tabla_adicionales.column("modelo", width=240, anchor="w")
        self.tabla_adicionales.column("temperatura", width=70, anchor="w")
        self.tabla_adicionales.grid(row=0, column=0, sticky="ew")
        self.tabla_adicionales.bind("<Double-1>", lambda _e: self._editar_adicional())
        self.tabla_adicionales.bind("<Delete>", lambda _e: self._quitar_adicional())

        scroll_mc = ttk.Scrollbar(
            frm_tabla_mc, orient="vertical", command=self.tabla_adicionales.yview)
        scroll_mc.grid(row=0, column=1, sticky="ns")
        self.tabla_adicionales.configure(yscrollcommand=scroll_mc.set)

        frm_btn_mc = ttk.Frame(frm_tabla_mc)
        frm_btn_mc.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        self.btn_agregar_adicional = ttk.Button(
            frm_btn_mc, text="Agregar...", command=self._agregar_adicional)
        self.btn_agregar_adicional.pack(fill="x", pady=(0, 3))
        self.btn_editar_adicional = ttk.Button(
            frm_btn_mc, text="Editar...", command=self._editar_adicional)
        self.btn_editar_adicional.pack(fill="x", pady=(0, 3))
        self.btn_quitar_adicional = ttk.Button(
            frm_btn_mc, text="Quitar", command=self._quitar_adicional)
        self.btn_quitar_adicional.pack(fill="x")
        self._botones_mc = [
            self.btn_agregar_adicional, self.btn_editar_adicional, self.btn_quitar_adicional]
        fila += 1
        return fila

    def _refrescar_tabla_adicionales(self):
        if not hasattr(self, 'tabla_adicionales'):
            return
        self.tabla_adicionales.delete(*self.tabla_adicionales.get_children())
        for i, p in enumerate(self._patrones_adicionales):
            modelo = p.get('modelo')
            etiqueta = gf.etiqueta_de_modelo(modelo)
            if modelo and P.advertencia_patron(modelo):
                etiqueta = f"⚠ {etiqueta}"
            temp = f"{float(p.get('temperatura_c', 25.0)):.1f}"
            self.tabla_adicionales.insert(
                "", "end", iid=str(i), values=(p.get('archivo', ''), etiqueta, temp))

    def _al_cambiar_usar_mc(self, actualizar_materiales=True):
        activo = bool(self.var_usar_mc.get())
        estado = "normal" if activo else "disabled"
        for b in getattr(self, '_botones_mc', []):
            b.configure(state=estado)
        activo_completo = bool(self.var_usar_metodo_completo.get())
        estado_est = "normal" if (activo_completo or activo) else "disabled"
        self.radio_estrategia_minimo.configure(state=estado_est)
        self.radio_estrategia_max.configure(state=estado_est)
        self.radio_estrategia_comparar.configure(state=estado_est)
        self._al_cambiar_acondicionar_gn()
        self._refrescar_cartel_calibracion()
        if actualizar_materiales and hasattr(self, '_materiales'):
            modificados = False
            for m in self._materiales:
                metodos = m.setdefault('metodos', ['simplificado'])
                if activo and 'minimos_cuadrados' not in metodos:
                    metodos.append('minimos_cuadrados')
                    modificados = True
                elif not activo and 'minimos_cuadrados' in metodos and len(metodos) > 1:
                    metodos.remove('minimos_cuadrados')
                    modificados = True
            if modificados and hasattr(self, 'tabla_materiales'):
                self._refrescar_tabla_materiales()

    def _indice_adicional_seleccionado(self):
        sel = self.tabla_adicionales.selection()
        return int(sel[0]) if sel else None

    def _agregar_adicional(self):
        dialogo = DialogoPatronAdicional(self)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._patrones_adicionales.append(dialogo.resultado)
            self._refrescar_tabla_adicionales()
            self._refrescar_cartel_calibracion()

    def _editar_adicional(self):
        if not self.var_usar_mc.get():
            return
        idx = self._indice_adicional_seleccionado()
        if idx is None:
            messagebox.showinfo(
                "Editar patron adicional", "Elegi primero una fila de la tabla.", parent=self)
            return
        dialogo = DialogoPatronAdicional(self, patron=copy.deepcopy(self._patrones_adicionales[idx]))
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._patrones_adicionales[idx] = dialogo.resultado
            self._refrescar_tabla_adicionales()
            self._refrescar_cartel_calibracion()

    def _quitar_adicional(self):
        if not self.var_usar_mc.get():
            return
        idx = self._indice_adicional_seleccionado()
        if idx is None:
            return
        del self._patrones_adicionales[idx]
        self._refrescar_tabla_adicionales()
        self._refrescar_cartel_calibracion()

    def _refrescar_cartel_calibracion(self):
        """Muestra (o esconde) el cartel de advertencia de la pestaña de
        Calibracion segun los liquidos elegidos como patron 3, patron 4
        y patrones adicionales de cuadrados minimos.

        Muestra el resumen corto de cada modelo que tenga advertencia
        (ver Patrones.INFO_PATRONES). Importa mas aca que en un material
        suelto, porque la permitividad del patron entra directo en las
        formulas de conversion y afecta a TODOS los materiales."""
        if not hasattr(self, 'cartel_calibracion'):
            return

        claves = [("Patr\u00f3n 3", gf.clave_de_etiqueta(
            self.vars_modelo_calibracion['patron3'].get()))]
        if self.var_usar_metodo_completo.get() or (
                hasattr(self, 'var_usar_mc') and self.var_usar_mc.get()):
            claves.append(("Patr\u00f3n 4", gf.clave_de_etiqueta(
                self.vars_modelo_calibracion['patron4'].get())))
        if hasattr(self, 'var_usar_mc') and self.var_usar_mc.get():
            for i, p in enumerate(self._patrones_adicionales, start=1):
                claves.append((f"Patr\u00f3n adicional #{i}", p.get('modelo')))

        partes = []
        nivel = 'info'
        for rol, clave in claves:
            if not clave or not P.advertencia_patron(clave):
                continue
            etiqueta = gf.etiqueta_de_modelo(clave)
            resumen = P.resumen_corto_patron(clave) or "ver advertencia del modelo"
            partes.append(f"\u2022 {rol} ({etiqueta}): {resumen}.")
            if P.es_modelo_estatico(clave):
                partes.append(
                    "   Al ser un l\u00edquido no polar, su permitividad (~2) est\u00e1 "
                    "cerca de la del aire: como 3er patr\u00f3n de una calibraci\u00f3n "
                    "de solo 3 queda mal condicionado. Kaatze (2007) s\u00ed lo "
                    "recomienda como patr\u00f3n ADICIONAL, porque para medir bien "
                    "muestras de baja permitividad hace falta un patr\u00f3n de baja "
                    "permitividad.")
            if P.nivel_advertencia_patron(clave) == 'critico':
                nivel = 'critico'
            elif nivel != 'critico' and P.nivel_advertencia_patron(clave) == 'aviso':
                nivel = 'aviso'

        if not partes:
            self.cartel_calibracion.ocultar()
            return
        partes.append(
            "La permitividad del patr\u00f3n entra directo en las f\u00f3rmulas de "
            "conversi\u00f3n S11 \u2192 er, as\u00ed que cualquier limitaci\u00f3n del modelo "
            "se propaga al resultado de TODOS los materiales.")
        self.cartel_calibracion.mostrar(
            "Patr\u00f3n de calibraci\u00f3n con advertencias",
            "\n".join(partes), nivel=nivel)

    def _al_cambiar_usar_metodo_completo(self):
        activo = bool(self.var_usar_metodo_completo.get())
        activo_mc = bool(getattr(self, 'var_usar_mc', None) and self.var_usar_mc.get())
        estado = "normal" if activo else "disabled"
        for widget in self._widgets_patron4:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state=("readonly" if activo else "disabled"))
                else:
                    widget.configure(state=estado)
            except tk.TclError:
                pass
        estado_est = "normal" if (activo or activo_mc) else "disabled"
        self.radio_estrategia_minimo.configure(state=estado_est)
        self.radio_estrategia_max.configure(state=estado_est)
        self.radio_estrategia_comparar.configure(state=estado_est)
        self._al_cambiar_acondicionar_gn()
        # El patron 4 deja de contar (o vuelve a contar) para el cartel de
        # advertencia de calibracion segun si el metodo completo esta on/off.
        self._refrescar_cartel_calibracion()

    def _al_cambiar_acondicionar_gn(self):
        activo_completo = bool(self.var_usar_metodo_completo.get())
        activo_mc = bool(getattr(self, 'var_usar_mc', None) and self.var_usar_mc.get())
        activo_alguno = activo_completo or activo_mc
        activo_acond = bool(self.var_acondicionar_gn.get()) and activo_alguno

        estado_chk = "normal" if activo_alguno else "disabled"
        self.chk_acondicionar_gn.configure(state=estado_chk)

        estado = "normal" if activo_acond else "disabled"
        self.combo_acond_tipo.configure(state=("readonly" if activo_acond else "disabled"))
        self.chk_comparar_gn_original.configure(state=estado)

        es_auto = (self.var_acond_modo.get() == "automatico")
        tipo_str = self.var_acond_tipo.get()
        tipo_clave = _TIPOS_ACOND_MAP.get(tipo_str, "transicion_polinomica_cero")

        usa_pb = (tipo_clave in ('pasabajos', 'transicion_pasabajos'))
        usa_sg = (tipo_clave in ('savitzky_golay', 'transicion_savgol'))
        usa_med = (tipo_clave == 'mediana_movil')
        usa_spl = (tipo_clave == 'splines')
        usa_freq = (tipo_clave in ('cero_clamping', 'transicion_polinomica_cero', 'tapering',
                                  'transicion_savgol', 'transicion_pasabajos',
                                  'pasabajos', 'savitzky_golay', 'mediana_movil', 'splines'))

        # Radiobutton modo auto/manual
        if activo_acond and usa_freq:
            self.lbl_acond_modo.configure(state="normal")
            self.radio_acond_auto.configure(state="normal")
            self.radio_acond_manual.configure(state="normal")
        else:
            self.lbl_acond_modo.configure(state="disabled")
            self.radio_acond_auto.configure(state="disabled")
            self.radio_acond_manual.configure(state="disabled")

        # Controles pasabajos
        if hasattr(self, 'lbl_acond_pb_orden'):
            if activo_acond and usa_pb:
                self.lbl_acond_pb_orden.pack(side="left")
                self.spin_acond_pb_orden.pack(side="left", padx=(4, 6))
                self.spin_acond_pb_orden.configure(state="normal")
                self.lbl_acond_pb_fcorte.pack(side="left")
                self.spin_acond_pb_fcorte.pack(side="left", padx=(4, 10))
                self.spin_acond_pb_fcorte.configure(state="normal")
            else:
                self.lbl_acond_pb_orden.pack_forget()
                self.spin_acond_pb_orden.pack_forget()
                self.lbl_acond_pb_fcorte.pack_forget()
                self.spin_acond_pb_fcorte.pack_forget()

        # Controles Savitzky-Golay
        if hasattr(self, 'lbl_acond_sg_window'):
            if activo_acond and usa_sg:
                self.lbl_acond_sg_window.pack(side="left")
                self.spin_acond_sg_window.pack(side="left", padx=(4, 6))
                self.spin_acond_sg_window.configure(state="normal")
                self.lbl_acond_sg_polyorder.pack(side="left")
                self.spin_acond_sg_polyorder.pack(side="left", padx=(4, 10))
                self.spin_acond_sg_polyorder.configure(state="normal")
            else:
                self.lbl_acond_sg_window.pack_forget()
                self.spin_acond_sg_window.pack_forget()
                self.lbl_acond_sg_polyorder.pack_forget()
                self.spin_acond_sg_polyorder.pack_forget()

        # Controles Mediana móvil
        if hasattr(self, 'lbl_acond_med_window'):
            if activo_acond and usa_med:
                self.lbl_acond_med_window.pack(side="left")
                self.spin_acond_med_window.pack(side="left", padx=(4, 10))
                self.spin_acond_med_window.configure(state="normal")
            else:
                self.lbl_acond_med_window.pack_forget()
                self.spin_acond_med_window.pack_forget()

        # Controles Splines
        if hasattr(self, 'lbl_acond_spline_s'):
            if activo_acond and usa_spl:
                self.lbl_acond_spline_s.pack(side="left")
                self.spin_acond_spline_s.pack(side="left", padx=(4, 10))
                self.spin_acond_spline_s.configure(state="normal")
            else:
                self.lbl_acond_spline_s.pack_forget()
                self.spin_acond_spline_s.pack_forget()

        # Controles de frecuencia y umbral
        if not (activo_acond and usa_freq):
            self.lbl_acond_umbral.pack_forget()
            self.spin_acond_umbral.pack_forget()
            self.lbl_acond_f1.pack_forget()
            self.spin_acond_f1.pack_forget()
            self.lbl_acond_f2.pack_forget()
            self.spin_acond_f2.pack_forget()
        elif es_auto:
            self.lbl_acond_f1.pack_forget()
            self.spin_acond_f1.pack_forget()
            self.lbl_acond_f2.pack_forget()
            self.spin_acond_f2.pack_forget()
            self.lbl_acond_umbral.pack(side="left")
            self.spin_acond_umbral.pack(side="left", padx=(4, 10))
            self.lbl_acond_umbral.configure(state="normal")
            self.spin_acond_umbral.configure(state="normal")
        else:
            self.lbl_acond_umbral.pack_forget()
            self.spin_acond_umbral.pack_forget()
            if tipo_clave == 'cero_clamping':
                self.lbl_acond_f1.pack_forget()
                self.spin_acond_f1.pack_forget()
                self.lbl_acond_f2.configure(text="f corte (GHz):", state="normal")
                self.lbl_acond_f2.pack(side="left")
                self.spin_acond_f2.pack(side="left", padx=(4, 10))
                self.spin_acond_f2.configure(state="normal")
            else:
                self.lbl_acond_f1.configure(text="f1 (GHz):", state="normal")
                self.lbl_acond_f1.pack(side="left")
                self.spin_acond_f1.pack(side="left", padx=(4, 6))
                self.spin_acond_f1.configure(state="normal")
                self.lbl_acond_f2.configure(text="f2 (GHz):", state="normal")
                self.lbl_acond_f2.pack(side="left")
                self.spin_acond_f2.pack(side="left", padx=(4, 10))
                self.spin_acond_f2.configure(state="normal")

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

        columnas = ("nombre", "archivo", "modelo", "temperatura", "filtro", "metodos")
        self.tabla_materiales = ttk.Treeview(
            frm_tabla, columns=columnas, show="headings", selectmode="extended", height=12)
        titulos = {"nombre": "Nombre", "archivo": "Archivo", "modelo": "Modelo teorico",
                   "temperatura": "T (°C)", "filtro": "Filtro S11", "metodos": "Metodos"}
        anchos = {"nombre": 140, "archivo": 240, "modelo": 160, "temperatura": 65, "filtro": 130, "metodos": 140}
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
        ttk.Button(frm_botones, text="Agregar carpeta...",
                   command=self._agregar_carpeta_materiales).pack(side="left", padx=6)
        ttk.Button(frm_botones, text="Duplicar", command=self._duplicar_material).pack(side="left")
        ttk.Button(frm_botones, text="Editar...", command=self._editar_material).pack(side="left", padx=6)
        ttk.Button(frm_botones, text="Quitar", command=self._quitar_material).pack(side="left")
        ttk.Separator(frm_botones, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(frm_botones, text="Subir", command=lambda: self._mover_material(-1)).pack(side="left")
        ttk.Button(frm_botones, text="Bajar", command=lambda: self._mover_material(1)).pack(side="left", padx=6)

        # --- Configuración en lote de filtro S11 para materiales ---
        lf_filtro_lote = ttk.LabelFrame(f, text="Filtrado de S11 para materiales (configuración en lote)", padding=8)
        lf_filtro_lote.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        frm_fl_1 = ttk.Frame(lf_filtro_lote)
        frm_fl_1.pack(anchor="w")

        ttk.Label(frm_fl_1, text="Tipo de filtro:").pack(side="left")
        self.var_lote_mat_tipo = tk.StringVar(value="Pasabajos (Butterworth)")
        self.combo_lote_mat_tipo = ttk.Combobox(
            frm_fl_1, textvariable=self.var_lote_mat_tipo, state="readonly", width=22,
            values=["Pasabajos (Butterworth)", "Savitzky-Golay", "Mediana móvil", "Splines de suavizado"])
        self.combo_lote_mat_tipo.pack(side="left", padx=(4, 10))
        self.combo_lote_mat_tipo.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_filtro_lote_mat())

        # Parámetros Pasabajos
        self.lbl_lote_pb_ord = ttk.Label(frm_fl_1, text="Orden PB:")
        self.lbl_lote_pb_ord.pack(side="left")
        self.var_lote_pb_ord = tk.IntVar(value=2)
        self.spin_lote_pb_ord = ttk.Spinbox(
            frm_fl_1, from_=1, to=5, increment=1, textvariable=self.var_lote_pb_ord, width=3)
        self.spin_lote_pb_ord.pack(side="left", padx=(2, 6))

        self.lbl_lote_pb_fc = ttk.Label(frm_fl_1, text="Corte PB (fc/fn):")
        self.lbl_lote_pb_fc.pack(side="left")
        self.var_lote_pb_fc = tk.DoubleVar(value=0.10)
        self.spin_lote_pb_fc = ttk.Spinbox(
            frm_fl_1, from_=0.01, to=0.50, increment=0.01, textvariable=self.var_lote_pb_fc, width=5)
        self.spin_lote_pb_fc.pack(side="left", padx=(2, 10))

        # Parámetros Savitzky-Golay
        self.lbl_lote_sg_win = ttk.Label(frm_fl_1, text="Ventana SG:")
        self.lbl_lote_sg_win.pack(side="left")
        self.var_lote_sg_win = tk.IntVar(value=15)
        self.spin_lote_sg_win = ttk.Spinbox(
            frm_fl_1, from_=3, to=101, increment=2, textvariable=self.var_lote_sg_win, width=4)
        self.spin_lote_sg_win.pack(side="left", padx=(2, 6))

        self.lbl_lote_sg_ord = ttk.Label(frm_fl_1, text="Orden SG:")
        self.lbl_lote_sg_ord.pack(side="left")
        self.var_lote_sg_ord = tk.IntVar(value=3)
        self.spin_lote_sg_ord = ttk.Spinbox(
            frm_fl_1, from_=1, to=5, increment=1, textvariable=self.var_lote_sg_ord, width=3)
        self.spin_lote_sg_ord.pack(side="left", padx=(2, 10))

        # Parametros Mediana móvil lote
        self.lbl_lote_med_win = ttk.Label(frm_fl_1, text="Ventana Med:")
        self.var_lote_med_win = tk.IntVar(value=15)
        self.spin_lote_med_win = ttk.Spinbox(
            frm_fl_1, from_=3, to=101, increment=2, textvariable=self.var_lote_med_win, width=4)

        # Parametros Splines lote
        self.lbl_lote_spline_s = ttk.Label(frm_fl_1, text="Factor s:")
        self.var_lote_spline_s = tk.DoubleVar(value=0.10)
        self.spin_lote_spline_s = ttk.Spinbox(
            frm_fl_1, from_=0.0001, to=50.0, increment=0.01, textvariable=self.var_lote_spline_s, width=6)

        # Rango lote
        self.lbl_lote_rango = ttk.Label(frm_fl_1, text="  Rango:")
        self.lbl_lote_rango.pack(side="left")
        self.var_lote_rango = tk.StringVar(value="todo")
        self.radio_lote_rango_todo = ttk.Radiobutton(
            frm_fl_1, text="Todo", value="todo", variable=self.var_lote_rango,
            command=self._al_cambiar_filtro_lote_mat)
        self.radio_lote_rango_todo.pack(side="left", padx=(2, 4))
        self.radio_lote_rango_sub = ttk.Radiobutton(
            frm_fl_1, text="[f1, f2]:", value="rango", variable=self.var_lote_rango,
            command=self._al_cambiar_filtro_lote_mat)
        self.radio_lote_rango_sub.pack(side="left", padx=(2, 6))

        self.lbl_lote_f1 = ttk.Label(frm_fl_1, text="f1 (GHz):")
        self.var_lote_f1 = tk.DoubleVar(value=0.5)
        self.spin_lote_f1 = ttk.Spinbox(
            frm_fl_1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_lote_f1, width=6)
        self.lbl_lote_f2 = ttk.Label(frm_fl_1, text="f2 (GHz):")
        self.var_lote_f2 = tk.DoubleVar(value=6.0)
        self.spin_lote_f2 = ttk.Spinbox(
            frm_fl_1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_lote_f2, width=6)

        self.btn_copiar_patrones_lote = ttk.Button(frm_fl_1, text="Copiar de patrones", command=self._copiar_filtro_patrones_a_lote)
        self.btn_copiar_patrones_lote.pack(side="left", padx=(4, 0))
        _tooltip(frm_fl_1, "Configurá los parámetros del filtro S11 y aplicalos a todos los materiales cargados o solo a los seleccionados.")

        frm_fl_2 = ttk.Frame(lf_filtro_lote)
        frm_fl_2.pack(anchor="w", pady=(6, 0))

        ttk.Button(frm_fl_2, text="Aplicar a todos", command=self._aplicar_filtro_lote_todos).pack(side="left")
        ttk.Button(frm_fl_2, text="Aplicar a seleccionados", command=self._aplicar_filtro_lote_seleccionados).pack(side="left", padx=6)
        ttk.Separator(frm_fl_2, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(frm_fl_2, text="Quitar filtro de todos", command=self._quitar_filtro_lote_todos).pack(side="left")
        ttk.Button(frm_fl_2, text="Quitar filtro de seleccionados", command=self._quitar_filtro_lote_seleccionados).pack(side="left", padx=6)

        self._al_cambiar_filtro_lote_mat()

    def _refrescar_tabla_materiales(self):
        self.tabla_materiales.delete(*self.tabla_materiales.get_children())
        for i, m in enumerate(self._materiales):
            metodos = "+".join(m.get('metodos', []))
            temp = f"{m['temperatura']:.1f}" if m.get('modelo') else "-"
            # Los modelos que tienen alguna advertencia declarada en
            # Patrones.INFO_PATRONES (acetona, ciclohexano, silicona: los
            # tres que no salen de las tablas de relajacion del MAT 23) se
            # marcan con un triangulo, para que se note de un vistazo
            # cuales materiales se estan comparando contra una referencia
            # con limitaciones. El detalle completo esta en el cartel del
            # dialogo de edicion y en el informe PDF.
            etiqueta_modelo = gf.etiqueta_de_modelo(m.get('modelo'))
            if m.get('modelo') and P.advertencia_patron(m['modelo']):
                etiqueta_modelo = f"⚠ {etiqueta_modelo}"
            filtrar_s11 = bool(m.get('filtrar_s11') or m.get('filtrar_sg', False))
            tipo_f = m.get('tipo_filtro', 'savitzky_golay')
            if filtrar_s11:
                if tipo_f == 'pasabajos':
                    ord_pb = m.get('pb_orden', 2) or 2
                    fc_pb = m.get('pb_fcorte', 0.10) or 0.10
                    col_filtro = f"Pasabajos ({ord_pb}, fc={fc_pb:.2f})"
                elif tipo_f == 'mediana_movil':
                    w_med = m.get('med_window', 15) or 15
                    col_filtro = f"Mediana ({w_med})"
                elif tipo_f == 'splines':
                    s_spl = m.get('spline_s', 0.10) or 0.10
                    col_filtro = f"Splines (s={s_spl:.3f})"
                else:
                    w = m.get('sg_window', 15) or 15
                    p = m.get('sg_polyorder', 3) or 3
                    col_filtro = f"Savitzky-Golay ({w}, {p})"
                if m.get('modo_rango') == 'rango':
                    f1_m = m.get('f1_ghz', 0.5) or 0.5
                    f2_m = m.get('f2_ghz', 6.0) or 6.0
                    col_filtro += f" [{f1_m:.2f}-{f2_m:.2f} GHz]"
            else:
                col_filtro = "—"
            self.tabla_materiales.insert(
                "", "end", iid=str(i),
                values=(m['nombre'], m['archivo'], etiqueta_modelo, temp, col_filtro, metodos))

    def _indices_seleccionados(self):
        return [int(x) for x in self.tabla_materiales.selection()]

    def _indice_seleccionado(self):
        sel = self._indices_seleccionados()
        return sel[0] if sel else None

    def _al_cambiar_filtro_lote_mat(self):
        tipo_l = self.var_lote_mat_tipo.get()
        for w in (self.lbl_lote_pb_ord, self.spin_lote_pb_ord, self.lbl_lote_pb_fc, self.spin_lote_pb_fc,
                  self.lbl_lote_sg_win, self.spin_lote_sg_win, self.lbl_lote_sg_ord, self.spin_lote_sg_ord,
                  self.lbl_lote_med_win, self.spin_lote_med_win, self.lbl_lote_spline_s, self.spin_lote_spline_s):
            w.pack_forget()

        before_w = getattr(self, 'lbl_lote_rango', None)
        if tipo_l == "Pasabajos (Butterworth)":
            self.lbl_lote_pb_ord.pack(side="left", before=before_w)
            self.spin_lote_pb_ord.pack(side="left", padx=(2, 6), before=before_w)
            self.lbl_lote_pb_fc.pack(side="left", before=before_w)
            self.spin_lote_pb_fc.pack(side="left", padx=(2, 10), before=before_w)
        elif tipo_l == "Mediana móvil":
            self.lbl_lote_med_win.pack(side="left", before=before_w)
            self.spin_lote_med_win.pack(side="left", padx=(2, 10), before=before_w)
        elif tipo_l == "Splines de suavizado":
            self.lbl_lote_spline_s.pack(side="left", before=before_w)
            self.spin_lote_spline_s.pack(side="left", padx=(2, 10), before=before_w)
        else:  # Savitzky-Golay
            self.lbl_lote_sg_win.pack(side="left", before=before_w)
            self.spin_lote_sg_win.pack(side="left", padx=(2, 6), before=before_w)
            self.lbl_lote_sg_ord.pack(side="left", before=before_w)
            self.spin_lote_sg_ord.pack(side="left", padx=(2, 10), before=before_w)

        es_rango = (self.var_lote_rango.get() == "rango")
        btn_copiar = getattr(self, 'btn_copiar_patrones_lote', None)
        if es_rango:
            self.lbl_lote_f1.pack(side="left", before=btn_copiar)
            self.spin_lote_f1.pack(side="left", padx=(2, 6), before=btn_copiar)
            self.lbl_lote_f2.pack(side="left", before=btn_copiar)
            self.spin_lote_f2.pack(side="left", padx=(2, 6), before=btn_copiar)
        else:
            self.lbl_lote_f1.pack_forget()
            self.spin_lote_f1.pack_forget()
            self.lbl_lote_f2.pack_forget()
            self.spin_lote_f2.pack_forget()

    def _copiar_filtro_patrones_a_lote(self):
        if hasattr(self, 'var_patrones_tipo_filtro'):
            self.var_lote_mat_tipo.set(self.var_patrones_tipo_filtro.get())
        if hasattr(self, 'var_patrones_pb_ord'):
            self.var_lote_pb_ord.set(self.var_patrones_pb_ord.get())
        if hasattr(self, 'var_patrones_pb_fc'):
            self.var_lote_pb_fc.set(self.var_patrones_pb_fc.get())
        if hasattr(self, 'var_patrones_sg_win'):
            self.var_lote_sg_win.set(self.var_patrones_sg_win.get())
        if hasattr(self, 'var_patrones_sg_ord'):
            self.var_lote_sg_ord.set(self.var_patrones_sg_ord.get())
        if hasattr(self, 'var_patrones_med_win'):
            self.var_lote_med_win.set(self.var_patrones_med_win.get())
        if hasattr(self, 'var_patrones_spline_s'):
            self.var_lote_spline_s.set(self.var_patrones_spline_s.get())
        if hasattr(self, 'var_patrones_rango'):
            self.var_lote_rango.set(self.var_patrones_rango.get())
        if hasattr(self, 'var_pat_f1'):
            self.var_lote_f1.set(self.var_pat_f1.get())
        if hasattr(self, 'var_pat_f2'):
            self.var_lote_f2.set(self.var_pat_f2.get())
        self._al_cambiar_filtro_lote_mat()
        self._status("Parámetros de filtro copiados desde la pestaña de Patrones.")

    def _aplicar_filtro_a_materiales(self, indices):
        if not indices:
            messagebox.showinfo("Filtrado de materiales", "No hay materiales seleccionados para aplicar el filtro.", parent=self)
            return
        tipo_map = {
            "Pasabajos (Butterworth)": "pasabajos",
            "Savitzky-Golay": "savitzky_golay",
            "Mediana móvil": "mediana_movil",
            "Splines de suavizado": "splines",
        }
        tipo_f = tipo_map.get(self.var_lote_mat_tipo.get(), 'savitzky_golay')
        pb_ord = int(self.var_lote_pb_ord.get() or 2)
        pb_fc = float(self.var_lote_pb_fc.get() or 0.10)
        sg_win = int(self.var_lote_sg_win.get() or 15)
        sg_ord = int(self.var_lote_sg_ord.get() or 3)
        med_win = int(self.var_lote_med_win.get() or 15)
        spline_s = float(self.var_lote_spline_s.get() or 0.10)
        modo_rango = self.var_lote_rango.get()
        f1_ghz = float(self.var_lote_f1.get() or 0.5)
        f2_ghz = float(self.var_lote_f2.get() or 6.0)

        for idx in indices:
            m = self._materiales[idx]
            m['filtrar_s11'] = True
            m['filtrar_sg'] = (tipo_f == 'savitzky_golay')
            m['tipo_filtro'] = tipo_f
            m['pb_orden'] = pb_ord
            m['pb_fcorte'] = pb_fc
            m['sg_window'] = sg_win
            m['sg_polyorder'] = sg_ord
            m['med_window'] = med_win
            m['spline_s'] = spline_s
            m['modo_rango'] = modo_rango
            m['f1_ghz'] = f1_ghz
            m['f2_ghz'] = f2_ghz

        self._refrescar_tabla_materiales()
        for idx in indices:
            if idx < len(self._materiales):
                self.tabla_materiales.selection_add(str(idx))
        self._status(f"Filtro '{tipo_f}' aplicado a {len(indices)} material(es).")

    def _aplicar_filtro_lote_todos(self):
        if not self._materiales:
            messagebox.showinfo("Filtrado de materiales", "No hay materiales cargados en la lista.", parent=self)
            return
        self._aplicar_filtro_a_materiales(list(range(len(self._materiales))))

    def _aplicar_filtro_lote_seleccionados(self):
        idxs = self._indices_seleccionados()
        if not idxs:
            messagebox.showinfo("Filtrado de materiales", "Seleccioná al menos un material de la tabla.", parent=self)
            return
        self._aplicar_filtro_a_materiales(idxs)

    def _quitar_filtro_a_materiales(self, indices):
        if not indices:
            messagebox.showinfo("Quitar filtro", "No hay materiales seleccionados.", parent=self)
            return
        for idx in indices:
            m = self._materiales[idx]
            m['filtrar_s11'] = False
            m['filtrar_sg'] = False
        self._refrescar_tabla_materiales()
        for idx in indices:
            if idx < len(self._materiales):
                self.tabla_materiales.selection_add(str(idx))
        self._status(f"Filtro quitado de {len(indices)} material(es).")

    def _quitar_filtro_lote_todos(self):
        if not self._materiales:
            return
        self._quitar_filtro_a_materiales(list(range(len(self._materiales))))

    def _quitar_filtro_lote_seleccionados(self):
        idxs = self._indices_seleccionados()
        if not idxs:
            messagebox.showinfo("Quitar filtro", "Seleccioná al menos un material de la tabla.", parent=self)
            return
        self._quitar_filtro_a_materiales(idxs)

    def _agregar_material(self):
        dialogo = DialogoMaterial(self)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._materiales.append(dialogo.resultado)
            self._refrescar_tabla_materiales()

    def _agregar_carpeta_materiales(self):
        """Escanea una carpeta elegida por el usuario y agrega un
        material por cada .s1p encontrado (nombre = nombre de archivo sin
        extension, sin modelo teorico asignado -- se puede editar despues
        con 'Editar...'). Util para cargar de una sola vez todas las
        muestras de una sesion de laboratorio en vez de repetir el
        dialogo 'Agregar...' una por una. No duplica archivos que ya
        estuvieran cargados (comparando la ruta completa resuelta)."""
        carpeta = filedialog.askdirectory(
            title="Elegir carpeta con archivos .s1p", parent=self)
        if not carpeta:
            return

        try:
            archivos = sorted(
                nombre for nombre in os.listdir(carpeta)
                if nombre.lower().endswith('.s1p')
                and os.path.isfile(os.path.join(carpeta, nombre)))
        except OSError as exc:
            messagebox.showerror(
                "Error al leer la carpeta", f"No se pudo leer '{carpeta}':\n{exc}", parent=self)
            return

        if not archivos:
            messagebox.showinfo(
                "Agregar carpeta", f"No se encontraron archivos .s1p en:\n{carpeta}", parent=self)
            return

        carpeta_base = self.var_carpeta_datos.get().strip()
        ya_cargados = {
            os.path.normcase(os.path.normpath(os.path.join(carpeta_base, m['archivo'])))
            for m in self._materiales
        }
        metodos_default = (['simplificado', 'completo']
                            if self.var_usar_metodo_completo.get() else ['simplificado'])
        if hasattr(self, 'var_usar_mc') and self.var_usar_mc.get():
            metodos_default.append('minimos_cuadrados')

        agregados = omitidos = 0
        for nombre_archivo in archivos:
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            clave = os.path.normcase(os.path.normpath(ruta_completa))
            if clave in ya_cargados:
                omitidos += 1
                continue
            self._materiales.append({
                'nombre': os.path.splitext(nombre_archivo)[0],
                'archivo': ruta_completa,
                'modelo': gf.SIN_MODELO,
                'temperatura': 25.0,
                'metodos': list(metodos_default),
            })
            ya_cargados.add(clave)
            agregados += 1

        self._refrescar_tabla_materiales()
        mensaje = f"Se agregaron {agregados} material(es) desde la carpeta."
        if omitidos:
            mensaje += f" ({omitidos} ya estaban cargados, se omitieron.)"
        self._status(mensaje)
        if agregados:
            messagebox.showinfo("Agregar carpeta", mensaje, parent=self)

    def _duplicar_material(self):
        idx = self._indice_seleccionado()
        if idx is None:
            messagebox.showinfo("Duplicar material", "Elegi primero una fila de la tabla.", parent=self)
            return
        clon = copy.deepcopy(self._materiales[idx])
        clon['nombre'] = f"{clon['nombre']} (copia)"
        dialogo = DialogoMaterial(self, material=clon)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._materiales.insert(idx + 1, dialogo.resultado)
            self._refrescar_tabla_materiales()
            self.tabla_materiales.selection_set(str(idx + 1))

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
        idxs = self._indices_seleccionados()
        if not idxs:
            return
        if len(idxs) == 1:
            nombre = self._materiales[idxs[0]]['nombre']
            if not messagebox.askyesno("Quitar material", f"¿Quitar '{nombre}' de la lista?", parent=self):
                return
        else:
            if not messagebox.askyesno("Quitar materiales", f"¿Quitar los {len(idxs)} materiales seleccionados de la lista?", parent=self):
                return
        for idx in sorted(idxs, reverse=True):
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
                    "correr el analisis completo para ver esto. Los graficos son "
                    "interactivos: rueda del mouse o el icono de la lupa para hacer "
                    "zoom, arrastrar para mover la vista, la casita para volver al "
                    "estado original, y la posicion del cursor se muestra abajo a la "
                    "derecha de cada grafico.",
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

        self.frm_preview_s11_magfase = ttk.LabelFrame(f, text="S11 medido (modulo y fase)")
        self.frm_preview_s11_magfase.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        self.frm_preview_s11_smith = ttk.LabelFrame(f, text="S11 medido (diagrama de Smith)")
        self.frm_preview_s11_smith.grid(row=3, column=1, sticky="nsew", padx=(6, 0))

        self.visor_preview_s11_magfase = VisorFigura(self.frm_preview_s11_magfase)
        self.visor_preview_s11_magfase.pack(fill="both", expand=True, padx=4, pady=4)
        self.visor_preview_s11_smith = VisorFigura(self.frm_preview_s11_smith)
        self.visor_preview_s11_smith.pack(fill="both", expand=True, padx=4, pady=4)

        self.var_preview_s11_status = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_preview_s11_status,
                  style="Ayuda.TLabel", wraplength=920, justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.var_preview_s11_acond = tk.StringVar(value="")
        self.lbl_preview_s11_acond = ttk.Label(
            f, textvariable=self.var_preview_s11_acond,
            font=("Segoe UI", 9, "bold"), foreground="#1b4f72",
            wraplength=920, justify="left")
        self.lbl_preview_s11_acond.grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _refrescar_lista_preview_s11(self):
        """Reconstruye la lista de 'que previsualizar' a partir de lo que
        haya cargado AHORA MISMO en las pestañas de Calibracion y
        Materiales. Se llama sola al entrar a esta pestaña (ver
        _al_cambiar_tab_notebook) y tambien desde el boton 'Refrescar
        lista', por si se prefiere refrescar sin cambiar de pestaña."""
        carpeta = self.var_carpeta_datos.get().strip()
        rutas = {}

        self._rutas_preview_s11 = rutas
        self._materiales_preview_s11 = {}
        self._patrones_preview_s11 = {}

        for clave in ('corto', 'aire', 'patron3', 'patron4'):
            archivo = self.vars_calibracion[clave].get().strip()
            if not archivo:
                continue
            if clave in _ETIQUETAS_CALIBRACION_PREVIEW:
                nombre_liquido = _ETIQUETAS_CALIBRACION_PREVIEW[clave]
            else:
                nombre_generico = _NOMBRES_PATRON_CALIBRACION.get(clave, clave)
                modelo_etiqueta = self.vars_modelo_calibracion[clave].get()
                nombre_liquido = (f"{nombre_generico} ({modelo_etiqueta})"
                                   if modelo_etiqueta else nombre_generico)
            etiqueta = f"[Calibracion] {nombre_liquido}"
            rutas[etiqueta] = os.path.join(carpeta, archivo)
            self._patrones_preview_s11[etiqueta] = clave

        for i, p in enumerate(self._patrones_adicionales, start=1):
            archivo = (p.get('archivo') or "").strip()
            if not archivo:
                continue
            et_mod = gf.etiqueta_de_modelo(p.get('modelo'))
            etiqueta = f"[Calibracion - adicional #{i}] {et_mod}"
            rutas[etiqueta] = os.path.join(carpeta, archivo)
            self._patrones_preview_s11[etiqueta] = 'adicionales'

        # Si estan cargados los 4 archivos de calibracion, ofrecer vista previa de Gn(f):
        cal_completos = all(
            bool(self.vars_calibracion[k].get().strip())
            for k in ('corto', 'aire', 'patron3', 'patron4')
        )
        if cal_completos:
            rutas["[Calibracion] Conductancia normalizada Gn(f)"] = "__gn_preview__"

        for m in self._materiales:
            archivo = (m.get('archivo') or "").strip()
            if archivo:
                etiqueta = f"[Material] {m.get('nombre') or archivo}"
                rutas[etiqueta] = os.path.join(carpeta, archivo)
                self._materiales_preview_s11[etiqueta] = m

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
        if hasattr(self, 'frm_preview_s11_magfase'):
            self.frm_preview_s11_magfase.configure(text="S11 medido (modulo y fase)")
            self.frm_preview_s11_smith.configure(text="S11 medido (diagrama de Smith)")
        self.visor_preview_s11_magfase.limpiar(mensaje)
        self.visor_preview_s11_smith.limpiar(mensaje)
        self.var_preview_s11_status.set(mensaje)
        if hasattr(self, 'var_preview_s11_acond'):
            self.var_preview_s11_acond.set("")

    def _mostrar_preview_s11_seleccionado(self):
        etiqueta = self.var_preview_s11_seleccion.get()
        ruta = self._rutas_preview_s11.get(etiqueta)
        if not ruta:
            return

        if ruta == "__gn_preview__":
            self._mostrar_preview_gn()
            return

        if hasattr(self, 'var_preview_s11_acond'):
            self.var_preview_s11_acond.set("")
        self.frm_preview_s11_magfase.configure(text="S11 medido (modulo y fase)")
        self.frm_preview_s11_smith.configure(text="S11 medido (diagrama de Smith)")

        if not os.path.isfile(ruta):
            self._limpiar_preview_s11(f"No se encontro el archivo de '{etiqueta}':\n{ruta}")
            return

        mat_info = getattr(self, '_materiales_preview_s11', {}).get(etiqueta)
        clave_patron = getattr(self, '_patrones_preview_s11', {}).get(etiqueta)

        filtrar_s11 = False
        tipo_filtro = "savitzky_golay"
        sg_w = 15
        sg_p = 3
        pb_ord = 2
        pb_fc = 0.10
        med_w = 15
        spl_s = 0.10
        m_rango = "todo"
        f1_g = 0.5
        f2_g = 6.0
        tipo_item = "elemento"

        if mat_info is not None:
            filtrar_s11 = bool(mat_info.get('filtrar_s11') or mat_info.get('filtrar_sg', False))
            tipo_filtro = mat_info.get('tipo_filtro', 'savitzky_golay')
            sg_w = int(mat_info.get('sg_window', 15) or 15)
            sg_p = int(mat_info.get('sg_polyorder', 3) or 3)
            pb_ord = int(mat_info.get('pb_orden', 2) or 2)
            pb_fc = float(mat_info.get('pb_fcorte', 0.10) or 0.10)
            med_w = int(mat_info.get('med_window', 15) or 15)
            spl_s = float(mat_info.get('spline_s', 0.10) or 0.10)
            m_rango = mat_info.get('modo_rango', 'todo')
            f1_g = float(mat_info.get('f1_ghz', 0.5) or 0.5)
            f2_g = float(mat_info.get('f2_ghz', 6.0) or 6.0)
            tipo_item = "material"
        elif clave_patron is not None:
            cfg_filtro_pat = self._config_desde_gui().get('filtrar_patrones_sg', {})
            if cfg_filtro_pat.get('habilitado', False):
                aplicar_en = cfg_filtro_pat.get('aplicar_en', [])
                if clave_patron in aplicar_en:
                    filtrar_s11 = True
                    tipo_filtro = cfg_filtro_pat.get('tipo', cfg_filtro_pat.get('tipo_filtro', 'savitzky_golay'))
                    sg_w = int(cfg_filtro_pat.get('sg_window', 15) or 15)
                    sg_p = int(cfg_filtro_pat.get('sg_polyorder', 3) or 3)
                    pb_ord = int(cfg_filtro_pat.get('pb_orden', 2) or 2)
                    pb_fc = float(cfg_filtro_pat.get('pb_fcorte', 0.10) or 0.10)
                    med_w = int(cfg_filtro_pat.get('med_window', 15) or 15)
                    spl_s = float(cfg_filtro_pat.get('spline_s', 0.10) or 0.10)
                    m_rango = cfg_filtro_pat.get('modo_rango', 'todo')
                    f1_g = float(cfg_filtro_pat.get('f1_ghz', 0.5) or 0.5)
                    f2_g = float(cfg_filtro_pat.get('f2_ghz', 6.0) or 6.0)
            tipo_item = "patrón de calibración"

        if filtrar_s11:
            desc_filtro = {
                'pasabajos': f"Pasabajos Butterworth (orden={pb_ord}, fc={pb_fc:.2f})",
                'savitzky_golay': f"Savitzky-Golay (ventana={sg_w}, orden={sg_p})",
                'mediana_movil': f"Mediana móvil (ventana={med_w})",
                'splines': f"Splines de suavizado (s={spl_s:.4f})",
            }.get(tipo_filtro, tipo_filtro)
            if m_rango == 'rango':
                desc_filtro += f" en [{f1_g:.2f}, {f2_g:.2f}] GHz"
            self.var_preview_s11_acond.set(
                f"Filtro {desc_filtro} activado para este {tipo_item}. Mostrando S11 original y filtrado simultáneamente."
            )
        else:
            if hasattr(self, 'var_preview_s11_acond'):
                self.var_preview_s11_acond.set("")

        self.var_preview_s11_status.set(f"Generando vista previa de '{etiqueta}'...")
        self.update_idletasks()  # forzar el redibujado antes de la parte lenta
        try:
            fig_mf, fig_sm, datos = _generar_preview_s11(
                ruta, log_x=bool(self.var_log_x.get()),
                filtrar_s11=filtrar_s11, tipo_filtro=tipo_filtro,
                sg_window=sg_w, sg_polyorder=sg_p,
                pb_orden=pb_ord, pb_fcorte=pb_fc,
                med_window=med_w, spline_s=spl_s,
                modo_rango=m_rango, f1_ghz=f1_g, f2_ghz=f2_g)
        except Exception as exc:
            self._limpiar_preview_s11(
                f"No se pudo generar la vista previa de '{etiqueta}':\n{exc}")
            return

        self.visor_preview_s11_magfase.mostrar(fig_mf)
        self.visor_preview_s11_smith.mostrar(fig_sm)

        frec = datos['Frec']
        msg_status = (
            f"{etiqueta}  ({os.path.basename(ruta)})  —  {len(frec)} puntos, "
            f"{frec[0] / 1e9:.3f}-{frec[-1] / 1e9:.3f} GHz"
        )
        if filtrar_s11:
            resumen_f = {
                'pasabajos': f"Pasabajos: ord={pb_ord}, fc={pb_fc:.2f}",
                'savitzky_golay': f"Savitzky-Golay: N={sg_w}, p={sg_p}",
                'mediana_movil': f"Mediana: N={med_w}",
                'splines': f"Splines: s={spl_s:.4f}",
            }.get(tipo_filtro, tipo_filtro)
            if m_rango == 'rango':
                resumen_f += f", [{f1_g:.2f}-{f2_g:.2f} GHz]"
            msg_status += f"  —  [{resumen_f}]"
        self.var_preview_s11_status.set(msg_status)

    def _mostrar_preview_gn(self):
        cfg = self._config_desde_gui()
        carpeta = cfg['carpeta_datos']
        archivos_cal = dict(cfg['archivos_calibracion'])
        for k in ('corto', 'aire', 'patron3', 'patron4'):
            arch = archivos_cal.get(k, "").strip()
            if not arch:
                self._limpiar_preview_s11(f"Falta definir el archivo para '{k}' en Calibracion.")
                return
            ruta = os.path.join(carpeta, arch)
            if not os.path.isfile(ruta):
                self._limpiar_preview_s11(f"No se encontro el archivo de '{k}':\n{ruta}")
                return

        self.var_preview_s11_status.set("Calculando vista previa de Gn(f)...")
        self.update_idletasks()

        try:
            calib = ap.cargar_calibracion(carpeta, archivos_cal, {})
            frecs = calib['patron3']['Frec']
            s11_corto = calib['corto']['Complex']
            s11_aire = calib['aire']['Complex']
            s11_patron3 = calib['patron3']['Complex']
            s11_patron4 = calib['patron4']['Complex']

            cfg_filtro_pat = cfg.get('filtrar_patrones_sg') or {}
            pat_filtrados = False
            if cfg_filtro_pat.get('habilitado', False):
                tipo_f_pat = cfg_filtro_pat.get('tipo', cfg_filtro_pat.get('tipo_filtro', 'savitzky_golay'))
                sg_w_pat = int(cfg_filtro_pat.get('sg_window', 15) or 15)
                sg_p_pat = int(cfg_filtro_pat.get('sg_polyorder', 3) or 3)
                pb_ord_pat = int(cfg_filtro_pat.get('pb_orden', 2) or 2)
                pb_fc_pat = float(cfg_filtro_pat.get('pb_fcorte', 0.10) or 0.10)
                aplicar_en = cfg_filtro_pat.get('aplicar_en', ['corto', 'aire', 'patron3', 'patron4'])

                def _filt_arr(arr):
                    if tipo_f_pat == 'pasabajos':
                        return ap.filtrar_s11_pasabajos(arr, orden=pb_ord_pat, f_corte=pb_fc_pat)
                    return ap.filtrar_s11_savgol(arr, window_length=sg_w_pat, polyorder=sg_p_pat)

                if 'corto' in aplicar_en:
                    s11_corto = _filt_arr(s11_corto)
                    pat_filtrados = True
                if 'aire' in aplicar_en:
                    s11_aire = _filt_arr(s11_aire)
                    pat_filtrados = True
                if 'patron3' in aplicar_en:
                    s11_patron3 = _filt_arr(s11_patron3)
                    pat_filtrados = True
                if 'patron4' in aplicar_en:
                    s11_patron4 = _filt_arr(s11_patron4)
                    pat_filtrados = True

            modelo_p3 = cfg['patron3_modelo']
            t_p3 = cfg['patron3_temperatura_c']
            Er_p3 = ap.PATRONES_TEORICOS[modelo_p3](frecs, T=t_p3)

            modelo_p4 = cfg['patron4_modelo']
            t_p4 = cfg['patron4_temperatura_c']
            Er_p4 = ap.PATRONES_TEORICOS[modelo_p4](frecs, T=t_p4)

            Gn_cal = ap.calcular_Gn(frecs, s11_patron3, s11_aire, s11_patron4, s11_corto,
                                    Er_p3, Er_p4)

            gn_acond = None
            info_acond = None
            cfg_acond = cfg.get('acondicionar_gn', {})
            tipo_acond = cfg_acond.get('tipo', 'transicion_polinomica_cero')
            modo_acond = cfg_acond.get('modo', 'automatico')
            if cfg_acond.get('habilitado', False):
                f1_acond = float(cfg_acond.get('f1_ghz', 0.0) or 0.0) * 1e9
                f2_acond = float(cfg_acond.get('f2_ghz', 0.5) or 0.5) * 1e9
                umbral_acond = float(cfg_acond.get('umbral_pct', 30.0) or 30.0)
                sg_w = int(cfg_acond.get('sg_window', 15) or 15)
                sg_p = int(cfg_acond.get('sg_polyorder', 3) or 3)
                pb_ord_ac = int(cfg_acond.get('pb_orden', 2) or 2)
                pb_fc_ac = float(cfg_acond.get('pb_fcorte', 0.10) or 0.10)

                gn_acond, info_acond = ap.acondicionar_gn(
                    frecs, Gn_cal, modo=modo_acond, tipo=tipo_acond,
                    f1=f1_acond, f2=f2_acond, umbral_pct=umbral_acond,
                    sg_window=sg_w, sg_polyorder=sg_p,
                    pb_orden=pb_ord_ac, pb_fcorte=pb_fc_ac
                )

            # Filtrar por rango f_min, f_max si esta definido
            try:
                f_min = float(cfg['f_min_ghz']) * 1e9
                f_max = float(cfg['f_max_ghz']) * 1e9
                mascara = (frecs >= f_min) & (frecs <= f_max)
                if np.any(mascara):
                    frecs_plot = frecs[mascara]
                    gn_plot = Gn_cal[mascara]
                    gn_acond_plot = gn_acond[mascara] if gn_acond is not None else None
                else:
                    frecs_plot = frecs
                    gn_plot = Gn_cal
                    gn_acond_plot = gn_acond
            except Exception:
                frecs_plot = frecs
                gn_plot = Gn_cal
                gn_acond_plot = gn_acond

            log_x = bool(cfg.get('escala_log_frecuencia', True))
            fig_re, fig_im = _generar_preview_gn(
                frecs_plot, gn_plot, Gn_acondicionado=gn_acond_plot, log_x=log_x,
                info_acond=info_acond)

        except Exception as exc:
            self._limpiar_preview_s11(
                f"No se pudo calcular la vista previa de Gn(f):\n{exc}")
            return

        self.frm_preview_s11_magfase.configure(text="Conductancia normalizada Re(Gn)")
        self.frm_preview_s11_smith.configure(text="Conductancia normalizada Im(Gn)")
        self.visor_preview_s11_magfase.mostrar(fig_re)
        self.visor_preview_s11_smith.mostrar(fig_im)

        texto_status = (
            f"[Calibracion] Conductancia normalizada Gn(f) \u2014 {len(frecs_plot)} puntos, "
            f"{frecs_plot[0] / 1e9:.3f}-{frecs_plot[-1] / 1e9:.3f} GHz"
        )
        if pat_filtrados:
            texto_status += f"  \u2014  [Patrones filtrados SG: N={sg_w_pat}, p={sg_p_pat}]"
        if gn_acond is not None:
            texto_status += f"  \u2014  Acondicionado: {tipo_acond} ({modo_acond})"
            if info_acond and info_acond.get('aviso'):
                texto_status += f"  [{info_acond['aviso']}]"
        self.var_preview_s11_status.set(texto_status)

        if gn_acond is not None and info_acond:
            tipo_info = info_acond.get('tipo')
            if tipo_info == 'savitzky_golay':
                w = info_acond.get('sg_window', 15)
                p = info_acond.get('sg_polyorder', 3)
                self.var_preview_s11_acond.set(
                    f"Filtro Savitzky-Golay aplicado a Gn(f): ventana = {w} puntos | orden polin\u00f3mico = {p}"
                )
            elif info_acond.get('modo') == 'automatico':
                if info_acond.get('disparo_detectado'):
                    if tipo_info == 'cero_clamping':
                        fc = info_acond.get('f_corte_aplicada', 0.0) or 0.0
                        fc_str = f"{fc/1e9:.4f} GHz ({fc/1e6:.2f} MHz)" if fc < 1e9 else f"{fc/1e9:.4f} GHz"
                        self.var_preview_s11_acond.set(
                            f"Frecuencia de corte autom\u00e1tica: f_corte = {fc_str} "
                            f"(anulaci\u00f3n a 0 por debajo de esta frec. con umbral {umbral_acond:.1f}%)"
                        )
                    else:
                        f1 = info_acond.get('f1_aplicada', 0.0) or 0.0
                        f2 = info_acond.get('f2_aplicada', 0.0) or 0.0
                        f1_str = f"{f1/1e9:.4f} GHz ({f1/1e6:.2f} MHz)" if (0 < f1 < 1e9) else f"{f1/1e9:.4f} GHz"
                        f2_str = f"{f2/1e9:.4f} GHz ({f2/1e6:.2f} MHz)" if f2 < 1e9 else f"{f2/1e9:.4f} GHz"
                        if tipo_info == 'transicion_polinomica_cero':
                            tipo_nom = "transici\u00f3n orden 5 (f\u2075)"
                        elif tipo_info == 'transicion_savgol':
                            tipo_nom = "transici\u00f3n orden 5 (f\u2075) + Savitzky-Golay"
                        else:
                            tipo_nom = "tapering"
                        self.var_preview_s11_acond.set(
                            f"Frecuencias de transici\u00f3n autom\u00e1ticas ({tipo_nom}): f1 = {f1_str} (Gn = 0)  |  "
                            f"f2 = {f2_str} (empalme continuo, detectada con umbral {umbral_acond:.1f}%)"
                        )
                else:
                    aviso_txt = info_acond.get('aviso', 'No se detect\u00f3 variaci\u00f3n mayor al umbral.')
                    self.var_preview_s11_acond.set(
                        f"Acondicionamiento autom\u00e1tico: {aviso_txt}"
                    )
            else:
                # Manual
                if tipo_info == 'cero_clamping':
                    fc = info_acond.get('f_corte_aplicada', 0.0) or 0.0
                    fc_str = f"{fc/1e9:.4f} GHz ({fc/1e6:.2f} MHz)" if fc < 1e9 else f"{fc/1e9:.4f} GHz"
                    self.var_preview_s11_acond.set(
                        f"Frecuencia de corte manual configurada: f_corte = {fc_str}"
                    )
                else:
                    f1 = info_acond.get('f1_aplicada', 0.0) or 0.0
                    f2 = info_acond.get('f2_aplicada', 0.0) or 0.0
                    f1_str = f"{f1/1e9:.4f} GHz"
                    f2_str = f"{f2/1e9:.4f} GHz ({f2/1e6:.2f} MHz)" if f2 < 1e9 else f"{f2/1e9:.4f} GHz"
                    if tipo_info == 'transicion_polinomica_cero':
                        tipo_nom = "transici\u00f3n orden 5 (f\u2075)"
                    elif tipo_info == 'transicion_savgol':
                        tipo_nom = "transici\u00f3n orden 5 (f\u2075) + Savitzky-Golay"
                    else:
                        tipo_nom = "tapering"
                    self.var_preview_s11_acond.set(
                        f"Frecuencias de transici\u00f3n manuales configuradas ({tipo_nom}): f1 = {f1_str}  |  f2 = {f2_str}"
                    )
        else:
            self.var_preview_s11_acond.set("")

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

        frm_err_freq = ttk.Frame(f)
        frm_err_freq.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(frm_err_freq, text="Intervalos de error para informe:").pack(side="left")
        self.var_f_err_int1 = tk.DoubleVar(value=2.0)
        self.var_f_err_int2 = tk.DoubleVar(value=4.0)
        ttk.Label(frm_err_freq, text="  F interm. 1 (GHz)").pack(side="left", padx=(12, 4))
        ttk.Spinbox(frm_err_freq, from_=0.0001, to=50, increment=0.1, textvariable=self.var_f_err_int1,
                    width=8).pack(side="left")
        ttk.Label(frm_err_freq, text="  F interm. 2 (GHz)").pack(side="left", padx=(12, 4))
        ttk.Spinbox(frm_err_freq, from_=0.0001, to=50, increment=0.1, textvariable=self.var_f_err_int2,
                    width=8).pack(side="left")
        _tooltip(frm_err_freq,
                 "Define 3 intervalos contiguos de frecuencia a partir de 4 frecuencias:\n"
                 "- Intervalo 1: [F min, F interm. 1]\n"
                 "- Intervalo 2: [F interm. 1, F interm. 2]\n"
                 "- Intervalo 3: [F interm. 2, F max]\n"
                 "En cada intervalo se calcula el error absoluto promedio y maximo para el informe PDF.")

        self.var_log_x = tk.BooleanVar(value=True)
        check_log_x = ttk.Checkbutton(
            f, text="Escala logaritmica en frecuencia (recomendado para barridos de varias decadas, p.ej. MHz a GHz)",
            variable=self.var_log_x)
        check_log_x.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))
        _tooltip(check_log_x,
                 "En escala lineal, si el barrido va de (por ejemplo) 1 MHz a\n"
                 "2 GHz, todo lo que pasa por debajo de un par de cientos de MHz\n"
                 "queda comprimido contra el borde izquierdo del grafico y no se\n"
                 "distingue nada. En log, cada decada ocupa el mismo ancho visual.\n"
                 "Se ignora automaticamente (cae a lineal) si el rango incluye 0 Hz.")

        # --- Filtrado de curvas de salida (er', er'') ---
        lf_filtro_salida = ttk.LabelFrame(f, text="Filtrado de curvas de salida (er', er'')", padding=8)
        lf_filtro_salida.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        frm_fs_f0 = ttk.Frame(lf_filtro_salida)
        frm_fs_f0.pack(anchor="w", pady=(0, 4))

        self.var_filtro_salida_activo = tk.BooleanVar(value=False)
        self.chk_filtro_salida = ttk.Checkbutton(
            frm_fs_f0, text="Aplicar filtro a permitividad calculada (er', er'')",
            variable=self.var_filtro_salida_activo,
            command=self._al_cambiar_filtro_salida)
        self.chk_filtro_salida.pack(side="left")

        self.var_filtro_salida_mostrar_original = tk.BooleanVar(value=True)
        self.chk_filtro_salida_orig = ttk.Checkbutton(
            frm_fs_f0, text="Mostrar curva sin filtro en el gráfico",
            variable=self.var_filtro_salida_mostrar_original)
        self.chk_filtro_salida_orig.pack(side="left", padx=(16, 0))
        _tooltip(self.chk_filtro_salida_orig,
                 "Muestra la curva original (sin filtrar) superpuesta con línea punteada\n"
                 "en el gráfico de permitividad para comparar el efecto del filtro.")

        self.frm_fs_f1 = ttk.Frame(lf_filtro_salida)
        self.frm_fs_f1.pack(anchor="w", pady=(2, 0))

        self.lbl_salida_filtro_tipo = ttk.Label(self.frm_fs_f1, text="Tipo:")
        self.lbl_salida_filtro_tipo.pack(side="left")
        self.var_filtro_salida_tipo = tk.StringVar(value="Savitzky-Golay")
        self.combo_filtro_salida_tipo = ttk.Combobox(
            self.frm_fs_f1, textvariable=self.var_filtro_salida_tipo, state="readonly", width=22,
            values=["Pasabajos (Butterworth)", "Savitzky-Golay", "Mediana móvil", "Splines de suavizado"])
        self.combo_filtro_salida_tipo.pack(side="left", padx=(4, 10))
        self.combo_filtro_salida_tipo.bind("<<ComboboxSelected>>", lambda _e: self._al_cambiar_filtro_salida())

        # Parametros Pasabajos salida
        self.lbl_salida_pb_ord = ttk.Label(self.frm_fs_f1, text="Orden PB:")
        self.var_salida_pb_ord = tk.IntVar(value=2)
        self.spin_salida_pb_ord = ttk.Spinbox(
            self.frm_fs_f1, from_=1, to=5, increment=1, textvariable=self.var_salida_pb_ord, width=3)
        self.lbl_salida_pb_fc = ttk.Label(self.frm_fs_f1, text="Corte PB (fc/fn):")
        self.var_salida_pb_fc = tk.DoubleVar(value=0.10)
        self.spin_salida_pb_fc = ttk.Spinbox(
            self.frm_fs_f1, from_=0.01, to=0.50, increment=0.01, textvariable=self.var_salida_pb_fc, width=5)

        # Parametros Savitzky-Golay salida
        self.lbl_salida_sg_win = ttk.Label(self.frm_fs_f1, text="Ventana SG:")
        self.var_salida_sg_win = tk.IntVar(value=15)
        self.spin_salida_sg_win = ttk.Spinbox(
            self.frm_fs_f1, from_=3, to=101, increment=2, textvariable=self.var_salida_sg_win, width=4)
        self.lbl_salida_sg_ord = ttk.Label(self.frm_fs_f1, text="Orden SG:")
        self.var_salida_sg_ord = tk.IntVar(value=3)
        self.spin_salida_sg_ord = ttk.Spinbox(
            self.frm_fs_f1, from_=1, to=5, increment=1, textvariable=self.var_salida_sg_ord, width=3)

        # Parametros Mediana móvil salida
        self.lbl_salida_med_win = ttk.Label(self.frm_fs_f1, text="Ventana Med:")
        self.var_salida_med_win = tk.IntVar(value=15)
        self.spin_salida_med_win = ttk.Spinbox(
            self.frm_fs_f1, from_=3, to=101, increment=2, textvariable=self.var_salida_med_win, width=4)

        # Parametros Splines salida
        self.lbl_salida_spline_s = ttk.Label(self.frm_fs_f1, text="Factor s:")
        self.var_salida_spline_s = tk.DoubleVar(value=0.10)
        self.spin_salida_spline_s = ttk.Spinbox(
            self.frm_fs_f1, from_=0.0001, to=50.0, increment=0.01, textvariable=self.var_salida_spline_s, width=6)

        # Rango
        self.lbl_salida_rango = ttk.Label(self.frm_fs_f1, text="  Rango:")
        self.lbl_salida_rango.pack(side="left")
        self.var_filtro_salida_rango = tk.StringVar(value="todo")
        self.radio_salida_rango_todo = ttk.Radiobutton(
            self.frm_fs_f1, text="Todo", value="todo", variable=self.var_filtro_salida_rango,
            command=self._al_cambiar_filtro_salida)
        self.radio_salida_rango_todo.pack(side="left", padx=(2, 4))
        self.radio_salida_rango_sub = ttk.Radiobutton(
            self.frm_fs_f1, text="[f1, f2]:", value="rango", variable=self.var_filtro_salida_rango,
            command=self._al_cambiar_filtro_salida)
        self.radio_salida_rango_sub.pack(side="left", padx=(2, 6))

        self.lbl_salida_f1 = ttk.Label(self.frm_fs_f1, text="f1 (GHz):")
        self.var_salida_f1 = tk.DoubleVar(value=0.5)
        self.spin_salida_f1 = ttk.Spinbox(
            self.frm_fs_f1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_salida_f1, width=6)
        self.lbl_salida_f2 = ttk.Label(self.frm_fs_f1, text="f2 (GHz):")
        self.var_salida_f2 = tk.DoubleVar(value=6.0)
        self.spin_salida_f2 = ttk.Spinbox(
            self.frm_fs_f1, from_=0.0, to=100.0, increment=0.05, textvariable=self.var_salida_f2, width=6)

        self._al_cambiar_filtro_salida()

        ttk.Separator(f).grid(row=7, column=0, columnspan=3, sticky="ew", pady=12)

        frm_acciones = ttk.Frame(f)
        frm_acciones.grid(row=8, column=0, columnspan=3, sticky="w")
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
            row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.barra_progreso = ttk.Progressbar(f, orient="horizontal", mode="determinate")
        self.barra_progreso.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        # --- Consola + vista previa, lado a lado ---
        frm_abajo = ttk.Frame(f)
        frm_abajo.grid(row=11, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        f.rowconfigure(11, weight=1)
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

        frm_preview_top = ttk.Frame(frm_preview)
        frm_preview_top.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        frm_preview_top.columnconfigure(0, weight=1)

        self.var_figura_seleccionada = tk.StringVar()
        self.combo_figuras = ttk.Combobox(frm_preview_top, textvariable=self.var_figura_seleccionada,
                                           state="readonly", values=[])
        self.combo_figuras.grid(row=0, column=0, sticky="ew")
        self.combo_figuras.bind("<<ComboboxSelected>>", lambda _e: self._mostrar_preview_seleccionada())

        self.var_limitar_eje_y = tk.BooleanVar(value=True)
        chk_lim_y = ttk.Checkbutton(frm_preview_top, text="Limitar eje Y a teórica",
                                    variable=self.var_limitar_eje_y,
                                    command=self._al_cambiar_limitar_eje_y)
        chk_lim_y.grid(row=0, column=1, padx=(10, 0))

        # El S11 (modulo/fase y Smith) de calibracion y materiales se ve en
        # la pestaña "3. Vista previa S11" -- ahi ademas se puede ver antes
        # de correr el analisis. Esta lista es solo lo que es resultado del
        # ANALISIS en si (permitividad medida vs. teorica).
        self.visor_preview = VisorFigura(frm_preview)
        self.visor_preview.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

    def _al_cambiar_limitar_eje_y(self):
        if hasattr(self, '_resultado_analisis') and self._resultado_analisis:
            self._mostrar_figuras_disponibles(self._figuras_desde_resultado(self._resultado_analisis))

    def _al_cambiar_filtro_salida(self):
        activo = bool(self.var_filtro_salida_activo.get())
        estado = "normal" if activo else "disabled"
        self.chk_filtro_salida_orig.configure(state=estado)
        self.lbl_salida_filtro_tipo.configure(state=estado)
        self.combo_filtro_salida_tipo.configure(state=("readonly" if activo else "disabled"))
        self.lbl_salida_rango.configure(state=estado)
        self.radio_salida_rango_todo.configure(state=estado)
        self.radio_salida_rango_sub.configure(state=estado)

        tipo = self.var_filtro_salida_tipo.get()

        for w in (self.lbl_salida_pb_ord, self.spin_salida_pb_ord, self.lbl_salida_pb_fc, self.spin_salida_pb_fc,
                  self.lbl_salida_sg_win, self.spin_salida_sg_win, self.lbl_salida_sg_ord, self.spin_salida_sg_ord,
                  self.lbl_salida_med_win, self.spin_salida_med_win, self.lbl_salida_spline_s, self.spin_salida_spline_s):
            w.pack_forget()

        if activo:
            if tipo == "Pasabajos (Butterworth)":
                self.lbl_salida_pb_ord.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_pb_ord.pack(side="left", padx=(2, 6), before=self.lbl_salida_rango)
                self.lbl_salida_pb_fc.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_pb_fc.pack(side="left", padx=(2, 10), before=self.lbl_salida_rango)
            elif tipo == "Mediana móvil":
                self.lbl_salida_med_win.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_med_win.pack(side="left", padx=(2, 10), before=self.lbl_salida_rango)
            elif tipo == "Splines de suavizado":
                self.lbl_salida_spline_s.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_spline_s.pack(side="left", padx=(2, 10), before=self.lbl_salida_rango)
            else:  # Savitzky-Golay
                self.lbl_salida_sg_win.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_sg_win.pack(side="left", padx=(2, 6), before=self.lbl_salida_rango)
                self.lbl_salida_sg_ord.pack(side="left", before=self.lbl_salida_rango)
                self.spin_salida_sg_ord.pack(side="left", padx=(2, 10), before=self.lbl_salida_rango)

        es_rango = (self.var_filtro_salida_rango.get() == "rango")
        if activo and es_rango:
            self.lbl_salida_f1.pack(side="left")
            self.spin_salida_f1.pack(side="left", padx=(2, 6))
            self.lbl_salida_f2.pack(side="left")
            self.spin_salida_f2.pack(side="left", padx=(2, 6))
        else:
            self.lbl_salida_f1.pack_forget()
            self.spin_salida_f1.pack_forget()
            self.lbl_salida_f2.pack_forget()
            self.spin_salida_f2.pack_forget()

    def _examinar_carpeta_salida(self):
        ruta = filedialog.askdirectory(title="Carpeta de salida", parent=self)
        if ruta:
            self.var_carpeta_salida.set(ruta)

    # -----------------------------------------------------------------
    # Pestaña 5: comparar mediciones desde CSVs ya generados (p.ej. de
    # distintos dias -- ver la estructura de carpetas con fecha/hora de
    # ejecutar_analisis). No hace falta correr el analisis para usar esta
    # pestaña: solo elegir CSVs de corridas anteriores.
    # -----------------------------------------------------------------
    def _construir_tab_comparar(self):
        f = self.tab_comparar
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        ttk.Label(f, text="Comparar mediciones", style="Titulo.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            f, text="Superponer en un mismo grafico varias curvas ya calculadas -- por "
                    "ejemplo, la misma muestra medida en distintos dias, o distintos "
                    "cortes de un mismo material. Cada curva se toma de un archivo "
                    "'tabla_<material>.csv' ya generado por este programa (los CSVs de "
                    "corridas anteriores siguen disponibles en sus carpetas por fecha/hora, "
                    "dentro de la carpeta de salida).",
            style="Ayuda.TLabel", wraplength=1000, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        paned = ttk.PanedWindow(f, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew")

        # --- Panel izquierdo: tabla de curvas cargadas ---
        frm_izq = ttk.Frame(paned)
        paned.add(frm_izq, weight=1)
        frm_izq.columnconfigure(0, weight=1)
        frm_izq.rowconfigure(0, weight=1)

        columnas = ("etiqueta", "archivo", "columna")
        self.tabla_comparar = ttk.Treeview(
            frm_izq, columns=columnas, show="headings", selectmode="browse", height=14)
        titulos = {"etiqueta": "Etiqueta", "archivo": "Archivo", "columna": "Columna"}
        anchos = {"etiqueta": 160, "archivo": 220, "columna": 170}
        for c in columnas:
            self.tabla_comparar.heading(c, text=titulos[c])
            self.tabla_comparar.column(c, width=anchos[c], anchor="w")
        self.tabla_comparar.grid(row=0, column=0, sticky="nsew")
        self.tabla_comparar.bind("<Double-1>", lambda _e: self._editar_curva_comparar())
        self.tabla_comparar.bind("<Delete>", lambda _e: self._quitar_curva_comparar())

        scroll = ttk.Scrollbar(frm_izq, orient="vertical", command=self.tabla_comparar.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tabla_comparar.configure(yscrollcommand=scroll.set)

        frm_botones = ttk.Frame(frm_izq)
        frm_botones.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(frm_botones, text="Agregar...", command=self._agregar_curva_comparar).pack(side="left")
        ttk.Button(frm_botones, text="Duplicar", command=self._duplicar_curva_comparar).pack(
            side="left", padx=6)
        ttk.Button(frm_botones, text="Editar...", command=self._editar_curva_comparar).pack(
            side="left", padx=6)
        ttk.Button(frm_botones, text="Quitar", command=self._quitar_curva_comparar).pack(side="left")
        ttk.Separator(frm_botones, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(frm_botones, text="Subir",
                   command=lambda: self._mover_curva_comparar(-1)).pack(side="left")
        ttk.Button(frm_botones, text="Bajar",
                   command=lambda: self._mover_curva_comparar(1)).pack(side="left", padx=6)

        self.var_log_x_comparar = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_izq, text="Escala logaritmica en frecuencia",
                         variable=self.var_log_x_comparar,
                         command=self._graficar_comparacion_curvas).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # --- Panel derecho: grafico interactivo ---
        frm_der = ttk.LabelFrame(paned, text="er' / er'' superpuestos")
        paned.add(frm_der, weight=2)
        self.visor_comparar = VisorFigura(frm_der)
        self.visor_comparar.pack(fill="both", expand=True, padx=4, pady=4)
        self.visor_comparar.limpiar(
            "Agrega una o mas curvas con 'Agregar...' para verlas superpuestas aca.")

    def _refrescar_tabla_comparar(self):
        self.tabla_comparar.delete(*self.tabla_comparar.get_children())
        for i, c in enumerate(self._curvas_comparar):
            self.tabla_comparar.insert(
                "", "end", iid=str(i),
                values=(c['etiqueta'], os.path.basename(c['archivo']), c['columna']))

    def _indice_curva_comparar_seleccionada(self):
        sel = self.tabla_comparar.selection()
        return int(sel[0]) if sel else None

    def _agregar_curva_comparar(self):
        dialogo = DialogoCurvaComparar(self)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_comparar.append(dialogo.resultado)
            self._refrescar_tabla_comparar()
            self._graficar_comparacion_curvas()

    def _duplicar_curva_comparar(self):
        idx = self._indice_curva_comparar_seleccionada()
        if idx is None:
            messagebox.showinfo("Duplicar curva", "Elegi primero una fila de la tabla.", parent=self)
            return
        clon = dict(self._curvas_comparar[idx])
        clon['etiqueta'] = f"{clon['etiqueta']} (copia)"
        dialogo = DialogoCurvaComparar(self, curva=clon)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_comparar.insert(idx + 1, dialogo.resultado)
            self._refrescar_tabla_comparar()
            self._graficar_comparacion_curvas()
            self.tabla_comparar.selection_set(str(idx + 1))

    def _editar_curva_comparar(self):
        idx = self._indice_curva_comparar_seleccionada()
        if idx is None:
            messagebox.showinfo("Editar curva", "Elegi primero una fila de la tabla.", parent=self)
            return
        dialogo = DialogoCurvaComparar(self, curva=dict(self._curvas_comparar[idx]))
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_comparar[idx] = dialogo.resultado
            self._refrescar_tabla_comparar()
            self._graficar_comparacion_curvas()

    def _quitar_curva_comparar(self):
        idx = self._indice_curva_comparar_seleccionada()
        if idx is None:
            return
        del self._curvas_comparar[idx]
        self._refrescar_tabla_comparar()
        self._graficar_comparacion_curvas()

    def _mover_curva_comparar(self, delta):
        idx = self._indice_curva_comparar_seleccionada()
        if idx is None:
            return
        nuevo = idx + delta
        if not (0 <= nuevo < len(self._curvas_comparar)):
            return
        self._curvas_comparar[idx], self._curvas_comparar[nuevo] = \
            self._curvas_comparar[nuevo], self._curvas_comparar[idx]
        self._refrescar_tabla_comparar()
        self.tabla_comparar.selection_set(str(nuevo))

    def _graficar_comparacion_curvas(self):
        """Relee (siempre desde cero, por si el CSV cambio) cada curva
        cargada y redibuja el grafico superpuesto. Se llama sola despues
        de agregar/editar/quitar/mover una curva o cambiar la escala."""
        if not self._curvas_comparar:
            self.visor_comparar.limpiar(
                "Agrega una o mas curvas con 'Agregar...' para verlas superpuestas aca.")
            return

        series = {}
        errores = []
        for c in self._curvas_comparar:
            try:
                frecs, columnas = ap.leer_csv_resultado(c['archivo'])
                series[c['etiqueta']] = (frecs, columnas[c['columna']])
            except Exception as exc:
                errores.append(f"'{c['etiqueta']}': {exc}")

        if not series:
            self.visor_comparar.limpiar(
                "No se pudo leer ninguna de las curvas cargadas:\n" + "\n".join(errores))
            return

        try:
            fig = ap.graficar_series_multiples(
                series, log_x=bool(self.var_log_x_comparar.get()))
        except Exception as exc:
            self.visor_comparar.limpiar(f"No se pudo generar el grafico:\n{exc}")
            return

        self.visor_comparar.mostrar(fig)
        if errores:
            self._status("Algunas curvas no se pudieron leer: " + "; ".join(errores))

    # -----------------------------------------------------------------
    # Pestaña 6: graficar modelos teoricos directamente (sin datos
    # medidos) -- para explorar como se ve un modelo de Patrones.py en
    # un rango de frecuencias, temperatura y cantidad de puntos elegidos,
    # o comparar el mismo modelo a distintas temperaturas entre si.
    # -----------------------------------------------------------------
    def _construir_tab_modelos(self):
        f = self.tab_modelos
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        ttk.Label(f, text="Modelos teoricos", style="Titulo.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            f, text="Grafica cualquier modelo teorico de Patrones.py directamente, sin "
                    "necesidad de ningun dato medido -- eligiendo el rango de frecuencias, "
                    "la temperatura y la cantidad de puntos. Sirve para explorar como se ve "
                    "un modelo, o para comparar el mismo liquido a distintas temperaturas "
                    "(o distintos liquidos entre si) superpuestos en un mismo grafico.",
            style="Ayuda.TLabel", wraplength=1000, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        paned = ttk.PanedWindow(f, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew")

        # --- Panel izquierdo: tabla de curvas cargadas ---
        frm_izq = ttk.Frame(paned)
        paned.add(frm_izq, weight=1)
        frm_izq.columnconfigure(0, weight=1)
        frm_izq.rowconfigure(0, weight=1)

        columnas = ("etiqueta", "modelo", "temperatura", "rango", "puntos")
        self.tabla_modelos = ttk.Treeview(
            frm_izq, columns=columnas, show="headings", selectmode="browse", height=14)
        titulos = {"etiqueta": "Etiqueta", "modelo": "Modelo", "temperatura": "T (°C)",
                   "rango": "Rango (GHz)", "puntos": "Puntos"}
        anchos = {"etiqueta": 140, "modelo": 150, "temperatura": 60, "rango": 110, "puntos": 60}
        for c in columnas:
            self.tabla_modelos.heading(c, text=titulos[c])
            self.tabla_modelos.column(c, width=anchos[c], anchor="w")
        self.tabla_modelos.grid(row=0, column=0, sticky="nsew")
        self.tabla_modelos.bind("<Double-1>", lambda _e: self._editar_modelo())
        self.tabla_modelos.bind("<Delete>", lambda _e: self._quitar_modelo())

        scroll = ttk.Scrollbar(frm_izq, orient="vertical", command=self.tabla_modelos.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tabla_modelos.configure(yscrollcommand=scroll.set)

        frm_botones = ttk.Frame(frm_izq)
        frm_botones.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(frm_botones, text="Agregar...", command=self._agregar_modelo).pack(side="left")
        ttk.Button(frm_botones, text="Duplicar", command=self._duplicar_modelo).pack(
            side="left", padx=6)
        ttk.Button(frm_botones, text="Editar...", command=self._editar_modelo).pack(side="left", padx=6)
        ttk.Button(frm_botones, text="Quitar", command=self._quitar_modelo).pack(side="left")
        ttk.Separator(frm_botones, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(frm_botones, text="Subir", command=lambda: self._mover_modelo(-1)).pack(side="left")
        ttk.Button(frm_botones, text="Bajar", command=lambda: self._mover_modelo(1)).pack(
            side="left", padx=6)

        boton_usar_actuales = ttk.Button(
            frm_izq, text="Usar Patr\u00f3n 3/4 actuales", command=self._usar_patrones_actuales)
        boton_usar_actuales.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        _tooltip(boton_usar_actuales,
                 "Agrega a la lista el/los liquido(s) configurados AHORA MISMO como\n"
                 "Patron 3 (y Patron 4, si el metodo completo esta habilitado) en la\n"
                 "pestaña de Calibracion, con su temperatura tal cual esta ahi -- para\n"
                 "no tener que volver a tipearlos si solo queres ver como luce la\n"
                 "calibracion vigente.")

        self.var_log_x_modelos = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_izq, text="Escala logaritmica en frecuencia",
                         variable=self.var_log_x_modelos,
                         command=self._graficar_modelos).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # --- Panel derecho: grafico interactivo ---
        frm_der = ttk.LabelFrame(paned, text="er' / er'' superpuestos")
        paned.add(frm_der, weight=2)
        self.visor_modelos = VisorFigura(frm_der)
        self.visor_modelos.pack(fill="both", expand=True, padx=4, pady=4)
        self.visor_modelos.limpiar(
            "Agrega un modelo teorico con 'Agregar...' para verlo graficado aca.")

    def _refrescar_tabla_modelos(self):
        self.tabla_modelos.delete(*self.tabla_modelos.get_children())
        for i, c in enumerate(self._curvas_modelos):
            self.tabla_modelos.insert(
                "", "end", iid=str(i),
                values=(c['etiqueta'], gf.etiqueta_de_modelo(c['modelo']), f"{c['temperatura']:.1f}",
                        f"{c['f_min_ghz']:g}\u2013{c['f_max_ghz']:g}", c['n_puntos']))

    def _indice_modelo_seleccionado(self):
        sel = self.tabla_modelos.selection()
        return int(sel[0]) if sel else None

    def _agregar_modelo(self):
        dialogo = DialogoModeloTeorico(self)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_modelos.append(dialogo.resultado)
            self._refrescar_tabla_modelos()
            self._graficar_modelos()

    def _duplicar_modelo(self):
        idx = self._indice_modelo_seleccionado()
        if idx is None:
            messagebox.showinfo("Duplicar modelo", "Elegi primero una fila de la tabla.", parent=self)
            return
        clon = dict(self._curvas_modelos[idx])
        clon['etiqueta'] = f"{clon['etiqueta']} (copia)"
        dialogo = DialogoModeloTeorico(self, curva=clon)
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_modelos.insert(idx + 1, dialogo.resultado)
            self._refrescar_tabla_modelos()
            self._graficar_modelos()
            self.tabla_modelos.selection_set(str(idx + 1))

    def _usar_patrones_actuales(self):
        """Agrega el/los liquido(s) configurados AHORA MISMO como Patron 3
        (y Patron 4, si el metodo completo esta habilitado) en la pestaña
        de Calibracion, con su temperatura tal cual esta ahi -- para no
        tener que volver a tipearlos si solo se quiere ver como luce la
        calibracion vigente. Usa el rango de frecuencias configurado en
        la pestaña 'Salida y ejecucion' (F min/F max)."""
        f_min = float(self._obtener_double(self.var_f_min, 0.5))
        f_max = float(self._obtener_double(self.var_f_max, 6.0))
        agregadas = 0

        modelo_p3 = gf.clave_de_etiqueta(self.vars_modelo_calibracion['patron3'].get())
        if modelo_p3:
            temp_p3 = float(self._obtener_double(self.vars_temp_calibracion['patron3'], 25.0))
            self._curvas_modelos.append({
                'etiqueta': f"Patron 3 actual ({gf.etiqueta_de_modelo(modelo_p3)}, {temp_p3:.1f}\u00b0C)",
                'modelo': modelo_p3, 'temperatura': temp_p3,
                'f_min_ghz': f_min, 'f_max_ghz': f_max, 'n_puntos': 200,
            })
            agregadas += 1

        if self.var_usar_metodo_completo.get():
            modelo_p4 = gf.clave_de_etiqueta(self.vars_modelo_calibracion['patron4'].get())
            if modelo_p4:
                temp_p4 = float(self._obtener_double(self.vars_temp_calibracion['patron4'], 25.0))
                self._curvas_modelos.append({
                    'etiqueta': f"Patron 4 actual ({gf.etiqueta_de_modelo(modelo_p4)}, "
                                f"{temp_p4:.1f}\u00b0C)",
                    'modelo': modelo_p4, 'temperatura': temp_p4,
                    'f_min_ghz': f_min, 'f_max_ghz': f_max, 'n_puntos': 200,
                })
                agregadas += 1

        if agregadas:
            self._refrescar_tabla_modelos()
            self._graficar_modelos()
            self._status(f"Se agregaron {agregadas} curva(s) desde la calibracion actual.")
        else:
            messagebox.showinfo(
                "Usar patrones actuales",
                "No hay ningun modelo teorico elegido todavia en la pesta\u00f1a de "
                "Calibracion.", parent=self)

    def _editar_modelo(self):
        idx = self._indice_modelo_seleccionado()
        if idx is None:
            messagebox.showinfo("Editar modelo", "Elegi primero una fila de la tabla.", parent=self)
            return
        dialogo = DialogoModeloTeorico(self, curva=dict(self._curvas_modelos[idx]))
        self.wait_window(dialogo)
        if dialogo.resultado is not None:
            self._curvas_modelos[idx] = dialogo.resultado
            self._refrescar_tabla_modelos()
            self._graficar_modelos()

    def _quitar_modelo(self):
        idx = self._indice_modelo_seleccionado()
        if idx is None:
            return
        del self._curvas_modelos[idx]
        self._refrescar_tabla_modelos()
        self._graficar_modelos()

    def _mover_modelo(self, delta):
        idx = self._indice_modelo_seleccionado()
        if idx is None:
            return
        nuevo = idx + delta
        if not (0 <= nuevo < len(self._curvas_modelos)):
            return
        self._curvas_modelos[idx], self._curvas_modelos[nuevo] = \
            self._curvas_modelos[nuevo], self._curvas_modelos[idx]
        self._refrescar_tabla_modelos()
        self.tabla_modelos.selection_set(str(nuevo))

    def _graficar_modelos(self):
        """Recalcula (siempre desde cero) cada curva cargada y redibuja
        el grafico superpuesto. Se llama sola despues de agregar/editar/
        quitar/mover una curva o cambiar la escala."""
        if not self._curvas_modelos:
            self.visor_modelos.limpiar(
                "Agrega un modelo teorico con 'Agregar...' para verlo graficado aca.")
            return

        log_x = bool(self.var_log_x_modelos.get())
        series = {}
        errores = []
        for c in self._curvas_modelos:
            try:
                frecs = ap.generar_grilla_frecuencias(
                    c['f_min_ghz'], c['f_max_ghz'], c['n_puntos'], log_x=log_x)
                er = ap.PATRONES_TEORICOS[c['modelo']](frecs, T=c['temperatura'])
                series[c['etiqueta']] = (frecs, er)
            except Exception as exc:
                errores.append(f"'{c['etiqueta']}': {exc}")

        if not series:
            self.visor_modelos.limpiar(
                "No se pudo calcular ninguna de las curvas cargadas:\n" + "\n".join(errores))
            return

        try:
            fig = ap.graficar_series_multiples(
                series, titulo="Modelos te\u00f3ricos", log_x=log_x)
        except Exception as exc:
            self.visor_modelos.limpiar(f"No se pudo generar el grafico:\n{exc}")
            return

        self.visor_modelos.mostrar(fig)
        if errores:
            self._status("Algunos modelos no se pudieron calcular: " + "; ".join(errores))

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
        aplicar_en_pat = []
        if getattr(self, 'var_sg_pat_corto', None) and self.var_sg_pat_corto.get():
            aplicar_en_pat.append('corto')
        if getattr(self, 'var_sg_pat_aire', None) and self.var_sg_pat_aire.get():
            aplicar_en_pat.append('aire')
        if getattr(self, 'var_sg_pat_p3', None) and self.var_sg_pat_p3.get():
            aplicar_en_pat.append('patron3')
        if getattr(self, 'var_sg_pat_p4', None) and self.var_sg_pat_p4.get():
            aplicar_en_pat.append('patron4')
        if getattr(self, 'var_sg_pat_adic', None) and self.var_sg_pat_adic.get():
            aplicar_en_pat.append('adicionales')

        return {
            'version': gf.CONFIG_VERSION,
            'carpeta_datos': self.var_carpeta_datos.get().strip(),
            'archivos_calibracion': {
                clave: var.get().strip() for clave, var in self.vars_calibracion.items()
            },
            'patron3_modelo': gf.clave_de_etiqueta(self.vars_modelo_calibracion['patron3'].get()),
            'patron3_temperatura_c': float(self._obtener_double(
                self.vars_temp_calibracion['patron3'], 25.0)),
            'patron4_modelo': gf.clave_de_etiqueta(self.vars_modelo_calibracion['patron4'].get()),
            'patron4_temperatura_c': float(self._obtener_double(
                self.vars_temp_calibracion['patron4'], 25.0)),
            'usar_metodo_completo': bool(self.var_usar_metodo_completo.get()),
            'usar_minimos_cuadrados': bool(self.var_usar_mc.get()),
            'patrones_adicionales': copy.deepcopy(self._patrones_adicionales),
            'estrategia_gn': self.var_estrategia_gn.get(),
            'filtrar_patrones_sg': {
                'habilitado': bool(self.var_filtrar_patrones_sg.get()) if hasattr(self, 'var_filtrar_patrones_sg') else False,
                'tipo': {
                    "Pasabajos (Butterworth)": "pasabajos",
                    "Savitzky-Golay": "savitzky_golay",
                    "Mediana móvil": "mediana_movil",
                    "Splines de suavizado": "splines",
                }.get(self.var_patrones_tipo_filtro.get() if hasattr(self, 'var_patrones_tipo_filtro') else '', 'savitzky_golay'),
                'sg_window': int(self.var_patrones_sg_win.get() if hasattr(self, 'var_patrones_sg_win') else 15),
                'sg_polyorder': int(self.var_patrones_sg_ord.get() if hasattr(self, 'var_patrones_sg_ord') else 3),
                'pb_orden': int(self.var_patrones_pb_ord.get() if hasattr(self, 'var_patrones_pb_ord') else 2),
                'pb_fcorte': float(self._obtener_double(getattr(self, 'var_patrones_pb_fc', None), 0.10)),
                'med_window': int(self.var_patrones_med_win.get() if hasattr(self, 'var_patrones_med_win') else 15),
                'spline_s': float(self._obtener_double(getattr(self, 'var_patrones_spline_s', None), 0.10)),
                'modo_rango': self.var_patrones_rango.get() if hasattr(self, 'var_patrones_rango') else 'todo',
                'f1_ghz': float(self._obtener_double(getattr(self, 'var_pat_f1', None), 0.5)),
                'f2_ghz': float(self._obtener_double(getattr(self, 'var_pat_f2', None), 6.0)),
                'aplicar_en': aplicar_en_pat,
            },
            'acondicionar_gn': {
                'habilitado': bool(self.var_acondicionar_gn.get()),
                'tipo': _TIPOS_ACOND_MAP.get(
                    self.var_acond_tipo.get(), 'transicion_polinomica_cero'),
                'modo': self.var_acond_modo.get(),
                'f1_ghz': float(self._obtener_double(self.var_acond_f1_ghz, 0.0)),
                'f2_ghz': float(self._obtener_double(self.var_acond_f2_ghz, 0.5)),
                'umbral_pct': float(self._obtener_double(self.var_acond_umbral_pct, 30.0)),
                'pb_orden': int(self.var_acond_pb_orden.get() if hasattr(self, 'var_acond_pb_orden') else 2),
                'pb_fcorte': float(self._obtener_double(getattr(self, 'var_acond_pb_fcorte', None), 0.10)),
                'sg_window': int(self.var_acond_sg_window.get() if hasattr(self, 'var_acond_sg_window') else 15),
                'sg_polyorder': int(self.var_acond_sg_polyorder.get() if hasattr(self, 'var_acond_sg_polyorder') else 3),
                'med_window': int(self.var_acond_med_window.get() if hasattr(self, 'var_acond_med_window') else 15),
                'spline_s': float(self._obtener_double(getattr(self, 'var_acond_spline_s', None), 0.10)),
                'comparar_original': bool(self.var_comparar_gn_original.get()),
            },
            'materiales': copy.deepcopy(self._materiales),
            'filtro_lote_materiales': {
                'tipo': {
                    "Pasabajos (Butterworth)": "pasabajos",
                    "Savitzky-Golay": "savitzky_golay",
                    "Mediana móvil": "mediana_movil",
                    "Splines de suavizado": "splines",
                }.get(self.var_lote_mat_tipo.get() if hasattr(self, 'var_lote_mat_tipo') else '', 'savitzky_golay'),
                'pb_orden': int(self.var_lote_pb_ord.get() if hasattr(self, 'var_lote_pb_ord') else 2),
                'pb_fcorte': float(self._obtener_double(getattr(self, 'var_lote_pb_fc', None), 0.10)),
                'sg_window': int(self.var_lote_sg_win.get() if hasattr(self, 'var_lote_sg_win') else 15),
                'sg_polyorder': int(self.var_lote_sg_ord.get() if hasattr(self, 'var_lote_sg_ord') else 3),
                'med_window': int(self.var_lote_med_win.get() if hasattr(self, 'var_lote_med_win') else 15),
                'spline_s': float(self._obtener_double(getattr(self, 'var_lote_spline_s', None), 0.10)),
                'modo_rango': self.var_lote_rango.get() if hasattr(self, 'var_lote_rango') else 'todo',
                'f1_ghz': float(self._obtener_double(getattr(self, 'var_lote_f1', None), 0.5)),
                'f2_ghz': float(self._obtener_double(getattr(self, 'var_lote_f2', None), 6.0)),
            },
            'carpeta_salida': self.var_carpeta_salida.get().strip(),
            'nombre_informe_pdf': self.var_nombre_pdf.get().strip() or "informe_permitividad.pdf",
            'f_min_ghz': float(self._obtener_double(self.var_f_min, 0.5)),
            'f_max_ghz': float(self._obtener_double(self.var_f_max, 6.0)),
            'f_err_int1_ghz': float(self._obtener_double(self.var_f_err_int1, 2.0)),
            'f_err_int2_ghz': float(self._obtener_double(self.var_f_err_int2, 4.0)),
            'escala_log_frecuencia': bool(self.var_log_x.get()),
            'filtro_salida': {
                'habilitado': bool(self.var_filtro_salida_activo.get()) if hasattr(self, 'var_filtro_salida_activo') else False,
                'mostrar_sin_filtro': bool(self.var_filtro_salida_mostrar_original.get()) if hasattr(self, 'var_filtro_salida_mostrar_original') else True,
                'tipo': {
                    "Pasabajos (Butterworth)": "pasabajos",
                    "Savitzky-Golay": "savitzky_golay",
                    "Mediana móvil": "mediana_movil",
                    "Splines de suavizado": "splines",
                }.get(self.var_filtro_salida_tipo.get() if hasattr(self, 'var_filtro_salida_tipo') else '', 'savitzky_golay'),
                'pb_orden': int(self.var_salida_pb_ord.get() if hasattr(self, 'var_salida_pb_ord') else 2),
                'pb_fcorte': float(self._obtener_double(getattr(self, 'var_salida_pb_fc', None), 0.10)),
                'sg_window': int(self.var_salida_sg_win.get() if hasattr(self, 'var_salida_sg_win') else 15),
                'sg_polyorder': int(self.var_salida_sg_ord.get() if hasattr(self, 'var_salida_sg_ord') else 3),
                'med_window': int(self.var_salida_med_win.get() if hasattr(self, 'var_salida_med_win') else 15),
                'spline_s': float(self._obtener_double(getattr(self, 'var_salida_spline_s', None), 0.10)),
                'modo_rango': self.var_filtro_salida_rango.get() if hasattr(self, 'var_filtro_salida_rango') else 'todo',
                'f1_ghz': float(self._obtener_double(getattr(self, 'var_salida_f1', None), 0.5)),
                'f2_ghz': float(self._obtener_double(getattr(self, 'var_salida_f2', None), 6.0)),
            },
            'limitar_eje_y': bool(self.var_limitar_eje_y.get()) if hasattr(self, 'var_limitar_eje_y') else True,
            'curvas_comparacion': copy.deepcopy(self._curvas_comparar),
            'curvas_modelos': copy.deepcopy(self._curvas_modelos),
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
        self.vars_modelo_calibracion['patron3'].set(
            gf.etiqueta_de_modelo(config.get('patron3_modelo') or 'agua'))
        self.vars_temp_calibracion['patron3'].set(
            float(config.get('patron3_temperatura_c', 25.0) or 25.0))
        self.vars_modelo_calibracion['patron4'].set(
            gf.etiqueta_de_modelo(config.get('patron4_modelo') or 'alcohol_isopropilico'))
        self.vars_temp_calibracion['patron4'].set(
            float(config.get('patron4_temperatura_c', 25.0) or 25.0))
        self.var_usar_metodo_completo.set(bool(config.get('usar_metodo_completo', True)))
        self.var_usar_mc.set(bool(config.get('usar_minimos_cuadrados', False)))
        self._patrones_adicionales = copy.deepcopy(config.get('patrones_adicionales', []) or [])
        self._refrescar_tabla_adicionales()
        self.var_estrategia_gn.set(config.get('estrategia_gn') or 'minimo_gn')

        cfg_filtro_pat = config.get('filtrar_patrones_sg') or config.get('filtrar_patrones') or {}
        if hasattr(self, 'var_filtrar_patrones_sg'):
            self.var_filtrar_patrones_sg.set(bool(cfg_filtro_pat.get('habilitado', False)))
        tipo_f_pat = cfg_filtro_pat.get('tipo', cfg_filtro_pat.get('tipo_filtro', 'savitzky_golay'))
        map_inv_pat = {
            'pasabajos': "Pasabajos (Butterworth)",
            'savitzky_golay': "Savitzky-Golay",
            'mediana_movil': "Mediana móvil",
            'splines': "Splines de suavizado",
        }
        if hasattr(self, 'var_patrones_tipo_filtro'):
            self.var_patrones_tipo_filtro.set(map_inv_pat.get(tipo_f_pat, "Savitzky-Golay"))
        if hasattr(self, 'var_patrones_pb_ord'):
            self.var_patrones_pb_ord.set(int(cfg_filtro_pat.get('pb_orden', 2) or 2))
        if hasattr(self, 'var_patrones_pb_fc'):
            self.var_patrones_pb_fc.set(float(cfg_filtro_pat.get('pb_fcorte', 0.10) or 0.10))
        if hasattr(self, 'var_patrones_sg_win'):
            self.var_patrones_sg_win.set(int(cfg_filtro_pat.get('sg_window', 15) or 15))
        if hasattr(self, 'var_patrones_sg_ord'):
            self.var_patrones_sg_ord.set(int(cfg_filtro_pat.get('sg_polyorder', 3) or 3))
        if hasattr(self, 'var_patrones_med_win'):
            self.var_patrones_med_win.set(int(cfg_filtro_pat.get('med_window', 15) or 15))
        if hasattr(self, 'var_patrones_spline_s'):
            self.var_patrones_spline_s.set(float(cfg_filtro_pat.get('spline_s', 0.10) or 0.10))
        if hasattr(self, 'var_patrones_rango'):
            self.var_patrones_rango.set(cfg_filtro_pat.get('modo_rango', 'todo'))
        if hasattr(self, 'var_pat_f1'):
            self.var_pat_f1.set(float(cfg_filtro_pat.get('f1_ghz', 0.5) or 0.5))
        if hasattr(self, 'var_pat_f2'):
            self.var_pat_f2.set(float(cfg_filtro_pat.get('f2_ghz', 6.0) or 6.0))
        aplicar_en_pat = cfg_filtro_pat.get('aplicar_en', ['corto', 'aire', 'patron3', 'patron4', 'adicionales'])
        if hasattr(self, 'var_sg_pat_corto'):
            self.var_sg_pat_corto.set('corto' in aplicar_en_pat)
        if hasattr(self, 'var_sg_pat_aire'):
            self.var_sg_pat_aire.set('aire' in aplicar_en_pat)
        if hasattr(self, 'var_sg_pat_p3'):
            self.var_sg_pat_p3.set('patron3' in aplicar_en_pat)
        if hasattr(self, 'var_sg_pat_p4'):
            self.var_sg_pat_p4.set('patron4' in aplicar_en_pat)
        if hasattr(self, 'var_sg_pat_adic'):
            self.var_sg_pat_adic.set('adicionales' in aplicar_en_pat)
        if hasattr(self, '_al_cambiar_filtrar_patrones_sg'):
            self._al_cambiar_filtrar_patrones_sg()

        acond_cfg = config.get('acondicionar_gn') or {}
        self.var_acondicionar_gn.set(bool(acond_cfg.get('habilitado', False)))
        tipo_c = acond_cfg.get('tipo', 'transicion_polinomica_cero')
        self.var_acond_tipo.set(
            _TIPOS_ACOND_INV.get(tipo_c, "Transicion orden 5 (f^5)"))
        self.var_acond_modo.set(acond_cfg.get('modo', 'automatico'))
        self.var_acond_f1_ghz.set(float(acond_cfg.get('f1_ghz', 0.0) or 0.0))
        self.var_acond_f2_ghz.set(float(acond_cfg.get('f2_ghz', 0.5) or 0.5))
        self.var_acond_umbral_pct.set(float(acond_cfg.get('umbral_pct', 30.0) or 30.0))
        if hasattr(self, 'var_acond_pb_orden'):
            self.var_acond_pb_orden.set(int(acond_cfg.get('pb_orden', 2) or 2))
        if hasattr(self, 'var_acond_pb_fcorte'):
            self.var_acond_pb_fcorte.set(float(acond_cfg.get('pb_fcorte', 0.10) or 0.10))
        if hasattr(self, 'var_acond_sg_window'):
            self.var_acond_sg_window.set(int(acond_cfg.get('sg_window', 15) or 15))
        if hasattr(self, 'var_acond_sg_polyorder'):
            self.var_acond_sg_polyorder.set(int(acond_cfg.get('sg_polyorder', 3) or 3))
        if hasattr(self, 'var_acond_med_window'):
            self.var_acond_med_window.set(int(acond_cfg.get('med_window', 15) or 15))
        if hasattr(self, 'var_acond_spline_s'):
            self.var_acond_spline_s.set(float(acond_cfg.get('spline_s', 0.10) or 0.10))
        self.var_comparar_gn_original.set(bool(acond_cfg.get('comparar_original', True)))

        self._al_cambiar_usar_metodo_completo()
        self._al_cambiar_usar_mc(actualizar_materiales=False)
        self._materiales = copy.deepcopy(config.get('materiales', []) or [])
        self._refrescar_tabla_materiales()
        flm = config.get('filtro_lote_materiales', {}) or {}
        if hasattr(self, 'var_lote_mat_tipo') and flm:
            tipo_flm = flm.get('tipo', 'pasabajos')
            map_inv_lote = {
                'pasabajos': "Pasabajos (Butterworth)",
                'savitzky_golay': "Savitzky-Golay",
                'mediana_movil': "Mediana móvil",
                'splines': "Splines de suavizado",
            }
            self.var_lote_mat_tipo.set(map_inv_lote.get(tipo_flm, "Savitzky-Golay"))
            if 'pb_orden' in flm:
                self.var_lote_pb_ord.set(int(flm['pb_orden']))
            if 'pb_fcorte' in flm:
                self.var_lote_pb_fc.set(float(flm['pb_fcorte']))
            if 'sg_window' in flm:
                self.var_lote_sg_win.set(int(flm['sg_window']))
            if 'sg_polyorder' in flm:
                self.var_lote_sg_ord.set(int(flm['sg_polyorder']))
            if 'med_window' in flm:
                self.var_lote_med_win.set(int(flm['med_window']))
            if 'spline_s' in flm:
                self.var_lote_spline_s.set(float(flm['spline_s']))
            if 'modo_rango' in flm:
                self.var_lote_rango.set(flm['modo_rango'])
            if 'f1_ghz' in flm:
                self.var_lote_f1.set(float(flm['f1_ghz']))
            if 'f2_ghz' in flm:
                self.var_lote_f2.set(float(flm['f2_ghz']))
            self._al_cambiar_filtro_lote_mat()
        self.var_carpeta_salida.set(config.get('carpeta_salida', "") or "")
        self.var_nombre_pdf.set(config.get('nombre_informe_pdf', "informe_permitividad.pdf"))
        self.var_f_min.set(float(config.get('f_min_ghz', 0.5) or 0.5))
        self.var_f_max.set(float(config.get('f_max_ghz', 6.0) or 6.0))
        self.var_f_err_int1.set(float(config.get('f_err_int1_ghz', 2.0) or 2.0))
        self.var_f_err_int2.set(float(config.get('f_err_int2_ghz', 4.0) or 4.0))
        self.var_log_x.set(bool(config.get('escala_log_frecuencia', True)))
        fs_cfg = config.get('filtro_salida') or {}
        if hasattr(self, 'var_filtro_salida_activo'):
            self.var_filtro_salida_activo.set(bool(fs_cfg.get('habilitado', False)))
            self.var_filtro_salida_mostrar_original.set(bool(fs_cfg.get('mostrar_sin_filtro', True)))
            tipo_fs_inv = {
                'pasabajos': "Pasabajos (Butterworth)",
                'savitzky_golay': "Savitzky-Golay",
                'mediana_movil': "Mediana móvil",
                'splines': "Splines de suavizado",
            }
            self.var_filtro_salida_tipo.set(tipo_fs_inv.get(fs_cfg.get('tipo', 'savitzky_golay'), "Savitzky-Golay"))
            if 'pb_orden' in fs_cfg:
                self.var_salida_pb_ord.set(int(fs_cfg['pb_orden']))
            if 'pb_fcorte' in fs_cfg:
                self.var_salida_pb_fc.set(float(fs_cfg['pb_fcorte']))
            if 'sg_window' in fs_cfg:
                self.var_salida_sg_win.set(int(fs_cfg['sg_window']))
            if 'sg_polyorder' in fs_cfg:
                self.var_salida_sg_ord.set(int(fs_cfg['sg_polyorder']))
            if 'med_window' in fs_cfg:
                self.var_salida_med_win.set(int(fs_cfg['med_window']))
            if 'spline_s' in fs_cfg:
                self.var_salida_spline_s.set(float(fs_cfg['spline_s']))
            self.var_filtro_salida_rango.set(fs_cfg.get('modo_rango', 'todo'))
            if 'f1_ghz' in fs_cfg:
                self.var_salida_f1.set(float(fs_cfg['f1_ghz']))
            if 'f2_ghz' in fs_cfg:
                self.var_salida_f2.set(float(fs_cfg['f2_ghz']))
            self._al_cambiar_filtro_salida()
        if hasattr(self, 'var_limitar_eje_y'):
            self.var_limitar_eje_y.set(bool(config.get('limitar_eje_y', True)))
        self._curvas_comparar = copy.deepcopy(config.get('curvas_comparacion', []) or [])
        self._refrescar_tabla_comparar()
        self._graficar_comparacion_curvas()
        self._curvas_modelos = copy.deepcopy(config.get('curvas_modelos', []) or [])
        self._refrescar_tabla_modelos()
        self._graficar_modelos()
        # Por si la configuracion cargada trae un patron de calibracion con
        # un modelo sin ecuacion de relajacion.
        self._refrescar_cartel_calibracion()

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
        self._graficos_disponibles = {}
        self.combo_figuras.configure(values=[])
        self.var_figura_seleccionada.set("")
        self.visor_preview.limpiar("Corriendo el analisis...")

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
        """Arma el dict {etiqueta: funcion() -> Figure} de graficos
        disponibles para la vista previa interactiva, a partir de un
        resultado de `ejecutar_analisis` (usado tanto si termino OK como
        si se cancelo a mitad de camino, para no duplicar esta logica).
        Cada valor es una funcion SIN argumentos que reconstruye la
        Figure recien cuando se selecciona esa entrada en el combobox
        (ver _mostrar_preview_seleccionada), a partir de los datos crudos
        que ya vienen en el resultado ('datos_grafico'/'datos_grafico_Gn'
        -- ver `procesar_material` y `ejecutar_analisis` en
        analisis_permitividad.py) -- no hace falta releer ningun .png."""
        self._resultado_analisis = resultado
        limitar_y = bool(self.var_limitar_eje_y.get()) if hasattr(self, 'var_limitar_eje_y') else True
        graficos = {}
        chequeo = resultado.get('chequeo_calibracion') or {}
        if chequeo.get('datos_grafico'):
            etiqueta_p3 = chequeo.get('etiqueta_patron3') or "patron 3"
            d = chequeo['datos_grafico']
            graficos[f"Chequeo de calibracion ({etiqueta_p3})"] = (
                lambda d=d, ly=limitar_y: ap.graficar_comparacion(
                    d['frecs'], d['medido'], d['teorico'], d['titulo'], log_x=d['log_x'],
                    etiqueta_teorico=d.get('etiqueta_teorico', "Teorico (Debye)"),
                    limitar_eje_y=ly))
        chequeo_p4 = chequeo.get('patron4')
        if chequeo_p4 and chequeo_p4.get('datos_grafico'):
            etiqueta_p4 = chequeo.get('etiqueta_patron4') or "patron 4"
            d4 = chequeo_p4['datos_grafico']
            graficos[f"Chequeo de calibracion ({etiqueta_p4})"] = (
                lambda d4=d4, ly=limitar_y: ap.graficar_comparacion(
                    d4['frecs'], d4['medido'], d4['teorico'], d4['titulo'], log_x=d4['log_x'],
                    etiqueta_teorico=d4.get('etiqueta_teorico', "Teorico (Debye)"),
                    limitar_eje_y=ly))
        if chequeo.get('datos_grafico_Gn'):
            dg = chequeo['datos_grafico_Gn']
            graficos["Diagnostico: Gn(f) (calibracion)"] = (
                lambda dg=dg: ap.graficar_Gn(dg['frecs'], dg['Gn'], log_x=dg['log_x'],
                                              marcas_inicio=dg.get('marcas_inicio'),
                                              Gn_acondicionado=dg.get('Gn_acondicionado'),
                                              info_acond=dg.get('info_acond')))
        mc = chequeo.get('minimos_cuadrados')
        if mc and (mc.get('datos_grafico_residuos') or mc.get('datos_grafico')):
            dmc = mc.get('datos_grafico_residuos') or mc.get('datos_grafico')
            graficos["Cuadrados minimos: residuo de cada patron"] = (
                lambda dmc=dmc: ap.graficar_residuos_minimos_cuadrados(
                    dmc['frecs'], dmc['residuos'], log_x=dmc['log_x']))
        for r in resultado.get('resultados_materiales', []):
            if r.get('datos_grafico'):
                d = r['datos_grafico']
                graficos[r['nombre']] = (
                    lambda d=d, ly=limitar_y: ap.graficar_comparacion(
                        d['frecs'], d['medido'], d['teorico'], d['titulo'], log_x=d['log_x'],
                        etiqueta_teorico=d.get('etiqueta_teorico', "Teorico (Debye)"),
                        limitar_eje_y=ly))
        return graficos

    def _mostrar_figuras_disponibles(self, graficos):
        self._graficos_disponibles = graficos
        self.combo_figuras.configure(values=list(graficos.keys()))
        if graficos:
            primera = next(iter(graficos))
            self.var_figura_seleccionada.set(primera)
            self._mostrar_preview_seleccionada()
        else:
            self.var_figura_seleccionada.set("")
            self.visor_preview.limpiar("Todavia no hay resultados para mostrar.")

    def _al_terminar_analisis_ok(self, resultado):
        self._analisis_corriendo = False
        self.boton_ejecutar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self._status("Analisis terminado con exito.")
        self.var_paso_actual.set("Listo.")
        self._agregar_linea_consola("\n=== Analisis terminado con exito ===")
        self._ultima_carpeta_run = resultado.get('carpeta_salida_run')

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
        self._ultima_carpeta_run = resultado.get('carpeta_salida_run')

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
    # Vista previa de resultados (interactiva, ver VisorFigura)
    # -----------------------------------------------------------------
    def _mostrar_preview_seleccionada(self):
        etiqueta = self.var_figura_seleccionada.get()
        constructor = self._graficos_disponibles.get(etiqueta)
        if constructor is None:
            return
        try:
            fig = constructor()
        except Exception as exc:
            self.visor_preview.limpiar(f"No se pudo mostrar la vista previa:\n{exc}")
            return
        self.visor_preview.mostrar(fig)

    # -----------------------------------------------------------------
    # Abrir carpeta de salida
    # -----------------------------------------------------------------
    def _abrir_carpeta_salida(self):
        # Preferir la carpeta especifica de la ULTIMA corrida (con fecha y
        # hora, ver ejecutar_analisis) si ya se corrio algo en esta
        # sesion: es la que realmente le interesa al usuario, en vez de
        # la carpeta base (que ahora solo contiene subcarpetas por dia).
        if self._ultima_carpeta_run and os.path.isdir(self._ultima_carpeta_run):
            gf.abrir_carpeta(self._ultima_carpeta_run)
            return
        carpeta = self.var_carpeta_salida.get().strip()
        if not carpeta or not os.path.isdir(carpeta):
            messagebox.showinfo("Carpeta de salida",
                                 "La carpeta de salida todavia no existe (corre el analisis "
                                 "primero, o revisa la ruta).", parent=self)
            return
        gf.abrir_carpeta(carpeta)