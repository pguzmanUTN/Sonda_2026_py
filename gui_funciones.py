"""
gui_funciones.py
----------------
Logica "no visual" de la GUI de analisis de permitividad: manejo de
configuracion (guardar/cargar JSON, recordar la ultima usada), listado
de modelos teoricos disponibles, validacion previa, y ejecucion del
analisis en un hilo de fondo (para no congelar la ventana) con captura
de todo lo que se imprime por consola.

Este modulo fuerza el backend de matplotlib a 'Agg' (sin ventana propia)
ANTES de importar `analisis_permitividad` (que a su vez hace
`import matplotlib.pyplot as plt`). Esto es importante porque:

  - Los backends interactivos de matplotlib (TkAgg, etc.) no son
    thread-safe. Si el analisis corre en un hilo de fondo mientras el
    hilo principal esta corriendo el mainloop() de Tkinter, se puede
    colgar la aplicacion o crashear.
  - La GUI ya muestra las figuras generadas (los .png guardados) con su
    propio visor (ver gui_permitividad.py), asi que no hace falta que
    matplotlib abra ventanas propias.

IMPORTANTE: por lo anterior, importa este modulo ANTES que cualquier
otra cosa toque matplotlib.pyplot en el proceso (gui_permitividad.py ya
lo hace en ese orden).
"""
import matplotlib
matplotlib.use("Agg")

import os
import platform
import subprocess
import json
import queue
import threading
import traceback
import contextlib
from pathlib import Path

import analisis_permitividad as ap
from Patrones import PATRONES_TEORICOS, ETIQUETAS_PATRONES, etiqueta_patron


# ===========================================================================
# Modelos teoricos disponibles (para el combobox de "Modelo teorico")
# ===========================================================================
SIN_MODELO = ""  # clave interna para "Ninguno"


def listar_modelos_teoricos():
    """Lista de (clave, etiqueta) para poblar un combobox: empieza con
    "Ninguno" y sigue con los modelos de Patrones.py (los que tienen
    etiqueta linda definida en Patrones.ETIQUETAS_PATRONES, en ese orden;
    cualquier modelo nuevo que se agregue a Patrones.PATRONES_TEORICOS en
    el futuro aparece al final automaticamente, sin tener que tocar este
    archivo)."""
    opciones = [(SIN_MODELO, "Ninguno")]
    ya_listados = set()
    for clave, etiqueta in ETIQUETAS_PATRONES.items():
        if clave in PATRONES_TEORICOS:
            opciones.append((clave, etiqueta))
            ya_listados.add(clave)
    for clave in sorted(PATRONES_TEORICOS):
        if clave not in ya_listados:
            opciones.append((clave, clave))
    return opciones


def etiqueta_de_modelo(clave):
    """Clave interna -> etiqueta linda ('Ninguno' si es SIN_MODELO/None/
    falsy, o la clave tal cual si no tiene etiqueta linda definida).
    Delegado a Patrones.etiqueta_patron para que la GUI y el informe PDF
    (reporte_pdf.py, via analisis_permitividad.py) muestren siempre el
    mismo texto para el mismo modelo."""
    return etiqueta_patron(clave) or "Ninguno"


def clave_de_etiqueta(etiqueta):
    """Etiqueta linda -> clave interna (inversa de listar_modelos_teoricos)."""
    for clave, et in listar_modelos_teoricos():
        if et == etiqueta:
            return clave
    return SIN_MODELO


# ===========================================================================
# Configuracion: default, guardar, cargar, "recordar la ultima usada"
# ===========================================================================
CONFIG_VERSION = 1

# Donde se guarda el puntero a la ultima configuracion usada, para poder
# auto-cargarla la proxima vez que se abra la GUI (asi no hay que volver
# a cargar todo a mano cada vez).
_CARPETA_APP = Path.home() / ".permitividad_gui"
_ARCHIVO_ULTIMA_CONFIG = _CARPETA_APP / "ultima_config.txt"


def config_default():
    """Config razonable para arrancar la GUI de cero, antes de cargar o
    guardar nada."""
    return {
        'version': CONFIG_VERSION,
        'carpeta_datos': "",
        'archivos_calibracion': {
            'corto': "",
            'aire': "",
            'agua': "",
            'alc_isoprop': "",
        },
        'temperatura_agua_c': 25.0,
        'temperatura_isoprop_cal_c': 25.0,
        'materiales': [],
        'carpeta_salida': str(Path.cwd() / "salidas"),
        'nombre_informe_pdf': "informe_permitividad.pdf",
        'f_min_ghz': 0.5,
        'f_max_ghz': 6.0,
        'escala_log_frecuencia': True,
    }


def guardar_config(ruta, config):
    """Guarda `config` como JSON legible en `ruta` y la recuerda como la
    'ultima configuracion usada'."""
    config = dict(config)
    config['version'] = CONFIG_VERSION
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    _recordar_ultima_config(ruta)


def cargar_config(ruta):
    """Carga una configuracion guardada, completando con los defaults
    cualquier clave que falte (por si en el futuro se agregan campos
    nuevos y se abre una configuracion vieja)."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    config = config_default()
    config.update(datos)
    archivos = config_default()['archivos_calibracion']
    archivos.update(datos.get('archivos_calibracion', {}) or {})
    config['archivos_calibracion'] = archivos
    config.setdefault('materiales', [])
    _recordar_ultima_config(ruta)
    return config


def _recordar_ultima_config(ruta):
    try:
        _CARPETA_APP.mkdir(parents=True, exist_ok=True)
        _ARCHIVO_ULTIMA_CONFIG.write_text(str(Path(ruta).resolve()), encoding="utf-8")
    except OSError:
        pass  # no es critico si esto falla


def ruta_ultima_config():
    """Ruta de la ultima configuracion guardada/cargada (para auto-
    cargarla al abrir la GUI), o None si no hay ninguna todavia o si el
    archivo referenciado ya no existe."""
    try:
        ruta = _ARCHIVO_ULTIMA_CONFIG.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return ruta if ruta and os.path.isfile(ruta) else None


# ===========================================================================
# Validacion previa (antes de correr el analisis) -- no bloqueante, solo
# junta avisos para mostrarle al usuario.
# ===========================================================================
_ETIQUETAS_CALIBRACION = {
    'corto': "Cortocircuito",
    'aire': "Aire",
    'agua': "Agua",
    'alc_isoprop': "Patron adicional (metodo completo)",
}


def validar_config(config):
    """Devuelve una lista de strings con problemas encontrados (archivos
    vacios/inexistentes, rango de frecuencias invalido, etc.). Lista
    vacia = todo OK."""
    problemas = []
    carpeta = config.get('carpeta_datos', "") or ""
    archivos = config.get('archivos_calibracion', {})
    for clave, etiqueta in _ETIQUETAS_CALIBRACION.items():
        ruta = archivos.get(clave, "") or ""
        if not ruta:
            problemas.append(f"Falta el archivo de calibracion '{etiqueta}'.")
        else:
            ruta_completa = os.path.join(carpeta, ruta)
            if not os.path.isfile(ruta_completa):
                problemas.append(f"No se encuentra el archivo de '{etiqueta}': {ruta_completa}")

    if not config.get('materiales'):
        problemas.append("No hay materiales cargados para analizar.")
    for m in config.get('materiales', []):
        nombre = m.get('nombre') or "(sin nombre)"
        archivo = m.get('archivo') or ""
        if not archivo:
            problemas.append(f"El material '{nombre}' no tiene archivo asignado.")
        else:
            ruta_completa = os.path.join(carpeta, archivo)
            if not os.path.isfile(ruta_completa):
                problemas.append(f"No se encuentra el archivo del material '{nombre}': {ruta_completa}")

    if not config.get('carpeta_salida'):
        problemas.append("Falta indicar la carpeta de salida.")

    try:
        if float(config.get('f_min_ghz', 0)) >= float(config.get('f_max_ghz', 0)):
            problemas.append("El rango de frecuencias no es valido (F min tiene que ser menor que F max).")
    except (TypeError, ValueError):
        problemas.append("El rango de frecuencias no es un numero valido.")

    return problemas


# ===========================================================================
# Ejecucion en hilo de fondo, con log y progreso via queue.Queue
# ===========================================================================
class _RedirectorTexto:
    """Stream minimo (solo write/flush) que manda todo lo que se le
    escribe (p.ej. via print) a una queue.Queue, linea por linea, para
    que el hilo principal de Tkinter lo muestre en la consola de la
    GUI."""

    def __init__(self, cola):
        self._cola = cola
        self._buffer = ""

    def write(self, texto):
        self._buffer += texto
        while "\n" in self._buffer:
            linea, self._buffer = self._buffer.split("\n", 1)
            self._cola.put(('log', linea))

    def flush(self):
        pass


def lanzar_analisis_en_hilo(config):
    """
    Arranca `analisis_permitividad.ejecutar_analisis(config)` en un hilo
    de fondo (daemon), capturando todo el stdout (prints de
    analisis_permitividad.py, funciones.py, etc.) y reportando el
    progreso, todo a traves de una queue.Queue.

    Devuelve (hilo, cola). La ventana debe "pollear" la cola
    periodicamente (p.ej. con root.after(100, ...)) y reaccionar a los
    eventos que va dejando:
        ('log', texto)              -> agregar una linea a la consola
        ('progreso', paso, total)   -> actualizar la barra de progreso
        ('ok', resultado)           -> termino bien; `resultado` es el
                                        dict que devuelve ejecutar_analisis
        ('error', mensaje, detalle) -> termino mal; mostrarle `mensaje`
                                        al usuario y loguear `detalle`
                                        (el traceback completo)
    """
    cola = queue.Queue()

    def _log(texto):
        cola.put(('log', str(texto)))

    def _progreso(paso, total):
        cola.put(('progreso', paso, total))

    def _correr():
        redir = _RedirectorTexto(cola)
        try:
            with contextlib.redirect_stdout(redir):
                resultado = ap.ejecutar_analisis(config, log=_log, progreso=_progreso)
            cola.put(('ok', resultado))
        except Exception as exc:  # se le muestra entero al usuario, mejor no filtrar
            detalle = traceback.format_exc()
            cola.put(('error', str(exc), detalle))

    hilo = threading.Thread(target=_correr, daemon=True)
    hilo.start()
    return hilo, cola


# ===========================================================================
# Utilidad: abrir una carpeta con el explorador de archivos del sistema
# ===========================================================================
def abrir_carpeta(ruta):
    """Abre `ruta` en el explorador de archivos del sistema operativo
    (Explorador en Windows, Finder en macOS, el gestor de archivos que
    corresponda en Linux). No hace nada (silenciosamente) si falla."""
    ruta = os.path.abspath(ruta)
    sistema = platform.system()
    try:
        if sistema == "Windows":
            os.startfile(ruta)  # type: ignore[attr-defined]
        elif sistema == "Darwin":
            subprocess.run(["open", ruta], check=False)
        else:
            subprocess.run(["xdg-open", ruta], check=False)
    except OSError:
        pass
