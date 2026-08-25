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
  - La vista previa EN VIVO (pestañas "Vista previa S11" y "Salida y
    ejecucion") usa por separado FigureCanvasTkAgg/NavigationToolbar2Tk
    (ver gui_permitividad.VisorFigura), que si es interactivo -- pero
    esas Figures se crean y se embeben SOLO desde el hilo principal, asi
    que no chocan con el pipeline de analisis (que usa Figure()+Agg por
    su lado, en el hilo de fondo).

IMPORTANTE: por lo anterior, importa este modulo ANTES que cualquier
otra cosa toque matplotlib.pyplot en el proceso (gui_permitividad.py ya
lo hace en ese orden).

Sobre gc.disable() (mas abajo)
-------------------------------
Ademas de forzar el backend Agg, este modulo desactiva el recolector de
basura CICLICO automatico de Python (`gc.disable()`; el recuento de
referencias normal sigue funcionando igual, esto no es un memory leak
generalizado). Es una mitigacion a un bug real y bastante oscuro que
aparecio al agregar la vista previa interactiva (VisorFigura): si el
recolector ciclico se dispara automaticamente (por umbral de
asignaciones) mientras esta corriendo EN EL HILO DE FONDO -- algo que
puede pasar en cualquier momento, sin relacion directa con lo que ese
hilo esta haciendo -- y encuentra basura ciclica que incluye un
PIL.ImageTk.PhotoImage (los iconos de NavigationToolbar2Tk, creados en
el hilo principal), Python llama a su __del__ desde el hilo de fondo.
Como Tcl/Tk no es thread-safe, eso deja al interprete de Tcl en un
estado invalido y puede colgar la aplicacion de forma silenciosa (sin
excepcion visible: solo un "Exception ignored in: Image.__del__" en
stderr y el analisis nunca termina). Desactivar la recoleccion ciclica
automatica evita el problema de raiz: la memoria ciclica (tipicamente
poca, sobre todo callbacks de widgets) se va acumulando durante la
sesion en vez de liberarse sola, un costo aceptable para una app de
escritorio que se usa por sesiones acotadas.
"""
import gc
gc.disable()

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


def listar_modelos_teoricos_calibracion():
    """Igual que `listar_modelos_teoricos()`, pero sin la opcion "Ninguno":
    un patron de calibracion (a diferencia de un material) SIEMPRE tiene
    que resolver a un modelo teorico conocido, porque su permitividad
    entra directamente en las formulas de conversion de funciones.py."""
    return [(clave, etiqueta) for clave, etiqueta in listar_modelos_teoricos() if clave]


# ===========================================================================
# Configuracion: default, guardar, cargar, "recordar la ultima usada"
# ===========================================================================
CONFIG_VERSION = 2

# Donde se guarda el puntero a la ultima configuracion usada, para poder
# auto-cargarla la proxima vez que se abra la GUI (asi no hay que volver
# a cargar todo a mano cada vez).
_CARPETA_APP = Path.home() / ".permitividad_gui"
_ARCHIVO_ULTIMA_CONFIG = _CARPETA_APP / "ultima_config.txt"

# Ademas de "la ultima", se guarda un historial corto de configuraciones
# usadas (mas nueva primero) para el menu "Archivo > Abrir reciente" de
# la GUI. Es un archivo aparte de _ARCHIVO_ULTIMA_CONFIG (que sigue
# existiendo tal cual, para no romper el auto-cargado al abrir la app).
_ARCHIVO_RECIENTES = _CARPETA_APP / "recientes.json"
MAX_RECIENTES = 8


def config_default():
    """Config razonable para arrancar la GUI de cero, antes de cargar o
    guardar nada."""
    return {
        'version': CONFIG_VERSION,
        'carpeta_datos': "",
        'archivos_calibracion': {
            'corto': "",
            'aire': "",
            'patron3': "",
            'patron4': "",
        },
        # Por defecto, agua y alcohol isopropilico (el par tradicional de
        # la catedra) -- pero se puede cambiar por cualquier otro par de
        # liquidos con modelo teorico cargado en Patrones.PATRONES_TEORICOS,
        # sin tener que calibrar si o si con esos dos.
        'patron3_modelo': 'agua',
        'patron3_temperatura_c': 25.0,
        'patron4_modelo': 'alcohol_isopropilico',
        'patron4_temperatura_c': 25.0,
        'usar_metodo_completo': True,
        'estrategia_gn': 'minimo_gn',
        'materiales': [],
        'carpeta_salida': str(Path.cwd() / "salidas"),
        'nombre_informe_pdf': "informe_permitividad.pdf",
        'f_min_ghz': 0.5,
        'f_max_ghz': 6.0,
        'escala_log_frecuencia': True,
        'curvas_comparacion': [],
        'curvas_modelos': [],
    }


def _migrar_config_vieja(datos):
    """Migra, en el lugar, una configuracion con el esquema viejo (los
    patrones de calibracion fijos 'agua' y 'alc_isoprop', con sus propias
    claves 'temperatura_agua_c'/'temperatura_isoprop_cal_c') al esquema
    nuevo, generalizado a 'patron3'/'patron4' con un modelo teorico
    elegible (ver PATRON3_MODELO_CAL/PATRON4_MODELO_CAL en
    analisis_permitividad.py). No hace nada si `datos` ya esta en el
    esquema nuevo (no tiene ninguna clave vieja). Se llama sola desde
    `cargar_config`, asi que abrir una configuracion JSON guardada por una
    version anterior de la GUI sigue funcionando sin tener que editarla a
    mano ni perder los archivos/temperaturas ya cargados."""
    archivos = datos.get('archivos_calibracion') or {}
    es_esquema_viejo = (
        'temperatura_agua_c' in datos
        or 'temperatura_isoprop_cal_c' in datos
        or 'agua' in archivos
        or 'alc_isoprop' in archivos
    )
    if not es_esquema_viejo:
        return datos

    nuevos_archivos = dict(archivos)
    if 'agua' in nuevos_archivos:
        nuevos_archivos.setdefault('patron3', nuevos_archivos.pop('agua'))
    if 'alc_isoprop' in nuevos_archivos:
        nuevos_archivos.setdefault('patron4', nuevos_archivos.pop('alc_isoprop'))
    datos['archivos_calibracion'] = nuevos_archivos

    datos.setdefault('patron3_modelo', 'agua')
    datos.setdefault('patron4_modelo', 'alcohol_isopropilico')
    if 'temperatura_agua_c' in datos:
        datos['patron3_temperatura_c'] = datos.pop('temperatura_agua_c')
    if 'temperatura_isoprop_cal_c' in datos:
        datos['patron4_temperatura_c'] = datos.pop('temperatura_isoprop_cal_c')

    return datos


def guardar_config(ruta, config):
    """Guarda `config` como JSON legible en `ruta` y la recuerda como la
    'ultima configuracion usada'."""
    config = dict(config)
    config['version'] = CONFIG_VERSION
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    _recordar_ultima_config(ruta)


def cargar_config(ruta):
    """Carga una configuracion guardada, migrando el esquema viejo de
    calibracion si hiciera falta (ver `_migrar_config_vieja`) y
    completando con los defaults cualquier clave que falte (por si en el
    futuro se agregan campos nuevos y se abre una configuracion vieja)."""
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    datos = _migrar_config_vieja(datos)
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
    _recordar_reciente(ruta)


def ruta_ultima_config():
    """Ruta de la ultima configuracion guardada/cargada (para auto-
    cargarla al abrir la GUI), o None si no hay ninguna todavia o si el
    archivo referenciado ya no existe."""
    try:
        ruta = _ARCHIVO_ULTIMA_CONFIG.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return ruta if ruta and os.path.isfile(ruta) else None


def _cargar_lista_recientes():
    try:
        datos = json.loads(_ARCHIVO_RECIENTES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [r for r in datos if isinstance(r, str)] if isinstance(datos, list) else []


def _guardar_lista_recientes(rutas):
    try:
        _CARPETA_APP.mkdir(parents=True, exist_ok=True)
        _ARCHIVO_RECIENTES.write_text(json.dumps(rutas, indent=2), encoding="utf-8")
    except OSError:
        pass  # no es critico si esto falla


def _recordar_reciente(ruta):
    """Agrega `ruta` al principio del historial de configuraciones
    recientes (sin duplicados: si ya estaba, se mueve al principio),
    recortando la lista a MAX_RECIENTES. Se llama sola desde
    `_recordar_ultima_config`, tanto al guardar como al cargar una
    configuracion."""
    ruta = str(Path(ruta).resolve())
    recientes = [r for r in _cargar_lista_recientes() if r != ruta]
    recientes.insert(0, ruta)
    _guardar_lista_recientes(recientes[:MAX_RECIENTES])


def rutas_recientes():
    """Historial de configuraciones recientes (mas nueva primero), para
    poblar el menu 'Archivo > Abrir reciente' de la GUI. Filtra las que
    ya no existan en disco (por si se movieron/borraron), sin necesidad
    de que el usuario las limpie a mano."""
    return [r for r in _cargar_lista_recientes() if os.path.isfile(r)]


def limpiar_recientes():
    """Vacia el historial de configuraciones recientes."""
    _guardar_lista_recientes([])


# ===========================================================================
# Validacion previa (antes de correr el analisis) -- no bloqueante, solo
# junta avisos para mostrarle al usuario.
# ===========================================================================
_ETIQUETAS_CALIBRACION_FIJAS = {
    'corto': "Cortocircuito",
    'aire': "Aire",
}


def _etiqueta_calibracion_patron(config, clave_modelo, nombre_generico):
    """'Patron 3 (Agua)' o 'Patron 3 (modelo no definido)' si el modelo
    configurado no existe/no se eligio ninguno -- para mensajes de
    validacion mas claros que solo 'patron3'."""
    modelo = config.get(clave_modelo)
    if modelo and modelo in PATRONES_TEORICOS:
        return f"{nombre_generico} ({etiqueta_de_modelo(modelo)})"
    return f"{nombre_generico} (modelo no definido)"


def validar_config(config):
    """Devuelve una lista de strings con problemas encontrados (archivos
    vacios/inexistentes, modelo de calibracion invalido, rango de
    frecuencias invalido, etc.). Lista vacia = todo OK."""
    problemas = []
    carpeta = config.get('carpeta_datos', "") or ""
    archivos = config.get('archivos_calibracion', {})
    usar_completo = bool(config.get('usar_metodo_completo', True))

    etiquetas = dict(_ETIQUETAS_CALIBRACION_FIJAS)
    etiquetas['patron3'] = _etiqueta_calibracion_patron(config, 'patron3_modelo', "Patron 3")
    if usar_completo:
        etiquetas['patron4'] = _etiqueta_calibracion_patron(config, 'patron4_modelo', "Patron 4")

    for clave, etiqueta in etiquetas.items():
        ruta = archivos.get(clave, "") or ""
        if not ruta:
            problemas.append(f"Falta el archivo de calibracion '{etiqueta}'.")
        else:
            ruta_completa = os.path.join(carpeta, ruta)
            if not os.path.isfile(ruta_completa):
                problemas.append(f"No se encuentra el archivo de '{etiqueta}': {ruta_completa}")

    claves_modelo = [('patron3_modelo', "Patron 3")]
    if usar_completo:
        claves_modelo.append(('patron4_modelo', "Patron 4"))
    for clave_modelo, nombre in claves_modelo:
        modelo = config.get(clave_modelo)
        if not modelo or modelo not in PATRONES_TEORICOS:
            problemas.append(
                f"El modelo teorico de '{nombre}' ('{modelo}') no es valido. "
                f"Elegi uno de la lista en la pestaña de Calibracion.")

    if (usar_completo and config.get('patron3_modelo')
            and config.get('patron3_modelo') == config.get('patron4_modelo')):
        problemas.append(
            "Patron 3 y Patron 4 tienen el mismo modelo teorico asignado: tienen "
            "que ser dos liquidos distintos entre si para que la calibracion "
            "tenga solucion.")

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

    Devuelve (hilo, cola, evento_cancelar). La ventana debe "pollear" la
    cola periodicamente (p.ej. con root.after(100, ...)) y reaccionar a
    los eventos que va dejando:
        ('log', texto)                    -> agregar una linea a la consola
        ('progreso', paso, total, etiqueta) -> actualizar la barra de
                                        progreso y, si etiqueta no es
                                        None, tambien el texto de "paso
                                        actual"
        ('ok', resultado)                 -> termino bien (sin cancelar);
                                        `resultado` es el dict que
                                        devuelve ejecutar_analisis
        ('cancelado', resultado)          -> se pidio cancelar y el
                                        analisis se corto a mitad de
                                        camino; `resultado` tiene lo
                                        mismo que 'ok' pero con
                                        resultado['cancelado'] = True y
                                        (probablemente) menos materiales
        ('error', mensaje, detalle)       -> termino mal; mostrarle
                                        `mensaje` al usuario y loguear
                                        `detalle` (el traceback completo)

    `evento_cancelar` es un `threading.Event`: llamar a
    `evento_cancelar.set()` (desde el hilo principal, p.ej. al apretar un
    boton "Cancelar") le pide al analisis que se corte apenas termine de
    procesar el material que este calculando en ese momento -- nunca a
    mitad de un calculo. El analisis sigue generando un informe PDF con
    lo que ya se alcanzo a calcular antes de cortar.
    """
    cola = queue.Queue()
    evento_cancelar = threading.Event()

    def _log(texto):
        cola.put(('log', str(texto)))

    def _progreso(paso, total, etiqueta=None):
        cola.put(('progreso', paso, total, etiqueta))

    def _correr():
        redir = _RedirectorTexto(cola)
        try:
            with contextlib.redirect_stdout(redir):
                resultado = ap.ejecutar_analisis(
                    config, log=_log, progreso=_progreso,
                    cancelado=evento_cancelar.is_set)
            if resultado.get('cancelado'):
                cola.put(('cancelado', resultado))
            else:
                cola.put(('ok', resultado))
        except Exception as exc:  # se le muestra entero al usuario, mejor no filtrar
            detalle = traceback.format_exc()
            cola.put(('error', str(exc), detalle))

    hilo = threading.Thread(target=_correr, daemon=True)
    hilo.start()
    return hilo, cola, evento_cancelar


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
