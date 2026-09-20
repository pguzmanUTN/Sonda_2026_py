"""
analisis_permitividad.py
-------------------------
Script principal del proyecto: lee los archivos .s1p medidos con el VNA
(sonda coaxial open-ended de 4 mm), calcula la permitividad relativa
compleja de los materiales de interes con los dos metodos de conversion
implementados en `funciones.py`, y grafica el resultado medido contra la
curva teorica correspondiente (si se conoce; si no, grafica solo lo medido).

Como se usa
-----------
1. Poner los archivos .s1p de calibracion (corto, aire, patron3, patron4)
   en la misma carpeta que este script, y completar ARCHIVOS_CALIBRACION
   mas abajo con sus nombres. "Patron3" y "patron4" son, por defecto,
   agua destilada y alcohol isopropilico (el par tradicional de la
   catedra), pero pueden ser CUALQUIER PAR de liquidos con un modelo
   teorico cargado en Patrones.PATRONES_TEORICOS -- ver PATRON3_MODELO_CAL
   / PATRON4_MODELO_CAL mas abajo.
2. Ajustar PATRON3_TEMPERATURA_CAL_C y PATRON4_TEMPERATURA_CAL_C a la
   temperatura real de esos 2 liquidos durante la calibracion (afecta la
   precision de los dos metodos de conversion).
3. Agregar cada material que se quiera analizar a la lista MATERIALES
   (ver mas abajo). Para un material sin modelo teorico conocido, dejar
   'modelo': None -- el script va a graficar unicamente lo medido, sin
   intentar calcular error.
4. Correr:  python analisis_permitividad.py

Los resultados de cada corrida quedan en
'<CARPETA_SALIDA>/<AAAA-MM-DD>/<HH-MM-SS>/' (una subcarpeta por dia y por
hora, para no pisar corridas anteriores), y los .s1p de entrada se copian
a la carpeta del dia como registro de con que mediciones se genero cada
informe (ver el docstring de `ejecutar_analisis`).

Como agregar una medicion nueva
--------------------------------
Alcanza con sumar un diccionario a la lista MATERIALES, no hace falta
tocar nada mas del script:

    {
        'nombre': "ResinaN",
        'archivo': "ResinaN.s1p",
        'modelo': None,   # no hay modelo teorico cargado en Patrones.py
    }

Si en el futuro se agrega un modelo teorico para ese material en
`Patrones.py` (agregandolo a PATRONES_TEORICOS), alcanza con poner esa
clave en 'modelo' en lugar de None y el script automaticamente empieza a
graficar la comparacion y el error.

Modelos teoricos SIN ecuacion de relajacion
--------------------------------------------
Ademas de los liquidos con Debye/Davidson-Cole, `Patrones.py` trae ahora
tres modelos de permitividad ESTATICA (constante en frecuencia), tomados
de la tabla de celda de admitancia de la Seccion 6 del reporte NPL
MAT 23: 'acetona', 'ciclohexano' y 'silicona'. Se usan igual que
cualquier otro ('modelo': 'acetona', mas su 'temperatura'), pero tienen
limitaciones importantes que conviene leer antes de sacar conclusiones:

  - 'acetona'     : la relajacion existe pero queda muy por encima de los
                    5 GHz medidos por NPL, asi que no hay Debye que
                    ajustar. er' es una aproximacion razonable en la banda
                    de este proyecto; er'' se devuelve como 0 y NO es
                    cierto (la acetona si tiene perdidas en GHz).
  - 'ciclohexano' : molecula no polar, sin relajacion dipolar. Aca la
                    constante es el comportamiento real, y er'' ~ 0
                    tambien es correcto.
  - 'silicona'    : idem ciclohexano (Dow-Corning 200, 1 cSt). Ojo: NPL
                    solo lo midio entre 5 y 35 °C.

El script imprime la advertencia completa de cada uno en consola al
procesar el material, y el informe PDF la incluye como caja destacada
junto al grafico. Donde el modelo de referencia no modela perdidas, la
columna de error relativo de er'' aparece como "n/a" en vez de un numero
sin sentido.

Patrones que dependen de la temperatura
----------------------------------------
Desde que Patrones.py soporta varias temperaturas (NPL Report MAT 23),
cada funcion teorica acepta `get_er_pat_xxx(frecs, T=...)`. Como el campo
'modelo' de MATERIALES (y de PATRON3_MODELO_CAL/PATRON4_MODELO_CAL) tiene
que ser una clave de texto (no la funcion ya resuelta), la temperatura se
"ata" con `functools.partial` en `resolver_modelo_teorico` (o, para los
patrones de calibracion, evaluando directamente
`PATRONES_TEORICOS[modelo](frecs, T=...)` en `ejecutar_analisis`). Para
medir a otra temperatura, alcanza con cambiar la constante correspondiente
(no hace falta tocar Patrones.py). Si la temperatura real no cae justo en
un escalon de 5 °C tabulado en el reporte, la funcion usa el mas cercano
disponible (y avisa con un warning si ademas cae fuera del rango
recomendado).
"""
import os
import re
import csv
import shutil
import warnings
import functools
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from Touchstone import (leer_s1p, mismas_frecuencias, resamplear,
                         graficar_s11_mag_fase, graficar_smith)
from funciones import get_er_DUTm, get_er_DUT_completo, error_relativo_porcentual, calcular_Gn
from Patrones import (PATRONES_TEORICOS, etiqueta_patron, advertencia_patron,
                       nivel_advertencia_patron, es_modelo_estatico,
                       modela_perdidas, resumen_corto_patron)
from reporte_pdf import generar_reporte_pdf

# ---------------------------------------------------------------------------
# CONFIGURACION -- editar segun corresponda
# ---------------------------------------------------------------------------
CARPETA_DATOS = "."   # carpeta donde estan los .s1p

# Patrones de calibracion: estos 4 son siempre necesarios (el metodo
# completo usa los 4; el simplificado usa corto/aire/patron3).
#
# 'patron3' y 'patron4' son dos liquidos con permitividad conocida
# (tradicionalmente agua destilada y alcohol isopropilico, ver
# PATRON3_MODELO_CAL/PATRON4_MODELO_CAL mas abajo), pero pueden ser
# CUALQUIER PAR de liquidos con un modelo teorico cargado en
# Patrones.PATRONES_TEORICOS -- no hace falta calibrar si o si con esos
# dos. El unico requisito real es que sean dos liquidos DISTINTOS entre
# si (y del aire), para que el sistema de ecuaciones tenga solucion.
ARCHIVOS_CALIBRACION = {
    'corto':   "sonda4-short.s1p",
    'aire':    "sonda4-aire.s1p",
    'patron3': "sonda4-agua.s1p",
    'patron4': "sonda4-alc-isoprop.s1p",
}


# ARCHIVOS = {
#     'corto':       "sonda4-short2.s1p",
#     'aire':        "sonda4-aire2.s1p",
#     'agua':        "sonda4-agua2.s1p",
#     'alc_etilico': "sonda4-alc-etilico.s1p",
#     'alc_isoprop': "sonda4-alc-isoprop2.s1p",
# }

# ARCHIVOS = {
#     'corto':       "sonda4-short.s1p",
#     'aire':        "sonda4-aire.s1p",
#     'agua':        "sonda4-agua.s1p",
#     'alc_etilico': "sonda4-alc-etilico.s1p",
#     'alc_isoprop': "sonda4-alc-isoprop.s1p",
# }

# Modelo teorico (clave de Patrones.PATRONES_TEORICOS) y temperatura real
# de cada uno de los 2 liquidos usados como patron 3 y patron 4 durante
# ESTA calibracion. Por defecto agua y alcohol isopropilico (el par
# tradicional de la catedra), pero se puede poner cualquier otro par de
# liquidos disponibles en Patrones.py (ver ETIQUETAS_PATRONES para la
# lista) sin tocar funciones.py ni el resto del pipeline.
PATRON3_MODELO_CAL = 'agua'
PATRON3_TEMPERATURA_CAL_C = 25.0

# La temperatura del patron 4 entra en el calculo de Gn (funciones.
# calcular_Gn), asi que afecta el resultado de TODOS los materiales
# analizados con el metodo completo, no solo el del propio patron 4.
PATRON4_MODELO_CAL = 'alcohol_isopropilico'
PATRON4_TEMPERATURA_CAL_C = 25.0

TEMPERATURA_ALC_ETILICO_C = 20.0
TEMPERATURA_ALC_ISOPROP_C = 25.0
TEMPERATURA_METANOL_C = 25.0
TEMPERATURA_DMSO_C = 25.0
TEMPERATURA_ETILENGLICOL_C = 25.0
TEMPERATURA_BUTANOL_C = 25.0
TEMPERATURA_PROPANOL_C = 25.0
# Liquidos con modelo de permitividad ESTATICA (sin ecuacion de Debye):
# ver la seccion "Modelos teoricos SIN ecuacion de relajacion" del
# docstring de arriba. Los rangos de temperatura medidos por NPL NO son
# los mismos que los de los liquidos con Debye: acetona 5-50 °C,
# ciclohexano 10-50 °C y silicona 5-35 °C solamente.
TEMPERATURA_ACETONA_C = 25.0
TEMPERATURA_CICLOHEXANO_C = 25.0
TEMPERATURA_SILICONA_C = 25.0

# Lista de materiales a analizar. PARA AGREGAR UNA MEDICION NUEVA, sumar un
# diccionario aca -- no hace falta tocar el resto del script.
#
# Claves de cada entrada:
#   'nombre'      : como se muestra en graficos/reportes/nombres de archivo.
#   'archivo'     : nombre del .s1p (dentro de CARPETA_DATOS).
#   'modelo'      : clave de Patrones.PATRONES_TEORICOS (p.ej.
#                   "alcohol_etilico"), o None/"" si no se conoce un
#                   modelo teorico (en ese caso se grafica solo lo medido,
#                   sin comparacion ni calculo de error).
#   'temperatura' : temperatura real (°C) de ese material durante el
#                   ensayo. Se ignora si 'modelo' es None. No hace falta
#                   que caiga justo en un escalon tabulado de 5°C: cada
#                   funcion de Patrones.py usa el mas cercano disponible.
#   'metodos'     : opcional, lista con 'simplificado' y/o 'completo'
#                   (default: los dos). Usar solo 'simplificado' para un
#                   material que ya es patron del metodo completo (como
#                   el alcohol isopropilico), porque ahi el metodo
#                   completo daria una comparacion circular sin sentido.
#
# ('modelo'+'temperatura' en vez de guardar la funcion ya resuelta como
# antes, para que esta lista sea puro texto/numeros: asi se puede guardar
# y cargar tal cual desde la GUI como un JSON de configuracion.)
def resolver_modelo_teorico(modelo, temperatura):
    """Convierte una clave de Patrones.PATRONES_TEORICOS (o None/"") mas
    una temperatura en la funcion(frecs) -> permitividad compleja que
    espera `procesar_material`. Devuelve None si no hay modelo."""
    if not modelo:
        return None
    if modelo not in PATRONES_TEORICOS:
        raise ValueError(
            f"Modelo teorico desconocido: '{modelo}'. Opciones validas: "
            f"{sorted(PATRONES_TEORICOS)}"
        )
    return functools.partial(PATRONES_TEORICOS[modelo], T=float(temperatura))


MATERIALES = [
    {
        'nombre': "Alcohol etilico",
        'archivo': "sonda4-alc-etilico.s1p",
        'modelo': "alcohol_etilico",
        'temperatura': TEMPERATURA_ALC_ETILICO_C,
    },
    {
        'nombre': "Alcohol isopropilico",
        'archivo': "sonda4-alc-isoprop.s1p",
        'modelo': "alcohol_isopropilico",
        'temperatura': TEMPERATURA_ALC_ISOPROP_C,
    },
    # {
    #     'nombre': "Acetona",
    #     'archivo': "sonda4-acetona.s1p",
    #     'modelo': "acetona",
    #     'temperatura': TEMPERATURA_ACETONA_C,
    # },
    # {
    #     'nombre': "Ciclohexano",
    #     'archivo': "sonda4-ciclohexano.s1p",
    #     'modelo': "ciclohexano",
    #     'temperatura': TEMPERATURA_CICLOHEXANO_C,
    # },
    # {
    #     'nombre': "Fluido de silicona 1 cSt",
    #     'archivo': "sonda4-silicona.s1p",
    #     'modelo': "silicona",
    #     'temperatura': TEMPERATURA_SILICONA_C,
    # },
    # {
    #     'nombre': "ResinaN",
    #     'archivo': "ResinaN.s1p",
    #     'modelo': None,
    # },
    # Nuevos patrones disponibles en Patrones.py (NPL MAT 23). Descomentar
    # y ajustar 'archivo' (y la temperatura correspondiente, mas arriba)
    # apenas se tenga el .s1p medido de cada uno:
    # {
    #     'nombre': "Metanol",
    #     'archivo': "sonda4-metanol.s1p",
    #     'modelo': "metanol",
    #     'temperatura': TEMPERATURA_METANOL_C,
    # },
    # {
    #     'nombre': "DMSO",
    #     'archivo': "sonda4-dmso.s1p",
    #     'modelo': "dmso",
    #     'temperatura': TEMPERATURA_DMSO_C,
    # },
    # {
    #     'nombre': "Etilenglicol",
    #     'archivo': "sonda4-etilenglicol.s1p",
    #     'modelo': "etilenglicol",
    #     'temperatura': TEMPERATURA_ETILENGLICOL_C,
    # },
    # {
    #     'nombre': "1-Butanol",
    #     'archivo': "sonda4-butanol.s1p",
    #     'modelo': "butanol",
    #     'temperatura': TEMPERATURA_BUTANOL_C,
    # },
    # {
    #     'nombre': "1-Propanol",
    #     'archivo': "sonda4-propanol.s1p",
    #     'modelo': "propanol",
    #     'temperatura': TEMPERATURA_PROPANOL_C,
    # },
]

# Rango de frecuencias de interes del proyecto (500 MHz - 6 GHz)
F_MIN_GHZ, F_MAX_GHZ = 0.5, 6.0

# Si el barrido cubre varias decadas (p.ej. MHz a GHz), conviene graficar
# el eje de frecuencias en escala logaritmica -- si no, todo lo que pasa
# por debajo de un par de cientos de MHz queda invisible, apretado contra
# el borde izquierdo del grafico en escala lineal. Se puede desactivar
# (False) si el rango analizado es mas bien estrecho / de una sola decada.
ESCALA_LOG_FRECUENCIA = True

CARPETA_SALIDA = "./salidas"

# Nombre del PDF final con graficos + tablas (se genera siempre que corra
# el script; queda en CARPETA_SALIDA).
NOMBRE_INFORME_PDF = "informe_permitividad.pdf"


# ---------------------------------------------------------------------------
def _slug(nombre):
    """Convierte un nombre legible en un nombre de archivo simple
    (sin espacios ni acentos), para usar en los .png/.csv de salida."""
    s = nombre.lower()
    reemplazos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n'}
    for a, b in reemplazos.items():
        s = s.replace(a, b)
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s


def _cargar_cache(ruta, cache):
    """Lee un .s1p (usando `cache` para no releer el mismo archivo dos
    veces si se referencia en mas de un lugar, p.ej. el alcohol
    isopropilico que es patron de calibracion Y material analizado)."""
    if ruta not in cache:
        if not os.path.isfile(ruta):
            raise FileNotFoundError(
                f"No se encontro '{ruta}'. Revisa CARPETA_DATOS/ARCHIVOS_"
                f"CALIBRACION/MATERIALES, o subi el archivo correspondiente."
            )
        cache[ruta] = leer_s1p(ruta)
    return cache[ruta]


def cargar_calibracion(carpeta, archivos, cache):
    """Lee los 4 patrones de calibracion y los deja alineados en la misma
    grilla de frecuencias (remuestrea contra 'patron3' si hiciera falta)."""
    datos = {}
    for clave, nombre in archivos.items():
        ruta = os.path.join(carpeta, nombre)
        datos[clave] = _cargar_cache(ruta, cache)
        print(f"  {clave:12s} <- {nombre}  "
              f"({len(datos[clave]['Frec'])} puntos, "
              f"{datos[clave]['Frec'][0]/1e9:.3f}-{datos[clave]['Frec'][-1]/1e9:.3f} GHz)")

    referencia = datos['patron3']
    if not mismas_frecuencias(*datos.values()):
        print("  [info] los patrones de calibracion no comparten exactamente "
              "la misma grilla de frecuencias: remuestreando todo a la "
              "grilla de 'patron3'.")
        for clave in datos:
            if clave != 'patron3':
                datos[clave] = resamplear(referencia['Frec'], datos[clave])

    return datos


def _copiar_insumos(carpeta_datos, archivos_calibracion, materiales, carpeta_destino, log):
    """
    Copia los .s1p de ENTRADA (los 4 patrones de calibracion + el archivo
    de cada material) a `carpeta_destino` -- la carpeta del DIA de esta
    corrida, ver `ejecutar_analisis` -- para dejar un registro de con que
    mediciones exactas se genero cada informe, sin depender de que la
    carpeta de datos original no se toque despues.

    Se copian a la carpeta del DIA (compartida entre todas las corridas
    de ese dia) y no a la de la hora (especifica de esta corrida), para
    no duplicar el mismo archivo de entrada cada vez que se vuelve a
    correr el analisis con la misma medicion.

    No hace fallar el analisis si un archivo puntual no se puede copiar
    (p.ej. problema de permisos): lo avisa por `log` y sigue con el resto.
    """
    os.makedirs(carpeta_destino, exist_ok=True)
    nombres = list(archivos_calibracion.values()) + \
        [m['archivo'] for m in materiales if m.get('archivo')]
    copiados = 0
    for nombre in nombres:
        origen = os.path.join(carpeta_datos, nombre)
        destino = os.path.join(carpeta_destino, os.path.basename(nombre))
        if not os.path.isfile(origen):
            continue  # si no existe, ya se avisa aparte al intentar leerlo
        if os.path.abspath(origen) == os.path.abspath(destino):
            continue  # ya esta en la carpeta de destino (raro, pero por las dudas)
        try:
            shutil.copy2(origen, destino)
            copiados += 1
        except OSError as exc:
            log(f"  [aviso] no se pudo copiar '{origen}' a la carpeta del dia: {exc}")
    if copiados:
        log(f"  {copiados} archivo(s) de entrada copiado(s) a: {os.path.abspath(carpeta_destino)}")


def graficar_comparacion(frecs, er_medido_dict, er_teorico, titulo, archivo_salida=None,
                          log_x=True, etiqueta_teorico="Teorico (Debye)"):
    """
    Grafica parte real e imaginaria: curva medida (una o varias) y, si se
    conoce, la curva teorica de referencia.

    er_medido_dict : dict {etiqueta: ndarray complejo}
    er_teorico     : ndarray complejo, o None si no hay modelo teorico
                      (en ese caso se grafica solo lo medido).
    archivo_salida : str, opcional
        Si se pasa, guarda la figura ahi (dpi=150) y devuelve esa ruta.
        Si se omite (None, default), no guarda nada y devuelve la Figure
        de matplotlib abierta (construida con la API orientada a objetos,
        `Figure()` directo -- ver la nota en
        `Touchstone.graficar_s11_mag_fase`), para poder embeberla en vivo
        en la GUI con `gui_permitividad.VisorFigura`.
    log_x : bool
        Si es True (default), el eje de frecuencias se grafica en escala
        logaritmica. Muy recomendable cuando el barrido cubre varias
        decadas (p.ej. 1 MHz a 2 GHz): en escala lineal, todo lo que pasa
        por debajo de ~10-20% de F max queda comprimido contra el borde
        izquierdo del grafico y no se distingue nada (un barrido de
        1 MHz a 2 GHz en lineal deja practicamente todo el rango audible/
        HF/VHF apretado en un puñado de pixeles). En log, cada decada
        ocupa el mismo ancho visual. Se ignora (cae a lineal) si `frecs`
        tiene algun valor <= 0, porque el logaritmo no esta definido ahi.
    etiqueta_teorico : str
        Texto de la leyenda de la curva teorica. NO siempre es un Debye:
        los modelos de permitividad estatica de `Patrones.py` (acetona,
        ciclohexano, silicona) no tienen ecuacion de relajacion, asi que
        etiquetarlos como "Debye" seria engañoso. `procesar_material`
        pasa la etiqueta que corresponde a cada modelo.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 7))
    ax_re, ax_im = fig.subplots(2, 1, sharex=True)

    if er_teorico is not None:
        ax_re.plot(frecs / 1e9, er_teorico.real, 'k--', linewidth=2, label=etiqueta_teorico)
    for etiqueta, er in er_medido_dict.items():
        ax_re.plot(frecs / 1e9, er.real, linewidth=1.6, label=etiqueta)
    ax_re.set_ylabel("er'  (parte real)")
    ax_re.set_title(titulo)
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_re.legend()

    if er_teorico is not None:
        ax_im.plot(frecs / 1e9, er_teorico.imag, 'k--', linewidth=2, label=etiqueta_teorico)
    for etiqueta, er in er_medido_dict.items():
        ax_im.plot(frecs / 1e9, er.imag, linewidth=1.6, label=etiqueta)
    ax_im.set_xlabel("Frecuencia (GHz)")
    ax_im.set_ylabel("er''  (perdidas, Im)")
    ax_im.minorticks_on()
    ax_im.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_im.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_im.legend()

    if log_x and frecs.size > 0 and np.all(frecs > 0):
        ax_re.set_xscale('log')
        # sharex=True ya deberia propagar la escala de ax_re a ax_im, pero
        # se deja explicito por las dudas (p.ej. si en algun momento se
        # deja de compartir eje entre los dos subplots).
        ax_im.set_xscale('log')
        # Con escala log, las grillas menores (2,3,4...x10^n) ayudan mucho
        # a ubicarse; minorticks_on() ya las habilita arriba.

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        print(f"  Figura guardada: {archivo_salida}")
        return archivo_salida
    return fig


def graficar_Gn(frecs, Gn, archivo_salida=None, log_x=True, marcas_inicio=None):
    """
    Grafica la conductancia normalizada Gn(f) calculada a partir de los 4
    patrones de calibracion (ver `funciones.calcular_Gn`), en modulo y
    fase (mismo estilo que `Touchstone.graficar_s11_mag_fase`). Es un
    diagnostico de la CALIBRACION, no de un material en particular: Gn no
    depende del DUT, asi que esta misma curva aplica a todos los
    materiales analizados con el metodo completo en esta corrida.

    Al estar relacionada con G0/(j*w*C0) -- una propiedad fisica continua
    de la sonda -- Gn(f) deberia verse suave tanto en modulo como en
    fase. Un salto brusco o un pico aislado suele indicar un problema con
    la medicion de alguno de los 4 patrones, tipicamente el patron 4 (el
    unico que entra en el calculo de Gn y en ningun otro lado del metodo
    simplificado, por lo que un problema ahi puede pasar desapercibido si
    solo se mira el chequeo de calibracion del patron 3).

    marcas_inicio : dict {etiqueta: frecuencia_hz}, opcional
        Dibuja una linea vertical punteada en cada frecuencia indicada,
        marcando donde arranca la "marcha en frecuencia" del metodo
        completo con cada estrategia (ver `funciones.get_er_DUT_completo`,
        parametro `estrategia_semilla`) -- para ver de un vistazo si el
        punto de arranque cae en una zona con |Gn| chico (deseable) o no,
        sin tener que leer el log de texto. Si son 2 marcas (una por
        estrategia, cuando se esta comparando "automatico" vs. "clasico"),
        quedan una al lado de la otra con colores distintos.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 6))
    ax_mag, ax_fase = fig.subplots(2, 1, sharex=True)

    mag_Gn = np.abs(Gn)
    # np.angle() sola da la fase "envuelta" en (-180, 180]; con np.unwrap
    # se corrigen los saltos artificiales de +-360 grados, igual que se
    # hace con la fase de S11 en Touchstone.graficar_s11_mag_fase.
    fase_Gn = np.degrees(np.unwrap(np.angle(Gn)))

    tiene_marcas = bool(marcas_inicio)
    ax_mag.plot(frecs / 1e9, mag_Gn, linewidth=1.6, color='tab:purple',
                label="|Gn(f)|" if tiene_marcas else None, zorder=3)
    ax_mag.set_ylabel("|Gn|")
    ax_mag.set_title("Conductancia normalizada Gn(f) \u2014 diagnostico de calibracion")
    ax_mag.minorticks_on()
    ax_mag.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_mag.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    ax_fase.plot(frecs / 1e9, fase_Gn, linewidth=1.6, color='tab:purple',
                 label="Fase Gn(f)" if tiene_marcas else None, zorder=3)
    ax_fase.set_xlabel("Frecuencia (GHz)")
    ax_fase.set_ylabel("Fase Gn (\u00b0)")
    ax_fase.minorticks_on()
    # Sin esto, cuando la fase varia muy poco en todo el barrido (queda
    # casi constante), matplotlib intenta ser "util" agregando una
    # notacion de offset+escala en la esquina que queda superpuesta y
    # dificil de leer (p.ej. "1e-11-9e1"). Se fuerza un formato de numero
    # llano con 2 decimales (de sobra para fase en grados) en vez de
    # offset automatico o precision de punto flotante completa.
    ax_fase.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_fase.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_fase.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    if marcas_inicio:
        colores_marca = ['#d62728', '#2ca02c', '#ff7f0e']
        for i, (etiqueta, f_hz) in enumerate(marcas_inicio.items()):
            color = colores_marca[i % len(colores_marca)]
            f_ghz = f_hz / 1e9
            ax_mag.axvline(f_ghz, color=color, linestyle='--', linewidth=1.4,
                            label=f"Arranque: {etiqueta}", zorder=2)
            ax_fase.axvline(f_ghz, color=color, linestyle='--', linewidth=1.4,
                             label=f"Arranque: {etiqueta}", zorder=2)
        ax_mag.legend(fontsize=8)
        ax_fase.legend(fontsize=8)

    if log_x and frecs.size > 0 and np.all(frecs > 0):
        ax_mag.set_xscale('log')
        ax_fase.set_xscale('log')

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        print(f"  Figura guardada: {archivo_salida}")
        return archivo_salida
    return fig


def calcular_error(er_medido, er_teorico):
    """Devuelve un dict con el error relativo (%) promedio y maximo de
    parte real e imaginaria, listo para tablas (consola o PDF).

    Usa nanmean/nanmax porque `error_relativo_porcentual` devuelve NaN
    donde el error no esta DEFINIDO -- tipicamente cuando la curva
    teorica tiene er'' identicamente 0, que es lo que pasa con los
    modelos de permitividad estatica de `Patrones.py` (acetona,
    ciclohexano, silicona). Si TODOS los puntos de una parte son NaN, el
    promedio queda en NaN y quien lo muestre debe imprimir "n/a" en vez
    de un numero (ver `reportar_error` y `reporte_pdf._fmt_error`), en
    lugar de inventar un 0 o un inf.
    """
    err_re, err_im = error_relativo_porcentual(er_medido, er_teorico)

    def _resumen(arr):
        arr = np.abs(np.asarray(arr, dtype=float))
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float('nan'), float('nan')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return float(np.nanmean(arr)), float(np.nanmax(arr))

    re_medio, re_max = _resumen(err_re)
    im_medio, im_max = _resumen(err_im)
    return {
        'err_re_medio': re_medio,
        'err_re_max': re_max,
        'err_im_medio': im_medio,
        'err_im_max': im_max,
    }


def _fmt_pct(valor):
    """Formatea un porcentaje de error, o 'n/a' si es NaN (error no
    definido -- ver `calcular_error`)."""
    return "    n/a" if valor is None or np.isnan(valor) else f"{valor:6.2f}%"


def reportar_error(nombre, er_medido, er_teorico):
    """Imprime el error relativo en consola y lo devuelve (ver `calcular_error`)."""
    e = calcular_error(er_medido, er_teorico)
    print(f"  {nombre}:")
    print(f"    error medio  er' = {_fmt_pct(e['err_re_medio'])}   "
          f"error medio  er'' = {_fmt_pct(e['err_im_medio'])}")
    print(f"    error maximo er' = {_fmt_pct(e['err_re_max'])}   "
          f"error maximo er'' = {_fmt_pct(e['err_im_max'])}")
    if np.isnan(e['err_im_medio']):
        print("    (er'' = n/a: el modelo teorico de referencia no modela "
              "perdidas, ver la advertencia de ese modelo)")
    return e


def muestrear_en_frecuencias(frecs, columnas, puntos_ghz=(0.5, 0.9, 1.8, 2.45, 3.5, 5.8, 6.0)):
    """Toma, de `columnas` (dict {titulo: ndarray complejo} evaluado en
    `frecs`), el valor mas cercano a cada frecuencia de `puntos_ghz`.
    Devuelve (frecs_ghz, columnas_muestreadas) listos para tabla."""
    frecs_ghz = []
    columnas_muestreadas = {c: [] for c in columnas}
    for fg in puntos_ghz:
        idx = int(np.argmin(np.abs(frecs - fg * 1e9)))
        frecs_ghz.append(frecs[idx] / 1e9)
        for c in columnas:
            columnas_muestreadas[c].append(columnas[c][idx])
    columnas_muestreadas = {c: np.array(v) for c, v in columnas_muestreadas.items()}
    return frecs_ghz, columnas_muestreadas


def imprimir_tabla_resumen(frecs, columnas, puntos_ghz=(0.5, 0.9, 1.8, 2.45, 3.5, 5.8, 6.0)):
    """Imprime una tabla comparativa en algunas frecuencias de interes.
    `columnas` es un dict {titulo: ndarray complejo} evaluado en `frecs`."""
    encabezado = f"{'f (GHz)':>8} | " + " | ".join(f"{c:>18}" for c in columnas)
    print(encabezado)
    print("-" * len(encabezado))
    for fg in puntos_ghz:
        idx = int(np.argmin(np.abs(frecs - fg * 1e9)))
        fila = f"{frecs[idx]/1e9:8.3f} | " + " | ".join(
            f"{columnas[c][idx]:18.2f}" for c in columnas)
        print(fila)


def exportar_csv(ruta, frecs, columnas):
    """Exporta frecuencia + columnas (dict {nombre: ndarray complejo}) a CSV
    con parte real e imaginaria separadas."""
    with open(ruta, "w", newline='') as f:
        w = csv.writer(f)
        encabezado = ["f_Hz"]
        for nombre in columnas:
            encabezado += [f"{nombre}_re", f"{nombre}_im"]
        w.writerow(encabezado)
        for i in range(len(frecs)):
            fila = [frecs[i]]
            for nombre in columnas:
                fila += [columnas[nombre][i].real, columnas[nombre][i].imag]
            w.writerow(fila)
    print(f"  CSV guardado: {ruta}")


def leer_csv_resultado(ruta):
    """
    Lee un CSV exportado por `exportar_csv` (columna 'f_Hz' + pares de
    columnas '<nombre>_re'/'<nombre>_im', una por cada serie que se haya
    exportado) y lo devuelve invertido: (frecs, columnas), con `columnas`
    un dict {nombre: ndarray complejo}.

    Pensado para la pestaña "Comparar mediciones" de la GUI: permite
    tomar cualquier `tabla_<material>.csv` ya generado (de esta corrida o
    de una anterior, incluso de otro dia -- ver la estructura de carpetas
    con fecha/hora de `ejecutar_analisis`) y superponer alguna de sus
    columnas contra otras mediciones.

    Levanta ValueError si el archivo no tiene el formato esperado (por
    si se elige por error un CSV de otro origen).
    """
    with open(ruta, "r", newline='') as f:
        r = csv.reader(f)
        try:
            encabezado = next(r)
        except StopIteration:
            encabezado = []
        filas = [fila for fila in r if fila]

    if not encabezado or encabezado[0] != 'f_Hz':
        raise ValueError(
            f"'{os.path.basename(ruta)}' no tiene el formato esperado "
            f"(falta la columna 'f_Hz' -- ¿es un CSV generado por este "
            f"programa?).")

    nombres = []
    i = 1
    while i < len(encabezado):
        col = encabezado[i]
        if col.endswith('_re'):
            nombres.append(col[:-3])
            i += 2  # saltea la columna '_im' correspondiente
        else:
            i += 1

    if not filas:
        return np.array([], dtype=float), {n: np.array([], dtype=complex) for n in nombres}

    frecs = np.array([float(fila[0]) for fila in filas], dtype=float)
    columnas = {}
    for nombre in nombres:
        idx_re = encabezado.index(f"{nombre}_re")
        idx_im = encabezado.index(f"{nombre}_im")
        re = np.array([float(fila[idx_re]) for fila in filas], dtype=float)
        im = np.array([float(fila[idx_im]) for fila in filas], dtype=float)
        columnas[nombre] = re + 1j * im
    return frecs, columnas


def graficar_series_multiples(series, titulo="Comparaci\u00f3n de mediciones", log_x=True):
    """
    Grafica varias series er'/er'' superpuestas, cada una con su PROPIA
    grilla de frecuencias -- a diferencia de `graficar_comparacion`, que
    asume una unica `frecs` compartida por todas las curvas. Pensado para
    comparar mediciones que no necesariamente comparten el mismo barrido
    del VNA (p.ej. de distintos dias, o remuestreadas de forma distinta).

    Parametros
    ----------
    series : dict {etiqueta: (frecs, er)}
        `frecs` en Hz, `er` complejo (er' - j*er''), mismo largo que
        `frecs`. Una entrada por curva a superponer.
    titulo : str
    log_x : bool
        Eje de frecuencias en escala logaritmica (default True). Se
        ignora (cae a lineal) si alguna frecuencia de alguna serie es
        <= 0.

    Retorna
    -------
    Figure de matplotlib (construida con la API orientada a objetos, para
    poder embeberla en vivo con `gui_permitividad.VisorFigura`).
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 7))
    ax_re, ax_im = fig.subplots(2, 1, sharex=True)

    frecs_vistas = []
    for etiqueta, (frecs, er) in series.items():
        frecs = np.asarray(frecs, dtype=float)
        er = np.asarray(er, dtype=complex)
        frecs_vistas.append(frecs)
        ax_re.plot(frecs / 1e9, er.real, linewidth=1.6, label=etiqueta)
        ax_im.plot(frecs / 1e9, er.imag, linewidth=1.6, label=etiqueta)

    ax_re.set_ylabel("er'  (parte real)")
    ax_re.set_title(titulo)
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_re.legend()

    ax_im.set_xlabel("Frecuencia (GHz)")
    ax_im.set_ylabel("er''  (perdidas, Im)")
    ax_im.minorticks_on()
    ax_im.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_im.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_im.legend()

    frecs_concat = np.concatenate(frecs_vistas) if frecs_vistas else np.array([])
    if log_x and frecs_concat.size > 0 and np.all(frecs_concat > 0):
        ax_re.set_xscale('log')
        ax_im.set_xscale('log')

    fig.tight_layout()
    return fig


def generar_grilla_frecuencias(f_min_ghz, f_max_ghz, n_puntos, log_x=True):
    """
    Genera una grilla de `n_puntos` frecuencias (en Hz) entre `f_min_ghz`
    y `f_max_ghz` (en GHz). Si `log_x` es True (y el rango no incluye 0),
    los puntos quedan espaciados logaritmicamente -- mas densos a bajas
    frecuencias, la distribucion mas natural si despues se grafican en
    escala log (ver `graficar_series_multiples`). Si es False, espaciados
    linealmente.

    Pensado para graficar un modelo teorico de Patrones.py directamente,
    sin depender de ningun dato medido -- ver la pestaña "6. Modelos
    teoricos" de la GUI (gui_permitividad._construir_tab_modelos).
    """
    f_min_hz = float(f_min_ghz) * 1e9
    f_max_hz = float(f_max_ghz) * 1e9
    n_puntos = max(int(n_puntos), 2)
    if log_x and f_min_hz > 0 and f_max_hz > 0:
        return np.geomspace(f_min_hz, f_max_hz, n_puntos)
    return np.linspace(f_min_hz, f_max_hz, n_puntos)


# Nombres lindos para cada estrategia, usados tanto en etiquetas de
# graficos/tablas como en metadata del PDF.
ETIQUETA_ESTRATEGIA_GN = {
    'minimo_gn': "Autom\u00e1tico (m\u00ednimo de |Gn|)",
    'frecuencia_maxima': "Cl\u00e1sico (frecuencia m\u00e1s alta)",
    'comparar_ambas': "Comparar ambas",
}


def _calcular_metodo_completo(frecs, s11_dut, s11_patron3, s11_aire, s11_patron4,
                               s11_corto, Er_patron3, Er_patron4, estrategia_gn,
                               prefijo="Medido (completo", verbose=False):
    """
    Calcula el metodo completo para un DUT (`s11_dut`), respetando
    `estrategia_gn`:

    - 'minimo_gn' / 'frecuencia_maxima': una sola corrida con esa
      estrategia. Devuelve {f"{prefijo})": resultado} (una sola entrada).
    - 'comparar_ambas': corre LAS DOS estrategias (automatica y clasica)
      y devuelve las dos, con etiquetas distintas -- para poder ver de
      un vistazo si la eleccion de estrategia cambia algo con los datos
      reales, sin tener que correr el analisis dos veces a mano.

    Se usa tanto para el chequeo de calibracion (patron 3 y patron 4,
    tratados como si fueran un DUT mas) como para cada material en
    `procesar_material` -- `prefijo` es lo unico que cambia entre esos
    usos, para que la etiqueta final tenga sentido en cada contexto.

    Retorna un dict {etiqueta: ndarray complejo} con 1 o 2 entradas.
    """
    if estrategia_gn == 'comparar_ambas':
        return {
            f"{prefijo}, Autom\u00e1tico)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla='minimo_gn'),
            f"{prefijo}, Cl\u00e1sico)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla='frecuencia_maxima'),
        }
    return {
        f"{prefijo})": get_er_DUT_completo(
            frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla=estrategia_gn),
    }


def procesar_material(material, carpeta_datos, cache, frecs,
                       s11_corto, s11_aire, s11_patron3, s11_patron4,
                       Er_patron3, Er_patron4, carpeta_salida=None,
                       f_min_ghz=None, f_max_ghz=None, log_x=None,
                       estrategia_gn='minimo_gn'):
    """Procesa una entrada de MATERIALES: lee su .s1p, calcula la
    permitividad con los metodos indicados, grafica (contra la curva
    teorica si se conoce, o solo lo medido si no), exporta CSV y devuelve
    un dict con todo lo necesario para armar el informe PDF (o None si el
    archivo no se encontro, en cuyo caso se salta este material).

    `Er_patron3`/`Er_patron4`: permitividad teorica de los patrones 3 y 4
    de la calibracion, YA EVALUADA en `frecs` (ver `ejecutar_analisis`,
    que resuelve que modelo y temperatura usar para cada uno).
    `s11_patron4`/`Er_patron4` pueden ser None si el metodo completo esta
    deshabilitado (sin patron 4): en ese caso, cualquier material que
    pida 'completo' en sus metodos simplemente se omite (con un aviso),
    y solo se calcula lo que haya pedido con 'simplificado'.
    `carpeta_salida`, `f_min_ghz`, `f_max_ghz`, `log_x`: si se omiten
    (None), se usan las constantes definidas mas arriba en este archivo
    (comportamiento identico al de antes). `ejecutar_analisis` los pasa
    explicitamente para poder usar valores distintos (p.ej. los que cargo
    la GUI) sin tocar esas constantes.
    `estrategia_gn`: se pasa tal cual a `funciones.get_er_DUT_completo`
    (parametro `estrategia_semilla`) -- ver su docstring."""
    carpeta_salida = CARPETA_SALIDA if carpeta_salida is None else carpeta_salida
    f_min_ghz = F_MIN_GHZ if f_min_ghz is None else f_min_ghz
    f_max_ghz = F_MAX_GHZ if f_max_ghz is None else f_max_ghz
    log_x = ESCALA_LOG_FRECUENCIA if log_x is None else log_x

    nombre = material['nombre']
    ruta = os.path.join(carpeta_datos, material['archivo'])
    print(f"\nProcesando: {nombre}  ({material['archivo']})")

    if not os.path.isfile(ruta):
        print(f"  [aviso] no se encontro '{ruta}': se salta este material. "
              f"Subi el archivo o corregi el nombre en MATERIALES.")
        return None

    datos = _cargar_cache(ruta, cache)
    print(f"  {nombre:20s} <- {material['archivo']}  "
          f"({len(datos['Frec'])} puntos, "
          f"{datos['Frec'][0]/1e9:.3f}-{datos['Frec'][-1]/1e9:.3f} GHz)")

    frec_dut = np.asarray(datos['Frec'], dtype=float)
    coincide = (len(frec_dut) == len(frecs)) and np.allclose(frec_dut, frecs, atol=1.0)
    if not coincide:
        print(f"  [info] '{material['archivo']}' no comparte la grilla de "
              f"frecuencias de calibracion: remuestreando.")
        datos = resamplear(frecs, datos)
    s11_dut = datos['Complex']

    metodos = material.get('metodos', ['simplificado', 'completo'])
    resultados = {}
    if 'simplificado' in metodos:
        resultados['Medido (simplificado, 3 patrones)'] = get_er_DUTm(
            frecs, s11_dut, s11_patron3, s11_aire, s11_corto, Er_patron3)
    if 'completo' in metodos:
        if s11_patron4 is None or Er_patron4 is None:
            print(f"  [aviso] '{nombre}' tenia tildado el metodo completo, pero el "
                  f"metodo completo esta deshabilitado (sin patron 4): se omite y "
                  f"solo se calcula lo que haya con el metodo simplificado.")
        else:
            resultados.update(_calcular_metodo_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, estrategia_gn,
                prefijo="Medido (completo, 4 patrones", verbose=True))

    modelo = material.get('modelo') or None
    temperatura_material = float(material.get('temperatura', 25.0)) if modelo else None
    teorico_fn = resolver_modelo_teorico(modelo, temperatura_material)
    er_teo = teorico_fn(frecs) if teorico_fn is not None else None

    # Advertencia del modelo teorico elegido (p.ej. los modelos de
    # permitividad ESTATICA -- acetona, ciclohexano, silicona -- que no
    # tienen ecuacion de relajacion ajustada). Se imprime en consola y
    # viaja en el resultado para que el informe PDF la muestre como caja
    # destacada junto al grafico de ese material.
    advertencia_modelo = advertencia_patron(modelo) if modelo else None
    if advertencia_modelo:
        etiqueta_mod = etiqueta_patron(modelo) or modelo
        print(f"  [ADVERTENCIA - modelo teorico '{etiqueta_mod}']")
        for linea in advertencia_modelo.splitlines():
            if linea.strip():
                print(f"    {linea.strip()}")

    mascara = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
    f_r = frecs[mascara]
    resultados_r = {k: v[mascara] for k, v in resultados.items()}
    er_teo_r = er_teo[mascara] if er_teo is not None else None

    slug = _slug(nombre)
    titulo = f"{nombre}: medido vs. teorico" if er_teo_r is not None else \
             f"{nombre}: medido (sin modelo teorico de referencia)"
    ruta_figura = os.path.join(carpeta_salida, f"er_{slug}.png")
    etiq_teorico = etiqueta_curva_teorica(modelo)

    graficar_comparacion(f_r, resultados_r, er_teo_r, titulo, ruta_figura,
                          log_x=log_x, etiqueta_teorico=etiq_teorico)

    # S11 medido (crudo, previo a la conversion a permitividad): modulo+fase
    # y diagrama de Smith, en el mismo recorte de frecuencias y la misma
    # escala log_x que el grafico de arriba, para poder correlacionar
    # cualquier anomalia de er'/er'' con lo que se ve directamente en S11
    # (p.ej. una resonancia de la sonda o un problema de contacto).
    s11_dut_r = s11_dut[mascara]
    datos_s11_r = {nombre: {'Frec': f_r, 'Complex': s11_dut_r}}

    ruta_figura_s11 = os.path.join(carpeta_salida, f"s11_{slug}.png")
    graficar_s11_mag_fase(datos_s11_r, titulo=f"{nombre}: S11 medido (modulo y fase)",
                          archivo_salida=ruta_figura_s11, log_x=log_x)

    ruta_figura_smith = os.path.join(carpeta_salida, f"smith_{slug}.png")
    graficar_smith(datos_s11_r, titulo=f"{nombre}: S11 medido (diagrama de Smith)",
                   archivo_salida=ruta_figura_smith)

    errores = None
    tabla_frecs_ghz = tabla_columnas = None
    nota = None
    if er_teo_r is not None:
        print(f"  Error relativo:")
        errores = {}
        for etiqueta, er in resultados_r.items():
            errores[etiqueta] = reportar_error(etiqueta, er, er_teo_r)
        print(f"  Tabla resumen ({nombre}):")
        columnas_tabla = {"Teorico": er_teo, **resultados}
        imprimir_tabla_resumen(frecs, columnas_tabla)
        tabla_frecs_ghz, tabla_columnas = muestrear_en_frecuencias(frecs, columnas_tabla)
    else:
        nota = (f"Sin modelo teorico cargado para '{nombre}': se grafica "
                f"unicamente el valor medido (no hay con que comparar el "
                f"error). Si mas adelante conseguis/armas un modelo "
                f"teorico, agregalo en Patrones.py y en la entrada de "
                f"MATERIALES.")
        print(f"  {nota}")
        tabla_frecs_ghz, tabla_columnas = muestrear_en_frecuencias(frecs, resultados)

    columnas_csv = dict(resultados)
    if er_teo is not None:
        columnas_csv = {'teorico': er_teo, **resultados}
    exportar_csv(os.path.join(carpeta_salida, f"tabla_{slug}.csv"), frecs, columnas_csv)

    return {
        'nombre': nombre,
        'archivo': material['archivo'],
        'figura': ruta_figura,
        'figura_s11': ruta_figura_s11,
        'figura_smith': ruta_figura_smith,
        'tiene_teorico': er_teo is not None,
        'modelo': modelo,
        'modelo_etiqueta': etiqueta_patron(modelo),
        'temperatura': temperatura_material,
        'errores': errores,
        'tabla_frecs_ghz': tabla_frecs_ghz,
        'tabla_columnas': tabla_columnas,
        'nota': nota,
        # Advertencia del modelo teorico (ver Patrones.INFO_PATRONES): la
        # muestra el informe PDF como caja destacada y la GUI como cartel.
        # `modelo_estatico`/`modelo_modela_perdidas` permiten que el PDF
        # aclare por que la columna de error de er'' puede decir "n/a".
        'advertencia_modelo': advertencia_modelo,
        'nivel_advertencia': nivel_advertencia_patron(modelo) if modelo else 'info',
        'modelo_estatico': es_modelo_estatico(modelo) if modelo else False,
        'modelo_modela_perdidas': modela_perdidas(modelo) if modelo else True,
        'modelo_resumen': resumen_corto_patron(modelo) if modelo else None,
        # Datos crudos para reconstruir la figura en vivo (interactiva,
        # con zoom/pan) en la GUI sin tener que releer el .png -- ver
        # gui_permitividad.VisorFigura y _figuras_desde_resultado. El PNG
        # de 'figura' arriba se sigue generando igual, para el informe PDF.
        'datos_grafico': {
            'frecs': f_r, 'medido': resultados_r, 'teorico': er_teo_r,
            'titulo': titulo, 'log_x': log_x,
            'etiqueta_teorico': etiq_teorico,
        },
    }


def etiqueta_curva_teorica(modelo):
    """Texto de leyenda para la curva teorica de `modelo`.

    No todos los patrones de `Patrones.py` son un Debye: los de
    permitividad estatica (acetona, ciclohexano, silicona) no tienen
    ecuacion de relajacion ajustada, asi que rotular su curva como
    "Teorico (Debye)" -- como hacia esta funcion antes, fijo para todos --
    seria directamente engañoso en el grafico que despues va al informe.
    """
    if not modelo:
        return "Teorico"
    if es_modelo_estatico(modelo):
        return "Teorico (ε estatica NPL, sin relajacion)"
    return "Teorico (Debye)"


def _advertencias_calibracion(modelo_patron3, modelo_patron4):
    """
    Junta las advertencias de los modelos teoricos usados como patron 3 y
    patron 4 de la calibracion, en una lista de dicts listos para mostrar
    (consola, GUI o caja destacada del informe PDF).

    Cada entrada: {'rol', 'modelo', 'etiqueta', 'nivel', 'texto'}.

    Ademas de repetir la advertencia propia del modelo, agrega un reparo
    EXTRA que solo aplica cuando el modelo se usa como patron: si el
    liquido es no polar (permitividad ~2, muy cerca de la del aire), el
    sistema de ecuaciones de calibracion queda mal condicionado -- el
    patron aporta muy poca informacion nueva respecto del circuito
    abierto, y el ruido de medicion se amplifica sobre TODOS los
    materiales analizados con esa calibracion.
    """
    advertencias = []
    for rol, modelo in (("Patron 3", modelo_patron3), ("Patron 4", modelo_patron4)):
        if not modelo:
            continue
        texto = advertencia_patron(modelo)
        info_extra = ""
        if es_modelo_estatico(modelo):
            info_extra = (
                f"\n\nUSADO COMO {rol.upper()} DE CALIBRACION: ademas de lo "
                f"anterior, tener en cuenta que la permitividad de este "
                f"patron entra DIRECTAMENTE en las formulas de conversion "
                f"S11 -> er, asi que cualquier limitacion del modelo se "
                f"propaga al resultado de TODOS los materiales, no solo al "
                f"de este patron."
            )
        if not texto and not info_extra:
            continue
        advertencias.append({
            'rol': rol,
            'modelo': modelo,
            'etiqueta': etiqueta_patron(modelo) or modelo,
            'nivel': nivel_advertencia_patron(modelo),
            'texto': (texto or "") + info_extra,
        })
    return advertencias


def ejecutar_analisis(config, log=print, progreso=None, cancelado=None):
    """
    Corre el pipeline completo (calibracion + cada material + informe
    PDF) a partir de un dict de configuracion. Tanto `main()` (modo
    script, con las constantes definidas arriba en este archivo) como la
    GUI (gui_permitividad.py / gui_funciones.py) llaman a esta misma
    funcion, para no duplicar logica en dos lugares.

    Parametros
    ----------
    config : dict
        'carpeta_datos'        : str, carpeta base para .s1p con ruta
                                  relativa (puede ser "" si las rutas de
                                  archivo ya son absolutas).
        'archivos_calibracion' : dict con las 4 claves fijas 'corto',
                                  'aire', 'patron3', 'patron4'.
        'patron3_modelo'       : clave de Patrones.PATRONES_TEORICOS para
                                  el 3er patron de calibracion (default:
                                  'agua', el par tradicional de la
                                  catedra -- pero puede ser cualquier
                                  liquido con modelo conocido).
        'patron3_temperatura_c': float, temperatura real de ese liquido
                                  durante la calibracion.
        'patron4_modelo'       : idem para el 4to patron (default:
                                  'alcohol_isopropilico'), solo usado por
                                  el metodo completo. Se ignora por
                                  completo si 'usar_metodo_completo' es
                                  False.
        'patron4_temperatura_c': float, temperatura real de ese liquido.
        'usar_metodo_completo' : bool, opcional (default True). Si es
                                  False, no hace falta el patron 4 para
                                  nada (ni archivo, ni modelo/temperatura):
                                  todos los materiales se procesan
                                  UNICAMENTE con el metodo simplificado,
                                  sin calcular Gn ni el diagnostico
                                  correspondiente. Util si solo interesan
                                  materiales de bajas perdidas y no se
                                  quiere medir/calibrar un 4to patron.
        'estrategia_gn'        : 'minimo_gn' (default) o
                                  'frecuencia_maxima' -- se pasa tal cual
                                  a `funciones.get_er_DUT_completo` (ver
                                  su parametro `estrategia_semilla`). Sin
                                  efecto si 'usar_metodo_completo' es
                                  False.
        'materiales'           : lista de dicts {nombre, archivo, modelo,
                                  temperatura, metodos} (ver formato de
                                  MATERIALES mas arriba).
        'carpeta_salida'       : str, carpeta BASE de salida. Los
                                  archivos de esta corrida en particular
                                  van en un subdirectorio
                                  '<carpeta_salida>/<AAAA-MM-DD>/<HH-MM-SS>/'
                                  (ver mas abajo), no directamente aca.
        'nombre_informe_pdf'   : str, nombre del PDF final.
        'f_min_ghz', 'f_max_ghz' : float, rango de frecuencias de interes.
        'escala_log_frecuencia' : bool, opcional (default True). Eje de
                                frecuencias en escala logaritmica en los
                                graficos generados. Recomendado cuando el
                                barrido cubre varias decadas (p.ej. 1 MHz
                                a 2 GHz); en lineal, todo lo que pasa por
                                debajo de un par de cientos de MHz queda
                                invisible contra el borde izquierdo.
    log : funcion(str) -> None
        Para reportar progreso textual (default: print). La GUI pasa una
        funcion que escribe en su consola en vez de la terminal.
    progreso : funcion(paso_actual, total_pasos, etiqueta=None) -> None,
        opcional. Se llama al empezar la calibracion, al terminarla, antes
        y despues de procesar cada material, y antes/despues de generar
        el PDF, para que la GUI pueda actualizar una barra de progreso Y
        mostrar que paso esta corriendo en ese momento (`etiqueta`, un
        string descriptivo, o None si no hay nada nuevo que mostrar). Si
        es None, no se reporta nada (no rompe el uso desde consola).
    cancelado : funcion() -> bool, opcional
        Si se pasa y en algun momento devuelve True, el analisis se
        interrumpe apenas termina de procesar el material actual (nunca a
        mitad de un calculo) y se genera igual un informe PDF PARCIAL con
        lo que ya se alcanzo a calcular. Pensado para conectarse a un
        `threading.Event().is_set` desde el boton "Cancelar" de la GUI.
        Si es None (default), el analisis nunca se cancela solo --
        comportamiento identico al de antes de que existiera este
        parametro.

    Carpeta de salida de esta corrida
    ----------------------------------
    Todo lo que genera esta corrida (figuras, CSVs, informe PDF) se
    guarda en '<carpeta_salida>/<AAAA-MM-DD>/<HH-MM-SS>/', no directamente
    en 'carpeta_salida': asi, corridas de dias o momentos distintos nunca
    se pisan entre si, y queda un historial ordenado sin tener que
    cambiar a mano el nombre de la carpeta de salida cada vez. Ademas, los
    .s1p de ENTRADA (calibracion + cada material) se copian a la carpeta
    del DIA -- '<carpeta_salida>/<AAAA-MM-DD>/', compartida entre todas
    las corridas de ese dia -- para dejar registro de con que mediciones
    exactas se genero cada informe.

    Retorna
    -------
    dict con 'chequeo_calibracion', 'resultados_materiales',
    'ruta_informe_pdf', 'carpeta_salida_run' (la carpeta real, con fecha y
    hora, donde quedo todo lo de ESTA corrida) y 'cancelado' (bool: True
    si se interrumpio a mitad de camino por el motivo de arriba).
    """
    if progreso is None:
        progreso = lambda paso, total, etiqueta=None: None
    if cancelado is None:
        cancelado = lambda: False

    carpeta_datos = config['carpeta_datos']
    carpeta_salida_base = config['carpeta_salida']
    cache = {}

    modelo_patron3 = config.get('patron3_modelo') or PATRON3_MODELO_CAL
    T_patron3 = float(config.get('patron3_temperatura_c', PATRON3_TEMPERATURA_CAL_C))
    modelo_patron4 = config.get('patron4_modelo') or PATRON4_MODELO_CAL
    T_patron4 = float(config.get('patron4_temperatura_c', PATRON4_TEMPERATURA_CAL_C))
    # Si esta en False, no hace falta el patron 4 en absoluto: ni el
    # archivo, ni el modelo/temperatura, ni Gn. Util cuando solo interesa
    # el metodo simplificado (p.ej. materiales de bajas perdidas) y no se
    # quiere tener que medir/calibrar un 4to patron para nada.
    usar_metodo_completo = bool(config.get('usar_metodo_completo', True))
    # Donde arranca la "marcha" en frecuencia del metodo completo -- ver
    # el parametro homonimo de `funciones.get_er_DUT_completo`. Sin efecto
    # si usar_metodo_completo es False.
    estrategia_gn = config.get('estrategia_gn', 'minimo_gn')
    f_min_ghz = config['f_min_ghz']
    f_max_ghz = config['f_max_ghz']
    log_x = config.get('escala_log_frecuencia', ESCALA_LOG_FRECUENCIA)
    materiales = config['materiales']
    archivos_cal = dict(config['archivos_calibracion'])
    if not usar_metodo_completo:
        archivos_cal.pop('patron4', None)  # no se lee ni se valida si no hace falta
    total_pasos = len(materiales) + 2  # calibracion + N materiales + informe PDF

    if cancelado():
        log("\n[cancelado] Analisis interrumpido antes de empezar.")
        return {'chequeo_calibracion': None, 'resultados_materiales': [],
                'ruta_informe_pdf': None, 'carpeta_salida_run': None,
                'cancelado': True}

    # -----------------------------------------------------------------
    # Carpeta de salida de ESTA corrida: <carpeta_salida>/<fecha>/<hora>/.
    # Ver docstring de arriba para el porque de esta estructura.
    # -----------------------------------------------------------------
    ahora = datetime.now()
    carpeta_dia = os.path.join(carpeta_salida_base, ahora.strftime("%Y-%m-%d"))
    carpeta_salida = os.path.join(carpeta_dia, ahora.strftime("%H-%M-%S"))
    os.makedirs(carpeta_salida, exist_ok=True)

    progreso(0, total_pasos, "Leyendo patrones de calibraci\u00f3n")
    log(f"Carpeta de esta corrida: {os.path.abspath(carpeta_salida)}")
    log("Leyendo patrones de calibracion...")
    calib = cargar_calibracion(carpeta_datos, archivos_cal, cache)
    frecs = calib['patron3']['Frec']

    s11_corto = calib['corto']['Complex']
    s11_aire = calib['aire']['Complex']
    s11_patron3 = calib['patron3']['Complex']
    s11_patron4 = calib['patron4']['Complex'] if usar_metodo_completo else None

    etiqueta_p3 = etiqueta_patron(modelo_patron3) or modelo_patron3
    log(f"  Patron 3 = {etiqueta_p3} a {T_patron3:.1f}\u00b0C")
    Er_patron3 = PATRONES_TEORICOS[modelo_patron3](frecs, T=T_patron3)

    if usar_metodo_completo:
        etiqueta_p4 = etiqueta_patron(modelo_patron4) or modelo_patron4
        log(f"  Patron 4 = {etiqueta_p4} a {T_patron4:.1f}\u00b0C "
            f"(estrategia de Gn: {estrategia_gn})")
        Er_patron4 = PATRONES_TEORICOS[modelo_patron4](frecs, T=T_patron4)
    else:
        etiqueta_p4 = None
        Er_patron4 = None
        log("  Metodo completo deshabilitado: no se usa Patron 4 ni se calcula Gn "
            "(todos los materiales se procesan solo con el metodo simplificado).")

    _copiar_insumos(carpeta_datos, archivos_cal, materiales, carpeta_dia, log)

    # Advertencias de los modelos usados como patron de calibracion (p.ej.
    # un modelo de permitividad ESTATICA como ciclohexano o silicona). Se
    # loguean aca, apenas se resuelve la calibracion y ANTES de procesar
    # ningun material, porque afectan el resultado de todos ellos.
    for adv in _advertencias_calibracion(
            modelo_patron3, modelo_patron4 if usar_metodo_completo else None):
        log(f"\n  [ADVERTENCIA - {adv['rol']}: {adv['etiqueta']}]")
        for linea in adv['texto'].splitlines():
            if linea.strip():
                log(f"    {linea.strip()}")

    # -----------------------------------------------------------------
    # Chequeo de calibracion: el patron 3 ES uno de los patrones, asi que
    # si se lo "mide" como si fuera un DUT mas, el resultado tiene que
    # coincidir casi exactamente con su propio modelo teorico. Si esto no
    # da ~0% de error, hay un problema de lectura/calibracion antes de
    # analizar cualquier material real.
    # -----------------------------------------------------------------
    log(f"\nChequeo de calibracion ({etiqueta_p3} medido vs. teorico, debe dar ~0% de error)...")
    er_p3_teo = Er_patron3
    er_p3_simpl = get_er_DUTm(frecs, s11_patron3, s11_patron3, s11_aire, s11_corto,
                               Er_patron3)

    mascara = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
    medido_p3 = {"Medido (simplificado)": er_p3_simpl[mascara]}
    errores_p3 = {
        "Metodo simplificado": reportar_error(
            "Metodo simplificado", er_p3_simpl[mascara], er_p3_teo[mascara]),
    }
    if usar_metodo_completo:
        resultados_p3_completo = _calcular_metodo_completo(
            frecs, s11_patron3, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, estrategia_gn, prefijo="Medido (completo")
        for etiqueta, arr in resultados_p3_completo.items():
            medido_p3[etiqueta] = arr[mascara]
            errores_p3[etiqueta] = reportar_error(etiqueta, arr[mascara], er_p3_teo[mascara])

    ruta_figura_p3 = os.path.join(carpeta_salida, "chequeo_calibracion_patron3.png")
    etiq_teorico_p3 = etiqueta_curva_teorica(modelo_patron3)
    graficar_comparacion(
        frecs[mascara],
        medido_p3,
        er_p3_teo[mascara],
        f"{etiqueta_p3}: chequeo de calibracion (medido vs. teorico)",
        ruta_figura_p3,
        log_x=log_x,
        etiqueta_teorico=etiq_teorico_p3,
    )
    chequeo_calibracion = {
        'figura': ruta_figura_p3,
        'etiqueta_patron3': etiqueta_p3,
        'etiqueta_patron4': etiqueta_p4,
        'errores': errores_p3,
        # Datos crudos para reconstruir la figura en vivo en la GUI -- ver
        # `procesar_material` (misma idea) y gui_permitividad.VisorFigura.
        'datos_grafico': {
            'frecs': frecs[mascara],
            'medido': medido_p3,
            'teorico': er_p3_teo[mascara],
            'titulo': f"{etiqueta_p3}: chequeo de calibracion (medido vs. teorico)",
            'log_x': log_x,
            'etiqueta_teorico': etiq_teorico_p3,
        },
    }

    # -----------------------------------------------------------------
    # Chequeo de calibracion ANALOGO para el patron 4 (si esta habilitado):
    # mismo principio que el de arriba para el patron 3 -- se lo "mide"
    # como si fuera un DUT mas y se compara contra su propio modelo
    # teorico. Da una segunda validacion independiente de la calibracion,
    # gratis: si el patron 3 da ~0% de error pero el patron 4 no, el
    # problema esta especificamente en la medicion/modelo de ese 4to
    # patron (y viceversa), algo que el chequeo del patron 3 solo no
    # puede detectar por si mismo.
    # -----------------------------------------------------------------
    if usar_metodo_completo:
        log(f"\nChequeo de calibracion ({etiqueta_p4} medido vs. teorico, debe dar ~0% de error)...")
        er_p4_teo = Er_patron4
        er_p4_simpl = get_er_DUTm(frecs, s11_patron4, s11_patron3, s11_aire, s11_corto,
                                   Er_patron3)
        medido_p4 = {"Medido (simplificado)": er_p4_simpl[mascara]}
        errores_p4 = {
            "Metodo simplificado": reportar_error(
                "Metodo simplificado", er_p4_simpl[mascara], er_p4_teo[mascara]),
        }
        resultados_p4_completo = _calcular_metodo_completo(
            frecs, s11_patron4, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, estrategia_gn, prefijo="Medido (completo")
        for etiqueta, arr in resultados_p4_completo.items():
            medido_p4[etiqueta] = arr[mascara]
            errores_p4[etiqueta] = reportar_error(etiqueta, arr[mascara], er_p4_teo[mascara])

        ruta_figura_p4 = os.path.join(carpeta_salida, "chequeo_calibracion_patron4.png")
        etiq_teorico_p4 = etiqueta_curva_teorica(modelo_patron4)
        graficar_comparacion(
            frecs[mascara],
            medido_p4,
            er_p4_teo[mascara],
            f"{etiqueta_p4}: chequeo de calibracion (medido vs. teorico)",
            ruta_figura_p4,
            log_x=log_x,
            etiqueta_teorico=etiq_teorico_p4,
        )
        chequeo_calibracion['patron4'] = {
            'figura': ruta_figura_p4,
            'errores': errores_p4,
            'datos_grafico': {
                'frecs': frecs[mascara],
                'medido': medido_p4,
                'teorico': er_p4_teo[mascara],
                'titulo': f"{etiqueta_p4}: chequeo de calibracion (medido vs. teorico)",
                'log_x': log_x,
                'etiqueta_teorico': etiq_teorico_p4,
            },
        }

    # -----------------------------------------------------------------
    # Diagnostico adicional: conductancia normalizada Gn(f), calculada
    # UNICAMENTE a partir de los 4 patrones de calibracion (no depende de
    # ningun material analizado). Tiene que verse suave; un salto brusco
    # o un pico aislado suele delatar un problema con la medicion de
    # alguno de los patrones (sobre todo el patron 4, el unico que
    # interviene en este calculo). Ver funciones.calcular_Gn.
    # -----------------------------------------------------------------
    if usar_metodo_completo:
        log("Calculando conductancia normalizada Gn(f) (diagnostico de calibracion)...")
        Gn_cal = calcular_Gn(frecs, s11_patron3, s11_aire, s11_patron4, s11_corto,
                              Er_patron3, Er_patron4)

        # Donde arrancaria la "marcha en frecuencia" con cada estrategia
        # -- se calcula aca (no adentro de get_er_DUT_completo) porque
        # solo depende de Gn y de la estrategia, no del DUT en si, asi
        # que es la misma para todos los materiales de esta corrida. Se
        # marca en el grafico de Gn(f) para ver de un vistazo si el
        # punto de arranque cae en una zona con |Gn| chico (deseable) o
        # no, sin tener que ir a leer el log de texto.
        marcas_inicio_gn = {}
        if estrategia_gn in ('minimo_gn', 'comparar_ambas'):
            idx_auto = int(np.argmin(np.abs(Gn_cal)))
            marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['minimo_gn']] = frecs[idx_auto]
        if estrategia_gn in ('frecuencia_maxima', 'comparar_ambas'):
            idx_clasico = int(np.argmax(frecs))
            marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['frecuencia_maxima']] = frecs[idx_clasico]

        ruta_figura_Gn = os.path.join(carpeta_salida, "chequeo_calibracion_Gn.png")
        graficar_Gn(frecs[mascara], Gn_cal[mascara], ruta_figura_Gn, log_x=log_x,
                    marcas_inicio=marcas_inicio_gn)
        chequeo_calibracion['figura_Gn'] = ruta_figura_Gn
        chequeo_calibracion['datos_grafico_Gn'] = {
            'frecs': frecs[mascara], 'Gn': Gn_cal[mascara], 'log_x': log_x,
            'marcas_inicio': marcas_inicio_gn,
        }

    progreso(1, total_pasos, "Calibraci\u00f3n completa")

    # -----------------------------------------------------------------
    # Cada material de la lista se procesa igual, tenga o no modelo
    # teorico de referencia. Se guarda el resultado de cada uno (si el
    # archivo existia) para armar el informe PDF al final.
    # -----------------------------------------------------------------
    resultados_materiales = []
    interrumpido = False
    for i, material in enumerate(materiales):
        if cancelado():
            log(f"\n[cancelado] Analisis interrumpido por el usuario antes de "
                f"procesar '{material['nombre']}'. Se genera un informe PDF "
                f"parcial con los {len(resultados_materiales)} material(es) ya "
                f"calculado(s).")
            interrumpido = True
            break
        progreso(1 + i, total_pasos, f"Procesando: {material['nombre']}")
        resultado = procesar_material(
            material, carpeta_datos, cache, frecs,
            s11_corto, s11_aire, s11_patron3, s11_patron4,
            Er_patron3, Er_patron4, carpeta_salida=carpeta_salida,
            f_min_ghz=f_min_ghz, f_max_ghz=f_max_ghz, log_x=log_x,
            estrategia_gn=estrategia_gn,
        )
        if resultado is not None:
            resultados_materiales.append(resultado)
        progreso(2 + i, total_pasos, f"{material['nombre']} listo")

    # -----------------------------------------------------------------
    # Informe PDF final: portada + chequeo de calibracion + un bloque por
    # material, todo junto en un solo archivo para guardar/adjuntar. Se
    # genera SIEMPRE, incluso si se cancelo a mitad de camino (con lo que
    # ya se alcanzo a calcular) -- asi no se pierde el trabajo ya hecho.
    # -----------------------------------------------------------------
    metadata = {
        'patron3_modelo': modelo_patron3,
        'patron3_etiqueta': etiqueta_p3,
        'patron3_temperatura': T_patron3,
        'usar_metodo_completo': usar_metodo_completo,
        'patron4_modelo': modelo_patron4 if usar_metodo_completo else None,
        'patron4_etiqueta': etiqueta_p4 if usar_metodo_completo else None,
        'patron4_temperatura': T_patron4 if usar_metodo_completo else None,
        'estrategia_gn': estrategia_gn if usar_metodo_completo else None,
        'f_min_ghz': f_min_ghz,
        'f_max_ghz': f_max_ghz,
        'archivos_calibracion': archivos_cal,
        'cancelado': interrumpido,
        # Advertencias de los modelos usados como PATRON DE CALIBRACION.
        # Aca importan aun mas que en un material cualquiera: la
        # permitividad del patron entra directo en las formulas de
        # conversion de funciones.py, asi que un modelo que no modela
        # perdidas (o cuya permitividad esta demasiado cerca de la del
        # aire, como el ciclohexano o la silicona) degrada el resultado de
        # TODOS los materiales, no solo el suyo.
        'advertencias_calibracion': _advertencias_calibracion(
            modelo_patron3, modelo_patron4 if usar_metodo_completo else None),
    }
    progreso(1 + len(resultados_materiales), total_pasos, "Generando informe PDF")
    ruta_pdf = os.path.join(carpeta_salida, config['nombre_informe_pdf'])
    generar_reporte_pdf(ruta_pdf, metadata, chequeo_calibracion, resultados_materiales)
    progreso(total_pasos, total_pasos,
             "Cancelado (informe parcial listo)" if interrumpido else "Listo")

    if interrumpido:
        log(f"\nAnalisis cancelado por el usuario. Informe PDF PARCIAL en: "
            f"{os.path.abspath(carpeta_salida)}")
    else:
        log(f"\nListo. Figuras, tablas e informe PDF en: {os.path.abspath(carpeta_salida)}")

    return {
        'chequeo_calibracion': chequeo_calibracion,
        'resultados_materiales': resultados_materiales,
        'ruta_informe_pdf': ruta_pdf,
        'carpeta_salida_run': carpeta_salida,
        'cancelado': interrumpido,
    }


def config_desde_constantes():
    """Arma el dict de configuracion que espera `ejecutar_analisis` a
    partir de las constantes definidas mas arriba en este archivo (el
    modo de uso "de siempre": editar el .py a mano y correr por
    consola)."""
    return {
        'carpeta_datos': CARPETA_DATOS,
        'archivos_calibracion': dict(ARCHIVOS_CALIBRACION),
        'patron3_modelo': PATRON3_MODELO_CAL,
        'patron3_temperatura_c': PATRON3_TEMPERATURA_CAL_C,
        'patron4_modelo': PATRON4_MODELO_CAL,
        'patron4_temperatura_c': PATRON4_TEMPERATURA_CAL_C,
        'materiales': MATERIALES,
        'carpeta_salida': CARPETA_SALIDA,
        'nombre_informe_pdf': NOMBRE_INFORME_PDF,
        'f_min_ghz': F_MIN_GHZ,
        'f_max_ghz': F_MAX_GHZ,
        'escala_log_frecuencia': ESCALA_LOG_FRECUENCIA,
    }


def main():
    ejecutar_analisis(config_desde_constantes())
    plt.show()


if __name__ == "__main__":
    main()