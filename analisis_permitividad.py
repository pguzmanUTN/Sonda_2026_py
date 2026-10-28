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
from funciones import (get_er_DUTm, get_er_DUT_completo, error_absoluto,
                       error_relativo_porcentual, calcular_Gn, acondicionar_gn,
                       calibrar_minimos_cuadrados, reajustar_abc_con_gn_fijo,
                       get_er_DUT_minimos_cuadrados,
                       validacion_cruzada_minimos_cuadrados, filtrar_s11_savgol,
                       filtrar_s11_pasabajos)
from Patrones import (PATRONES_TEORICOS, ER_AIRE, etiqueta_patron,
                       advertencia_patron, nivel_advertencia_patron,
                       es_modelo_estatico, modela_perdidas, perdidas_comparables,
                       resumen_corto_patron, etiqueta_curva_teorica)
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

# Calibracion redundante por cuadrados minimos (opcional): usa TODOS los
# patrones disponibles a la vez (corto + aire + patron 3 + patron 4 + los
# patrones adicionales de abajo) para ajustar los 4 parametros del modelo
# de la sonda en el sentido de cuadrados minimos sobre S11 medido, en vez
# de resolver un sistema exacto con solo 3 o 4 patrones. Con >= 5 patrones
# el sistema queda sobredeterminado: el error de medicion se promedia entre
# todos y ademas se puede medir el residuo de cada patron para detectar si
# alguno quedo mal medido.
USAR_MINIMOS_CUADRADOS = False
PATRONES_ADICIONALES = [
    # {'archivo': "sonda4-alc-etilico.s1p", 'modelo': 'alcohol_etilico', 'temperatura_c': 20.0},
    # {'archivo': "sonda4-acetona.s1p",     'modelo': 'acetona',         'temperatura_c': 25.0},
    # {'archivo': "sonda4-ciclohexano.s1p", 'modelo': 'ciclohexano',     'temperatura_c': 25.0},
]

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


def _copiar_insumos(carpeta_datos, archivos_calibracion, materiales, carpeta_destino, log,
                    patrones_adicionales=()):
    """
    Copia los .s1p de ENTRADA (los patrones de calibracion + patrones
    adicionales + el archivo de cada material) a `carpeta_destino` -- la
    carpeta del DIA de esta corrida, ver `ejecutar_analisis` -- para dejar un
    registro de con que mediciones exactas se genero cada informe, sin
    depender de que la carpeta de datos original no se toque despues.

    Se copian a la carpeta del DIA (compartida entre todas las corridas
    de ese dia) y no a la de la hora (especifica de esta corrida), para
    no duplicar el mismo archivo de entrada cada vez que se vuelve a
    correr el analisis con la misma medicion.

    No hace fallar el analisis si un archivo puntual no se puede copiar
    (p.ej. problema de permisos): lo avisa por `log` y sigue con el resto.
    """
    os.makedirs(carpeta_destino, exist_ok=True)
    nombres = (list(archivos_calibracion.values())
               + [p['archivo'] for p in patrones_adicionales if p.get('archivo')]
               + [m['archivo'] for m in materiales if m.get('archivo')])
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
                          log_x=True, etiqueta_teorico="Teorico (Debye)", limitar_eje_y=True):
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
    limitar_eje_y : bool, opcional (default True)
        Si es True, y algun trace de medicion excede el rango
        [curva_teorica_min - 1, curva_teorica_max + 1], limita el eje Y
        a dicho intervalo para evitar distorsiones. Si es False, se deja
        el autoescalado por defecto de matplotlib.
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

    # Limitar rango del eje Y si algun trace de las calibraciones excede
    # el intervalo [curva_teorica_min - 1, curva_teorica_max + 1].
    # En caso contrario (o si limitar_eje_y es False), se preserva el autoescalado natural de matplotlib.
    if limitar_eje_y and er_teorico is not None:
        # 1. Parte real (ax_re)
        teo_re = np.real(er_teorico)
        valid_teo_re = teo_re[np.isfinite(teo_re)]
        if len(valid_teo_re) > 0:
            lim_inf_re = float(np.min(valid_teo_re)) - 1.0
            lim_sup_re = float(np.max(valid_teo_re)) + 1.0
            excede_re = False
            for er in er_medido_dict.values():
                v = np.real(er)
                v_valid = v[np.isfinite(v)]
                if len(v_valid) > 0 and (np.min(v_valid) < lim_inf_re or np.max(v_valid) > lim_sup_re):
                    excede_re = True
                    break
            if excede_re:
                ax_re.set_ylim(lim_inf_re, lim_sup_re)

        # 2. Parte imaginaria (ax_im)
        teo_im = np.imag(er_teorico)
        valid_teo_im = teo_im[np.isfinite(teo_im)]
        if len(valid_teo_im) > 0:
            lim_inf_im = float(np.min(valid_teo_im)) - 1.0
            lim_sup_im = float(np.max(valid_teo_im)) + 1.0
            excede_im = False
            for er in er_medido_dict.values():
                v = np.imag(er)
                v_valid = v[np.isfinite(v)]
                if len(v_valid) > 0 and (np.min(v_valid) < lim_inf_im or np.max(v_valid) > lim_sup_im):
                    excede_im = True
                    break
            if excede_im:
                ax_im.set_ylim(lim_inf_im, lim_sup_im)

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


def graficar_Gn(frecs, Gn, archivo_salida=None, log_x=True, marcas_inicio=None,
                Gn_acondicionado=None, info_acond=None):
    """
    Grafica la conductancia normalizada Gn(f) calculada a partir de los 4
    patrones de calibracion (ver `funciones.calcular_Gn`), en modulo y
    fase (mismo estilo que `Touchstone.graficar_s11_mag_fase`). Es un
    diagnostico de la CALIBRACION, no de un material en particular: Gn no
    depende del DUT, asi que esta misma curva aplica a todos los
    materiales analizados con el metodo completo en esta corrida.
    Si se pasa `Gn_acondicionado`, grafica ambas curvas superpuestas para
    comparar el original con el acondicionado.

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
    ax_re, ax_im = fig.subplots(2, 1, sharex=True)

    re_Gn = np.real(Gn)
    im_Gn = np.imag(Gn)

    tiene_marcas = bool(marcas_inicio)
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
                   label="Re(Gn)" if tiene_marcas else None, zorder=3)
        ax_im.plot(frecs / 1e9, im_Gn, linewidth=1.6, color='tab:purple',
                   label="Im(Gn)" if tiene_marcas else None, zorder=3)

    ax_re.set_ylabel("Re(Gn)")
    ax_re.set_title("Conductancia normalizada Gn(f) \u2014 diagnostico de calibracion")
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    ax_im.set_xlabel("Frecuencia (GHz)")
    ax_im.set_ylabel("Im(Gn)")
    ax_im.minorticks_on()
    ax_im.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_im.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    if marcas_inicio:
        colores_marca = ['#d62728', '#2ca02c', '#ff7f0e']
        for i, (etiqueta, f_hz) in enumerate(marcas_inicio.items()):
            color = colores_marca[i % len(colores_marca)]
            f_ghz = f_hz / 1e9
            ax_re.axvline(f_ghz, color=color, linestyle='--', linewidth=1.4,
                          label=f"Arranque: {etiqueta}", zorder=2)
            ax_im.axvline(f_ghz, color=color, linestyle='--', linewidth=1.4,
                          label=f"Arranque: {etiqueta}", zorder=2)

    if info_acond and info_acond.get('disparo_detectado'):
        f_disp = info_acond.get('f2_aplicada') or info_acond.get('f_corte_aplicada')
        if f_disp is not None and f_disp > 0:
            f_disp_ghz = f_disp / 1e9
            lbl_trans = f"Empalme/Corte: {f_disp_ghz:.3f} GHz"
            ax_re.axvline(f_disp_ghz, color='#8856a7', linestyle=':', linewidth=1.4,
                          label=lbl_trans, zorder=2)
            ax_im.axvline(f_disp_ghz, color='#8856a7', linestyle=':', linewidth=1.4,
                          label=lbl_trans, zorder=2)

    if tiene_marcas or tiene_acond:
        ax_re.legend(fontsize=8)
        ax_im.legend(fontsize=8)

    if log_x and frecs.size > 0 and np.all(frecs > 0):
        ax_re.set_xscale('log')
        ax_im.set_xscale('log')

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        print(f"  Figura guardada: {archivo_salida}")
        return archivo_salida
    return fig


def construir_intervalos_error(f_min_ghz, f_max_ghz, f_int1_ghz=None, f_int2_ghz=None):
    """
    Construye 3 intervalos contiguos de frecuencia [f_a, f_b] en Hz
    a partir de las 4 frecuencias:
      f_min_ghz (inicial)
      f_int1_ghz (intermedia 1 ingresada por el usuario)
      f_int2_ghz (intermedia 2 ingresada por el usuario)
      f_max_ghz (final)

    Retorna
    -------
    lista de 3 tuplas: [(f0, f1), (f1, f2), (f2, f3)] en Hz.
    """
    f_min = min(float(f_min_ghz), float(f_max_ghz)) * 1e9
    f_max = max(float(f_min_ghz), float(f_max_ghz)) * 1e9
    span = f_max - f_min

    try:
        f1_val = float(f_int1_ghz) * 1e9 if f_int1_ghz is not None else None
    except (ValueError, TypeError):
        f1_val = None

    try:
        f2_val = float(f_int2_ghz) * 1e9 if f_int2_ghz is not None else None
    except (ValueError, TypeError):
        f2_val = None

    if f1_val is None or f2_val is None:
        puntos = [f_min + span / 3.0, f_min + 2.0 * span / 3.0]
    else:
        puntos = sorted([f1_val, f2_val])
        if puntos[0] <= f_min or puntos[1] >= f_max or puntos[0] >= puntos[1]:
            puntos = [f_min + span / 3.0, f_min + 2.0 * span / 3.0]

    return [
        (f_min, puntos[0]),
        (puntos[0], puntos[1]),
        (puntos[1], f_max)
    ]


def calcular_error_intervalos(frecs, er_medido, er_teorico, intervalos, comparar_imag=True):
    """Calcula el error absoluto medio y maximo (|er_medido - er_teorico|)
    en cada uno de los intervalos de frecuencia especificados."""
    frecs = np.asarray(frecs, dtype=float)
    er_m = np.asarray(er_medido, dtype=complex)
    er_t = np.asarray(er_teorico, dtype=complex)

    res = []
    for f_a, f_b in intervalos:
        mascara = (frecs >= f_a) & (frecs <= f_b)

        f_a_ghz = f_a / 1e9
        f_b_ghz = f_b / 1e9
        etiq = f"{f_a_ghz:.2f} - {f_b_ghz:.2f} GHz"

        if not np.any(mascara):
            res.append({
                'intervalo_str': etiq,
                'f_a': f_a, 'f_b': f_b,
                'err_re_medio': float('nan'), 'err_re_max': float('nan'),
                'err_im_medio': float('nan'), 'err_im_max': float('nan'),
            })
            continue

        err_re, err_im = error_absoluto(er_m[mascara], er_t[mascara])
        if not comparar_imag:
            err_im = np.full(np.shape(err_im), np.nan)

        def _resumen(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return float('nan'), float('nan')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return float(np.nanmean(arr)), float(np.nanmax(arr))

        re_medio, re_max = _resumen(err_re)
        im_medio, im_max = _resumen(err_im)
        res.append({
            'intervalo_str': etiq,
            'f_a': f_a, 'f_b': f_b,
            'err_re_medio': re_medio,
            'err_re_max': re_max,
            'err_im_medio': im_medio,
            'err_im_max': im_max,
        })
    return res


def calcular_error(er_medido, er_teorico, comparar_imag=True):
    """Devuelve un dict con el error absoluto promedio y maximo de
    parte real e imaginaria."""
    err_re, err_im = error_absoluto(er_medido, er_teorico)
    if not comparar_imag:
        err_im = np.full(np.shape(err_im), np.nan)

    def _resumen(arr):
        arr = np.asarray(arr, dtype=float)
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


def _fmt_abs(valor):
    """Formatea un error absoluto, o 'n/a' si es NaN."""
    return "    n/a" if valor is None or np.isnan(valor) else f"{valor:8.4f}"


def reportar_error_intervalos(nombre, frecs, er_medido, er_teorico, intervalos, comparar_imag=True):
    """Imprime el error absoluto por intervalos en consola y devuelve la lista de dicts."""
    res = calcular_error_intervalos(frecs, er_medido, er_teorico, intervalos, comparar_imag=comparar_imag)
    print(f"  {nombre} (error absoluto por intervalos):")
    for r in res:
        print(f"    [{r['intervalo_str']}]:")
        print(f"      error medio  er' = {_fmt_abs(r['err_re_medio'])}   "
              f"error medio  er'' = {_fmt_abs(r['err_im_medio'])}")
        print(f"      error maximo er' = {_fmt_abs(r['err_re_max'])}   "
              f"error maximo er'' = {_fmt_abs(r['err_im_max'])}")
    if any(np.isnan(r['err_im_medio']) for r in res):
        ref_sin_perdidas = bool(np.all(np.imag(np.asarray(er_teorico)) == 0))
        motivo = ("el modelo de referencia no modela perdidas (er'' = 0)"
                  if ref_sin_perdidas else
                  "las perdidas de referencia son menores que la resolucion de la sonda")
        print(f"      (er'' = n/a: {motivo}, ver la advertencia de ese modelo)")
    return res


def reportar_error(nombre, er_medido, er_teorico, comparar_imag=True):
    """Imprime el error absoluto en consola y lo devuelve (ver `calcular_error`)."""
    e = calcular_error(er_medido, er_teorico, comparar_imag=comparar_imag)
    print(f"  {nombre}:")
    print(f"    error medio  er' = {_fmt_abs(e['err_re_medio'])}   "
          f"error medio  er'' = {_fmt_abs(e['err_im_medio'])}")
    print(f"    error maximo er' = {_fmt_abs(e['err_re_max'])}   "
          f"error maximo er'' = {_fmt_abs(e['err_im_max'])}")
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


def graficar_residuos_minimos_cuadrados(frecs, residuos, archivo_salida=None, log_x=True):
    """
    Grafica |S11 medido - S11 predicho por el ajuste| en funcion de la
    frecuencia para cada patron incluido en la calibracion por cuadrados
    minimos. Con >= 5 patrones este residuo mide directamente la
    consistencia entre patrones: una curva mucho mas alta que las demas
    apunta al patron con peor medicion o modelo teorico menos fiel.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 5.2))
    ax = fig.subplots(1, 1)

    for nombre, res in residuos.items():
        mag = np.maximum(np.abs(res), 1e-16)
        ax.plot(frecs / 1e9, mag, linewidth=1.6, label=nombre)

    ax.set_xlabel("Frecuencia (GHz)")
    ax.set_ylabel("|S11 medido \u2212 S11 modelo|")
    ax.set_title("Calibraci\u00f3n por cuadrados m\u00ednimos \u2014 residuo de S11 por patr\u00f3n")
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)

    if log_x and frecs.size > 0 and np.all(frecs > 0):
        ax.set_xscale('log')

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        print(f"  Figura guardada: {archivo_salida}")
        return archivo_salida
    return fig


# Nombres lindos para cada estrategia, usados tanto en etiquetas de
# graficos/tablas como en metadata del PDF.
ETIQUETA_ESTRATEGIA_GN = {
    'minimo_gn': "Autom\u00e1tico (m\u00ednimo de |Gn|)",
    'frecuencia_maxima': "Cl\u00e1sico (frecuencia m\u00e1s alta)",
    'comparar_ambas': "Comparar ambas",
}


def _calcular_metodo_completo(frecs, s11_dut, s11_patron3, s11_aire, s11_patron4,
                               s11_corto, Er_patron3, Er_patron4, estrategia_gn,
                               prefijo="Medido (completo", verbose=False,
                               gn_original=None, gn_acondicionado=None,
                               comparar_gn=False):
    """
    Calcula el metodo completo para un DUT (`s11_dut`), respetando
    `estrategia_gn` y el posible acondicionamiento de Gn:

    - Si `gn_acondicionado` esta activo y `comparar_gn=True`: corre ambos
      (Gn original y Gn acondicionado) para poder comparar el impacto
      directamente en el mismo grafico y tablas.
    - Si `gn_acondicionado` esta activo y `comparar_gn=False`: corre unicamente
      con el Gn acondicionado.
    - Si no hay acondicionamiento: comportamiento estandar (o comparacion
      de estrategias automatico vs clasico si `estrategia_gn == 'comparar_ambas'`).

    Retorna un dict {etiqueta: ndarray complejo} con 1 o 2 entradas.
    """
    semilla = 'minimo_gn' if estrategia_gn == 'comparar_ambas' else estrategia_gn

    if gn_acondicionado is not None and comparar_gn:
        return {
            f"{prefijo}, Gn original)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla=semilla,
                Gn=gn_original),
            f"{prefijo}, Gn acondicionado)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla=semilla,
                Gn=gn_acondicionado),
        }
    elif gn_acondicionado is not None:
        return {
            f"{prefijo}, Gn acondicionado)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla=semilla,
                Gn=gn_acondicionado),
        }

    if estrategia_gn == 'comparar_ambas':
        return {
            f"{prefijo}, Autom\u00e1tico)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla='minimo_gn',
                Gn=gn_original),
            f"{prefijo}, Cl\u00e1sico)": get_er_DUT_completo(
                frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
                Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla='frecuencia_maxima',
                Gn=gn_original),
        }
    return {
        f"{prefijo})": get_er_DUT_completo(
            frecs, s11_dut, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, verbose=verbose, estrategia_semilla=estrategia_gn,
            Gn=gn_original),
    }


def _calcular_metodo_mc(frecs, s11_dut, estrategia_gn, calibracion_mc,
                        calibracion_mc_acond=None, comparar_gn=False,
                        prefijo=None, verbose=False):
    """
    Calcula el metodo de cuadrados minimos para un DUT (`s11_dut`),
    respetando `estrategia_gn` y el posible acondicionamiento de Gn (con
    reajuste de (a, b, c) en `calibracion_mc_acond`).

    Retorna un dict {etiqueta: ndarray complejo} con 1 o 2 entradas.
    """
    n_pat = len(calibracion_mc.get('nombres', []))
    if prefijo is None:
        prefijo = f"Medido (cuadrados m\u00ednimos, {n_pat} patrones"
    semilla = 'minimo_gn' if estrategia_gn == 'comparar_ambas' else estrategia_gn

    if calibracion_mc_acond is not None and comparar_gn:
        return {
            f"{prefijo}, Gn original)": get_er_DUT_minimos_cuadrados(
                frecs, s11_dut, calibracion_mc, verbose=verbose,
                estrategia_semilla=semilla),
            f"{prefijo}, Gn acondicionado)": get_er_DUT_minimos_cuadrados(
                frecs, s11_dut, calibracion_mc_acond, verbose=verbose,
                estrategia_semilla=semilla),
        }
    elif calibracion_mc_acond is not None:
        return {
            f"{prefijo}, Gn acondicionado)": get_er_DUT_minimos_cuadrados(
                frecs, s11_dut, calibracion_mc_acond, verbose=verbose,
                estrategia_semilla=semilla),
        }

    if estrategia_gn == 'comparar_ambas':
        return {
            f"{prefijo}, Autom\u00e1tico)": get_er_DUT_minimos_cuadrados(
                frecs, s11_dut, calibracion_mc, verbose=verbose,
                estrategia_semilla='minimo_gn'),
            f"{prefijo}, Cl\u00e1sico)": get_er_DUT_minimos_cuadrados(
                frecs, s11_dut, calibracion_mc, verbose=verbose,
                estrategia_semilla='frecuencia_maxima'),
        }
    return {
        f"{prefijo})": get_er_DUT_minimos_cuadrados(
            frecs, s11_dut, calibracion_mc, verbose=verbose,
            estrategia_semilla=estrategia_gn),
    }


def procesar_material(material, carpeta_datos, cache, frecs,
                       s11_corto, s11_aire, s11_patron3, s11_patron4,
                       Er_patron3, Er_patron4, carpeta_salida=None,
                       f_min_ghz=None, f_max_ghz=None, log_x=None,
                       estrategia_gn='minimo_gn',
                       gn_original=None, gn_acondicionado=None,
                       comparar_gn=False,
                       intervalos_error=None,
                       calibracion_mc=None,
                       calibracion_mc_acond=None,
                       limitar_eje_y=True):
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
    (parametro `estrategia_semilla`) -- ver su docstring.
    `gn_original`, `gn_acondicionado`, `comparar_gn`: si estan presentes,
    permiten calcular y comparar el metodo completo con ambos Gn.
    `intervalos_error`: lista de tuplas [(f_a, f_b), ...] en Hz para
    evaluar error absoluto.
    `calibracion_mc`, `calibracion_mc_acond`: si estan presentes y el
    material incluye 'minimos_cuadrados' en sus metodos, calcula la
    permitividad por cuadrados minimos (original y/o Gn acondicionado)."""
    carpeta_salida = CARPETA_SALIDA if carpeta_salida is None else carpeta_salida
    f_min_ghz = F_MIN_GHZ if f_min_ghz is None else f_min_ghz
    f_max_ghz = F_MAX_GHZ if f_max_ghz is None else f_max_ghz
    log_x = ESCALA_LOG_FRECUENCIA if log_x is None else log_x
    if intervalos_error is None:
        intervalos_error = construir_intervalos_error(f_min_ghz, f_max_ghz)

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
    s11_dut_orig = s11_dut.copy()

    filtrar_s11 = bool(material.get('filtrar_s11') or material.get('filtrar_sg', False))
    tipo_filtro = material.get('tipo_filtro', 'savitzky_golay')
    if filtrar_s11:
        if tipo_filtro == 'pasabajos':
            pb_ord = int(material.get('pb_orden', 2) or 2)
            pb_fc = float(material.get('pb_fcorte', 0.10) or 0.10)
            s11_dut = filtrar_s11_pasabajos(s11_dut, orden=pb_ord, f_corte=pb_fc)
            print(f"  [info] Filtro Pasabajos Butterworth aplicado a S11 de '{nombre}' (orden={pb_ord}, corte={pb_fc:.3f}).")
        else:
            sg_w = int(material.get('sg_window', 15) or 15)
            sg_p = int(material.get('sg_polyorder', 3) or 3)
            s11_dut = filtrar_s11_savgol(s11_dut, window_length=sg_w, polyorder=sg_p)
            print(f"  [info] Filtro Savitzky-Golay aplicado a S11 de '{nombre}' (ventana={sg_w}, orden={sg_p}).")

    default_metodos = ['simplificado', 'completo']
    if calibracion_mc is not None:
        default_metodos.append('minimos_cuadrados')
    metodos = material.get('metodos', default_metodos)
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
                prefijo="Medido (completo, 4 patrones", verbose=True,
                gn_original=gn_original, gn_acondicionado=gn_acondicionado,
                comparar_gn=comparar_gn))
    if 'minimos_cuadrados' in metodos and calibracion_mc is not None:
        resultados.update(_calcular_metodo_mc(
            frecs, s11_dut, estrategia_gn, calibracion_mc,
            calibracion_mc_acond=calibracion_mc_acond,
            comparar_gn=comparar_gn, verbose=True))

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
                          log_x=log_x, etiqueta_teorico=etiq_teorico,
                          limitar_eje_y=limitar_eje_y)

    # S11 medido (crudo, previo a la conversion a permitividad): modulo+fase
    # y diagrama de Smith, en el mismo recorte de frecuencias y la misma
    # escala log_x que el grafico de arriba, para poder correlacionar
    # cualquier anomalia de er'/er'' con lo que se ve directamente en S11
    # (p.ej. una resonancia de la sonda o un problema de contacto).
    s11_dut_r = s11_dut[mascara]
    if filtrar_s11:
        s11_orig_r = s11_dut_orig[mascara]
        et_filt = f"{nombre} (filtrado PB)" if tipo_filtro == 'pasabajos' else f"{nombre} (filtrado SG)"
        datos_s11_r = {
            f"{nombre} (original)": {'Frec': f_r, 'Complex': s11_orig_r},
            et_filt: {'Frec': f_r, 'Complex': s11_dut_r},
        }
    else:
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
        print(f"  Error absoluto:")
        errores = {}
        comparar_im = perdidas_comparables(modelo)
        for etiqueta, er in resultados_r.items():
            errores[etiqueta] = reportar_error_intervalos(etiqueta, f_r, er, er_teo_r,
                                                          intervalos_error,
                                                          comparar_imag=comparar_im)
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
        'modelo_perdidas_comparables': perdidas_comparables(modelo) if modelo else True,
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


def _advertencias_calibracion(modelo_patron3, modelo_patron4, adicionales=()):
    """
    Junta las advertencias de los modelos teoricos usados como patron 3,
    patron 4 y patrones adicionales de la calibracion, en una lista de
    dicts listos para mostrar (consola, GUI o caja destacada del informe PDF).

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
    pares = [("Patron 3", modelo_patron3), ("Patron 4", modelo_patron4)]
    for idx_ad, ad in enumerate(adicionales or (), start=5):
        pares.append((f"Patron {idx_ad} (cuadrados minimos)", ad.get('modelo')))
    for rol, modelo in pares:
        if not modelo:
            continue
        texto = advertencia_patron(modelo)
        info_extra = ""
        if texto:
            info_extra = (
                f"\n\nUSADO COMO {rol.upper()} DE CALIBRACION: ademas de lo "
                f"anterior, tener en cuenta que la permitividad de este "
                f"patron entra DIRECTAMENTE en las formulas de conversion "
                f"S11 -> er, asi que cualquier limitacion del modelo se "
                f"propaga al resultado de TODOS los materiales, no solo al "
                f"de este patron."
            )
            if es_modelo_estatico(modelo):
                info_extra += (
                    " En particular, un liquido no polar tiene permitividad "
                    "cercana a la del aire: como 3er patron de una "
                    "calibracion de solo 3 queda mal condicionado, aunque "
                    "Kaatze (2007) si lo recomienda como patron ADICIONAL "
                    "para medir bien muestras de baja permitividad."
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
    usar_metodo_completo = bool(config.get('usar_metodo_completo', True))
    usar_mc = bool(config.get('usar_minimos_cuadrados', False))
    adicionales = [p for p in (config.get('patrones_adicionales') or []) if p.get('archivo')]
    estrategia_gn = config.get('estrategia_gn', 'minimo_gn')
    f_min_ghz = config['f_min_ghz']
    f_max_ghz = config['f_max_ghz']
    f_int1_ghz = config.get('f_err_int1_ghz')
    f_int2_ghz = config.get('f_err_int2_ghz')
    intervalos_error = construir_intervalos_error(f_min_ghz, f_max_ghz, f_int1_ghz, f_int2_ghz)
    log(f"  Intervalos de error para el informe: " +
        ", ".join([f"[{fa/1e9:.2f} - {fb/1e9:.2f}] GHz" for fa, fb in intervalos_error]))
    log_x = config.get('escala_log_frecuencia', ESCALA_LOG_FRECUENCIA)
    limitar_eje_y = bool(config.get('limitar_eje_y', True))
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
        log("  Metodo completo deshabilitado: no se usa Patron 4 ni se calcula Gn de 4 patrones.")

    cfg_filtro_pat = config.get('filtrar_patrones_sg') or config.get('filtrar_patrones') or {}
    _filtrar_pat = None
    if cfg_filtro_pat.get('habilitado', False):
        tipo_filtro_pat = cfg_filtro_pat.get('tipo', cfg_filtro_pat.get('tipo_filtro', 'savitzky_golay'))
        sg_w_pat = int(cfg_filtro_pat.get('sg_window', 15) or 15)
        sg_p_pat = int(cfg_filtro_pat.get('sg_polyorder', 3) or 3)
        pb_ord_pat = int(cfg_filtro_pat.get('pb_orden', 2) or 2)
        pb_fc_pat = float(cfg_filtro_pat.get('pb_fcorte', 0.10) or 0.10)
        aplicar_en = cfg_filtro_pat.get('aplicar_en', ['corto', 'aire', 'patron3', 'patron4', 'adicionales'])

        def _filtrar_pat_fn(s11_arr, nombre_pat):
            if tipo_filtro_pat == 'pasabajos':
                s11_f = filtrar_s11_pasabajos(s11_arr, orden=pb_ord_pat, f_corte=pb_fc_pat)
                log(f"  [info] Filtro Pasabajos aplicado a '{nombre_pat}' (orden={pb_ord_pat}, f_corte={pb_fc_pat:.2f}).")
            else:
                s11_f = filtrar_s11_savgol(s11_arr, window_length=sg_w_pat, polyorder=sg_p_pat)
                log(f"  [info] Filtro Savitzky-Golay aplicado a '{nombre_pat}' (ventana={sg_w_pat}, orden={sg_p_pat}).")
            return s11_f
        _filtrar_pat = _filtrar_pat_fn

        if 'corto' in aplicar_en:
            s11_corto = _filtrar_pat(s11_corto, 'corto')
        if 'aire' in aplicar_en:
            s11_aire = _filtrar_pat(s11_aire, 'aire')
        if 'patron3' in aplicar_en:
            s11_patron3 = _filtrar_pat(s11_patron3, f"patron3 ({etiqueta_p3})")
        if s11_patron4 is not None and 'patron4' in aplicar_en:
            s11_patron4 = _filtrar_pat(s11_patron4, f"patron4 ({etiqueta_p4})")

    _copiar_insumos(carpeta_datos, archivos_cal, materiales, carpeta_dia, log,
                    patrones_adicionales=adicionales if usar_mc else ())

    for adv in _advertencias_calibracion(
            modelo_patron3,
            modelo_patron4 if usar_metodo_completo else None,
            adicionales=adicionales if usar_mc else ()):
        log(f"\n  [ADVERTENCIA - {adv['rol']}: {adv['etiqueta']}]")
        for linea in adv['texto'].splitlines():
            if linea.strip():
                log(f"    {linea.strip()}")

    # -----------------------------------------------------------------
    # Configuracion de acondicionamiento de Gn(f)
    # -----------------------------------------------------------------
    cfg_acond = config.get('acondicionar_gn') or {}
    acond_habilitado = bool(cfg_acond.get('habilitado', False))
    tipo_acond = cfg_acond.get('tipo', 'transicion_polinomica_cero')
    modo_acond = cfg_acond.get('modo', 'automatico')
    f1_acond = float(cfg_acond.get('f1_ghz', 0.0) or 0.0) * 1e9
    f2_acond = float(cfg_acond.get('f2_ghz', 0.5) or 0.5) * 1e9
    umbral_acond = float(cfg_acond.get('umbral_pct', 30.0) or 30.0)
    sg_w_acond = int(cfg_acond.get('sg_window', 15) or 15)
    sg_p_acond = int(cfg_acond.get('sg_polyorder', 3) or 3)
    pb_ord_acond = int(cfg_acond.get('pb_orden', 2) or 2)
    pb_fc_acond = float(cfg_acond.get('pb_fcorte', 0.10) or 0.10)
    comparar_gn = bool(cfg_acond.get('comparar_original', True)) if acond_habilitado else False

    # -----------------------------------------------------------------
    # Diagnostico y acondicionamiento de Gn(f) (si el metodo completo
    # esta habilitado)
    # -----------------------------------------------------------------
    gn_original = None
    gn_acondicionado = None
    info_acond = None
    Gn_cal = None

    if usar_metodo_completo:
        log("Calculando conductancia normalizada Gn(f) (diagnostico de calibracion)...")
        Gn_cal = calcular_Gn(frecs, s11_patron3, s11_aire, s11_patron4, s11_corto,
                              Er_patron3, Er_patron4)
        gn_original = Gn_cal

        if acond_habilitado:
            log(f"  Acondicionando Gn(f) (4 patrones): tipo='{tipo_acond}', modo='{modo_acond}'...")
            gn_acondicionado, info_acond = acondicionar_gn(
                frecs, Gn_cal, modo=modo_acond, tipo=tipo_acond,
                f1=f1_acond, f2=f2_acond, umbral_pct=umbral_acond,
                sg_window=sg_w_acond, sg_polyorder=sg_p_acond,
                pb_orden=pb_ord_acond, pb_fcorte=pb_fc_acond)
            if info_acond.get('aviso'):
                log(f"    {info_acond['aviso']}")
            if info_acond.get('disparo_detectado'):
                if info_acond.get('f_corte_aplicada'):
                    log(f"    Corte aplicado en: {info_acond['f_corte_aplicada']/1e9:.3f} GHz")
                if info_acond.get('f2_aplicada'):
                    log(f"    Transicion aplicada entre {info_acond['f1_aplicada']/1e9:.3f} GHz y {info_acond['f2_aplicada']/1e9:.3f} GHz")

    # -----------------------------------------------------------------
    # Calibracion redundante por cuadrados minimos (opcional):
    # ajusta (a, b, c, Gn) usando todos los patrones disponibles a la vez.
    # Si `acondicionar_gn` esta activo, acondiciona el Gn obtenido por
    # cuadrados minimos y reajusta (a, b, c) con ese Gn fijo.
    # -----------------------------------------------------------------
    mascara = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
    calibracion_mc = None
    calibracion_mc_acond = None
    chequeo_mc = None
    if usar_mc:
        patrones_mc = [
            {'nombre': "Corto", 'S11': s11_corto, 'er': None},
            {'nombre': "Aire", 'S11': s11_aire,
             'er': np.full(len(frecs), ER_AIRE, dtype=complex), 'es_aire': True},
            {'nombre': f"{etiqueta_p3} ({T_patron3:.0f}\u00b0C)",
             'S11': s11_patron3, 'er': Er_patron3, 'modelo': modelo_patron3},
        ]
        if usar_metodo_completo and s11_patron4 is not None:
            patrones_mc.append({
                'nombre': f"{etiqueta_p4} ({T_patron4:.0f}\u00b0C)",
                'S11': s11_patron4, 'er': Er_patron4, 'modelo': modelo_patron4,
            })
        for idx_ad, ad in enumerate(adicionales, start=len(patrones_mc) + 1):
            ruta_ad = os.path.join(carpeta_datos, ad['archivo'])
            datos_ad = _cargar_cache(ruta_ad, cache)
            frec_ad = np.asarray(datos_ad['Frec'], dtype=float)
            if not (len(frec_ad) == len(frecs) and np.allclose(frec_ad, frecs, atol=1.0)):
                log(f"  [info] patron adicional '{ad['archivo']}' remuestreado a la grilla de calibracion.")
                datos_ad = resamplear(frecs, datos_ad)
            s11_ad = datos_ad['Complex']
            if _filtrar_pat is not None and 'adicionales' in cfg_filtro_pat.get('aplicar_en', []):
                s11_ad = _filtrar_pat(s11_ad, f"adicional ({ad['archivo']})")
            modelo_ad = ad['modelo']
            T_ad = float(ad.get('temperatura_c', 25.0))
            etiq_ad = etiqueta_patron(modelo_ad) or modelo_ad
            er_ad = PATRONES_TEORICOS[modelo_ad](frecs, T=T_ad)
            log(f"  Patron {idx_ad} (cuadrados minimos) = {etiq_ad} a {T_ad:.1f}\u00b0C <- {ad['archivo']}")
            patrones_mc.append({
                'nombre': f"{etiq_ad} ({T_ad:.0f}\u00b0C)",
                'S11': s11_ad,
                'er': er_ad,
                'modelo': modelo_ad,
            })

        if len(patrones_mc) < 4:
            log("  [aviso] cuadrados minimos requiere al menos 4 patrones "
                "(corto + aire + 2 liquidos): se desactiva en esta corrida.")
        else:
            log(f"\nCalibrando por cuadrados minimos con {len(patrones_mc)} patrones "
                f"({2 * len(patrones_mc) - 8} grados de libertad reales por frecuencia)...")
            calibracion_mc = calibrar_minimos_cuadrados(frecs, patrones_mc)
            for nom, res_arr in calibracion_mc['residuos'].items():
                rms = float(np.sqrt(np.mean(np.abs(res_arr[mascara]) ** 2)))
                log(f"  residuo rms S11 [{nom:28s}] = {rms:.4e}")

            info_acond_mc = None
            gn_mc_acond = None
            if acond_habilitado:
                log(f"  Acondicionando Gn(f) de cuadrados minimos y reajustando (a, b, c)...")
                gn_mc_acond, info_acond_mc = acondicionar_gn(
                    frecs, calibracion_mc['Gn'], modo=modo_acond, tipo=tipo_acond,
                    f1=f1_acond, f2=f2_acond, umbral_pct=umbral_acond,
                    sg_window=sg_w_acond, sg_polyorder=sg_p_acond,
                    pb_orden=pb_ord_acond, pb_fcorte=pb_fc_acond)
                calibracion_mc_acond = reajustar_abc_con_gn_fijo(
                    frecs, patrones_mc, gn_mc_acond, calibracion_base=calibracion_mc)
                if info_acond_mc.get('aviso'):
                    log(f"    {info_acond_mc['aviso']}")
                # Si el metodo de 4 patrones estaba deshabilitado, usamos el Gn de
                # cuadrados minimos para el grafico de diagnostico de Gn(f).
                if Gn_cal is None:
                    Gn_cal = calibracion_mc['Gn']
                    gn_original = Gn_cal
                    gn_acondicionado = gn_mc_acond
                    info_acond = info_acond_mc

            validacion = validacion_cruzada_minimos_cuadrados(
                frecs, patrones_mc, cfg_acond=cfg_acond if acond_habilitado else None)
            errores_vc = {}
            if validacion:
                log("  Validacion cruzada (sacando 1 patron liquido por vez y midiendo su er):")
                for nom, v in validacion.items():
                    comp_im = perdidas_comparables(v.get('modelo')) if v.get('modelo') else True
                    etiq_vc = f"Al excluir {nom}"
                    err_int = reportar_error_intervalos(
                        etiq_vc,
                        frecs[mascara],
                        v['er_predicho'][mascara],
                        v['er_teorico'][mascara],
                        intervalos_error,
                        comparar_imag=comp_im,
                    )
                    v['errores_intervalos'] = err_int
                    errores_vc[nom] = err_int

            residuos_recortados = {
                nom: res[mascara] for nom, res in calibracion_mc['residuos'].items()
            }
            residuo_rms_recortado = {
                nom: float(np.sqrt(np.mean(np.abs(res) ** 2)))
                for nom, res in residuos_recortados.items()
            }
            ruta_fig_mc = os.path.join(
                carpeta_salida, "calibracion_minimos_cuadrados_residuos.png")
            graficar_residuos_minimos_cuadrados(
                frecs[mascara], residuos_recortados, ruta_fig_mc, log_x=log_x)
            chequeo_mc = {
                'n_patrones': len(patrones_mc),
                'grados_libertad': calibracion_mc['grados_libertad'],
                'nombres': calibracion_mc['nombres'],
                'residuo_rms': residuo_rms_recortado,
                'validacion_cruzada': validacion,
                'errores_validacion_cruzada': errores_vc,
                'acondicionado': calibracion_mc_acond is not None,
                'info_acond_mc': info_acond_mc,
                'figura_residuos': ruta_fig_mc,
                'datos_grafico_residuos': {
                    'frecs': frecs[mascara],
                    'residuos': residuos_recortados,
                    'log_x': log_x,
                },
            }

    # -----------------------------------------------------------------
    # Chequeo de calibracion: el patron 3 ES uno de los patrones, asi que
    # si se lo "mide" como si fuera un DUT mas, el resultado tiene que
    # coincidir casi exactamente con su propio modelo teorico.
    # -----------------------------------------------------------------
    log(f"\nChequeo de calibracion ({etiqueta_p3} medido vs. teorico, debe dar ~0 de error)...")
    er_p3_teo = Er_patron3
    er_p3_simpl = get_er_DUTm(frecs, s11_patron3, s11_patron3, s11_aire, s11_corto,
                               Er_patron3)

    medido_p3 = {"Medido (simplificado)": er_p3_simpl[mascara]}
    comparar_im_p3 = perdidas_comparables(modelo_patron3)
    errores_p3 = {
        "Metodo simplificado": reportar_error_intervalos(
            "Metodo simplificado", frecs[mascara], er_p3_simpl[mascara], er_p3_teo[mascara],
            intervalos_error, comparar_imag=comparar_im_p3),
    }
    if usar_metodo_completo:
        resultados_p3_completo = _calcular_metodo_completo(
            frecs, s11_patron3, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, estrategia_gn, prefijo="Medido (completo",
            gn_original=gn_original, gn_acondicionado=gn_acondicionado,
            comparar_gn=comparar_gn)
        for etiqueta, arr in resultados_p3_completo.items():
            medido_p3[etiqueta] = arr[mascara]
            errores_p3[etiqueta] = reportar_error_intervalos(
                etiqueta, frecs[mascara], arr[mascara], er_p3_teo[mascara],
                intervalos_error, comparar_imag=comparar_im_p3)
    if calibracion_mc is not None:
        resultados_p3_mc = _calcular_metodo_mc(
            frecs, s11_patron3, estrategia_gn, calibracion_mc,
            calibracion_mc_acond=calibracion_mc_acond,
            comparar_gn=comparar_gn)
        for etiqueta, arr in resultados_p3_mc.items():
            medido_p3[etiqueta] = arr[mascara]
            errores_p3[etiqueta] = reportar_error_intervalos(
                etiqueta, frecs[mascara], arr[mascara], er_p3_teo[mascara],
                intervalos_error, comparar_imag=comparar_im_p3)

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
        limitar_eje_y=limitar_eje_y,
    )
    chequeo_calibracion = {
        'figura': ruta_figura_p3,
        'etiqueta_patron3': etiqueta_p3,
        'etiqueta_patron4': etiqueta_p4,
        'errores': errores_p3,
        'datos_grafico': {
            'frecs': frecs[mascara],
            'medido': medido_p3,
            'teorico': er_p3_teo[mascara],
            'titulo': f"{etiqueta_p3}: chequeo de calibracion (medido vs. teorico)",
            'log_x': log_x,
            'etiqueta_teorico': etiq_teorico_p3,
            'limitar_eje_y': limitar_eje_y,
        },
    }

    # -----------------------------------------------------------------
    # Chequeo de calibracion ANALOGO para el patron 4 (si esta habilitado)
    # -----------------------------------------------------------------
    if usar_metodo_completo:
        log(f"\nChequeo de calibracion ({etiqueta_p4} medido vs. teorico, debe dar ~0 de error)...")
        er_p4_teo = Er_patron4
        er_p4_simpl = get_er_DUTm(frecs, s11_patron4, s11_patron3, s11_aire, s11_corto,
                                   Er_patron3)
        medido_p4 = {"Medido (simplificado)": er_p4_simpl[mascara]}
        comparar_im_p4 = perdidas_comparables(modelo_patron4)
        errores_p4 = {
            "Metodo simplificado": reportar_error_intervalos(
                "Metodo simplificado", frecs[mascara], er_p4_simpl[mascara], er_p4_teo[mascara],
                intervalos_error, comparar_imag=comparar_im_p4),
        }
        resultados_p4_completo = _calcular_metodo_completo(
            frecs, s11_patron4, s11_patron3, s11_aire, s11_patron4, s11_corto,
            Er_patron3, Er_patron4, estrategia_gn, prefijo="Medido (completo",
            gn_original=gn_original, gn_acondicionado=gn_acondicionado,
            comparar_gn=comparar_gn)
        for etiqueta, arr in resultados_p4_completo.items():
            medido_p4[etiqueta] = arr[mascara]
            errores_p4[etiqueta] = reportar_error_intervalos(
                etiqueta, frecs[mascara], arr[mascara], er_p4_teo[mascara],
                intervalos_error, comparar_imag=comparar_im_p4)
        if calibracion_mc is not None:
            resultados_p4_mc = _calcular_metodo_mc(
                frecs, s11_patron4, estrategia_gn, calibracion_mc,
                calibracion_mc_acond=calibracion_mc_acond,
                comparar_gn=comparar_gn)
            for etiqueta, arr in resultados_p4_mc.items():
                medido_p4[etiqueta] = arr[mascara]
                errores_p4[etiqueta] = reportar_error_intervalos(
                    etiqueta, frecs[mascara], arr[mascara], er_p4_teo[mascara],
                    intervalos_error, comparar_imag=comparar_im_p4)

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
            limitar_eje_y=limitar_eje_y,
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
                'limitar_eje_y': limitar_eje_y,
            },
        }

    # -----------------------------------------------------------------
    # Grafico de diagnostico de Gn(f) (original y/o acondicionado)
    # -----------------------------------------------------------------
    if (usar_metodo_completo or usar_mc) and Gn_cal is not None:
        marcas_inicio_gn = {}
        if estrategia_gn in ('minimo_gn', 'comparar_ambas'):
            gn_referencia_marcas = gn_acondicionado if gn_acondicionado is not None else Gn_cal
            idx_auto = int(np.argmin(np.abs(gn_referencia_marcas)))
            marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['minimo_gn']] = frecs[idx_auto]
        if estrategia_gn in ('frecuencia_maxima', 'comparar_ambas'):
            idx_clasico = int(np.argmax(frecs))
            marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['frecuencia_maxima']] = frecs[idx_clasico]

        ruta_figura_Gn = os.path.join(carpeta_salida, "chequeo_calibracion_Gn.png")
        graficar_Gn(frecs[mascara], Gn_cal[mascara], ruta_figura_Gn, log_x=log_x,
                    marcas_inicio=marcas_inicio_gn,
                    Gn_acondicionado=(gn_acondicionado[mascara] if gn_acondicionado is not None else None),
                    info_acond=info_acond)
        chequeo_calibracion['figura_Gn'] = ruta_figura_Gn
        chequeo_calibracion['info_acond_Gn'] = info_acond
        chequeo_calibracion['datos_grafico_Gn'] = {
            'frecs': frecs[mascara], 'Gn': Gn_cal[mascara],
            'Gn_acondicionado': (gn_acondicionado[mascara] if gn_acondicionado is not None else None),
            'log_x': log_x, 'marcas_inicio': marcas_inicio_gn,
            'info_acond': info_acond,
        }

    if chequeo_mc is not None:
        chequeo_calibracion['minimos_cuadrados'] = chequeo_mc

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
            gn_original=gn_original, gn_acondicionado=gn_acondicionado,
            comparar_gn=comparar_gn,
            intervalos_error=intervalos_error,
            calibracion_mc=calibracion_mc,
            calibracion_mc_acond=calibracion_mc_acond,
            limitar_eje_y=limitar_eje_y,
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
        'estrategia_gn': estrategia_gn if (usar_metodo_completo or usar_mc) else None,
        'filtrar_patrones_sg': cfg_filtro_pat,
        'acondicionar_gn': cfg_acond,
        'info_acond_Gn': info_acond,
        'usar_minimos_cuadrados': usar_mc and chequeo_mc is not None,
        'minimos_cuadrados': chequeo_mc,
        'f_min_ghz': f_min_ghz,
        'f_max_ghz': f_max_ghz,
        'intervalos_error': intervalos_error,
        'archivos_calibracion': archivos_cal,
        'cancelado': interrumpido,
        'advertencias_calibracion': _advertencias_calibracion(
            modelo_patron3,
            modelo_patron4 if usar_metodo_completo else None,
            adicionales=adicionales if usar_mc else ()),
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
        'usar_minimos_cuadrados': USAR_MINIMOS_CUADRADOS,
        'patrones_adicionales': list(PATRONES_ADICIONALES),
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