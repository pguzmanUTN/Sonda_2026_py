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
1. Poner los archivos .s1p de calibracion (corto, aire, agua, alcohol
   isopropilico) en la misma carpeta que este script, y completar
   ARCHIVOS_CALIBRACION mas abajo con sus nombres.
2. Ajustar TEMPERATURA_AGUA_C a la temperatura real del agua destilada
   durante la calibracion (afecta la precision, sobre todo del metodo
   simplificado).
3. Agregar cada material que se quiera analizar a la lista MATERIALES
   (ver mas abajo). Para un material sin modelo teorico conocido, dejar
   'teorico': None -- el script va a graficar unicamente lo medido, sin
   intentar calcular error.
4. Correr:  python analisis_permitividad.py

Como agregar una medicion nueva (por ejemplo, acetona)
-------------------------------------------------------
Alcanza con sumar un diccionario a la lista MATERIALES, no hace falta
tocar nada mas del script:

    {
        'nombre': "Acetona",
        'archivo': "sonda4-acetona.s1p",
        'teorico': None,   # no hay modelo teorico cargado en Patrones.py
    }

Si en el futuro se agrega un modelo teorico para ese material en
`Patrones.py`, alcanza con poner esa funcion en lugar de `None` y el
script automaticamente empieza a graficar la comparacion y el error.

Patrones que dependen de la temperatura
----------------------------------------
Desde que Patrones.py soporta varias temperaturas (NPL Report MAT 23),
cada funcion teorica acepta `get_er_pat_xxx(frecs, T=...)`. Como el campo
'teorico' de MATERIALES tiene que ser una funcion de UN solo argumento
(frecs), la temperatura de cada material se "ata" con functools.partial,
tal como se hace mas abajo con TEMPERATURA_ALC_ETILICO_C, etc. Para medir
a otra temperatura, alcanza con cambiar esa constante (no hace falta
tocar Patrones.py). Si la temperatura real no cae justo en un escalon de
5 °C tabulado en el reporte, la funcion usa el mas cercano disponible
(y avisa con un warning si ademas cae fuera del rango recomendado).
"""
import os
import re
import csv
import functools
import numpy as np
import matplotlib.pyplot as plt

from Touchstone import (leer_s1p, mismas_frecuencias, resamplear,
                         graficar_s11_mag_fase, graficar_smith)
from funciones import get_er_DUTm, get_er_DUT_completo, error_relativo_porcentual, calcular_Gn
from Patrones import (
    get_er_agua,
    get_er_pat_alc_etilico,
    get_er_pat_alc_isopropilico,
    get_er_pat_metanol,
    get_er_pat_dmso,
    get_er_pat_etilenglicol,
    get_er_pat_butanol,
    get_er_pat_propanol,
    PATRONES_TEORICOS,
    etiqueta_patron,
)
from reporte_pdf import generar_reporte_pdf

# ---------------------------------------------------------------------------
# CONFIGURACION -- editar segun corresponda
# ---------------------------------------------------------------------------
CARPETA_DATOS = "."   # carpeta donde estan los .s1p

# Patrones de calibracion: estos 4 son siempre necesarios (el metodo
# completo usa los 4; el simplificado usa corto/aire/agua).
ARCHIVOS_CALIBRACION = {
    'corto':       "sonda4-short.s1p",
    'aire':        "sonda4-aire.s1p",
    'agua':        "sonda4-agua.s1p",
    'alc_isoprop': "sonda4-alc-isoprop.s1p",
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

# Temperatura real de cada liquido durante el ensayo (°C). Ajustar segun
# corresponda: cada patron teorico en Patrones.py aproxima internamente a
# la temperatura tabulada mas cercana (pasos de 5 °C, NPL MAT 23), asi que
# no hace falta que coincida exactamente con un valor de la tabla.
TEMPERATURA_AGUA_C = 25.0

# Temperatura del alcohol isopropilico usado como 4to PATRON DE CALIBRACION
# (metodo completo). Es distinta de TEMPERATURA_ALC_ISOPROP_C de aca abajo
# (la de la entrada en MATERIALES, que es cuando se lo analiza COMO SI
# fuera un material mas, para comparar contra su propia curva teorica) --
# aunque en general van a ser el mismo valor, porque casi siempre es la
# misma muestra fisica medida una sola vez. Esta temperatura entra en el
# calculo de Gn (funciones.get_er_DUT_completo), asi que afecta el
# resultado de TODOS los materiales analizados con el metodo completo, no
# solo el del alcohol isopropilico.
TEMPERATURA_ALC_ISOPROP_CAL_C = 25.0

TEMPERATURA_ALC_ETILICO_C = 20.0
TEMPERATURA_ALC_ISOPROP_C = 25.0
TEMPERATURA_METANOL_C = 25.0
TEMPERATURA_DMSO_C = 25.0
TEMPERATURA_ETILENGLICOL_C = 25.0
TEMPERATURA_BUTANOL_C = 25.0
TEMPERATURA_PROPANOL_C = 25.0

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
    #     'archivo': "Alcohol.s1p",
    #     'modelo': None,
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
    grilla de frecuencias (remuestrea contra 'agua' si hiciera falta)."""
    datos = {}
    for clave, nombre in archivos.items():
        ruta = os.path.join(carpeta, nombre)
        datos[clave] = _cargar_cache(ruta, cache)
        print(f"  {clave:12s} <- {nombre}  "
              f"({len(datos[clave]['Frec'])} puntos, "
              f"{datos[clave]['Frec'][0]/1e9:.3f}-{datos[clave]['Frec'][-1]/1e9:.3f} GHz)")

    referencia = datos['agua']
    if not mismas_frecuencias(*datos.values()):
        print("  [info] los patrones de calibracion no comparten exactamente "
              "la misma grilla de frecuencias: remuestreando todo a la "
              "grilla de 'agua'.")
        for clave in datos:
            if clave != 'agua':
                datos[clave] = resamplear(referencia['Frec'], datos[clave])

    return datos


def graficar_comparacion(frecs, er_medido_dict, er_teorico, titulo, archivo_salida,
                          log_x=True):
    """
    Grafica parte real e imaginaria: curva medida (una o varias) y, si se
    conoce, la curva teorica de referencia.

    er_medido_dict : dict {etiqueta: ndarray complejo}
    er_teorico     : ndarray complejo, o None si no hay modelo teorico
                      (en ese caso se grafica solo lo medido).
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
    """
    fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    if er_teorico is not None:
        ax_re.plot(frecs / 1e9, er_teorico.real, 'k--', linewidth=2, label="Teorico (Debye)")
    for etiqueta, er in er_medido_dict.items():
        ax_re.plot(frecs / 1e9, er.real, linewidth=1.6, label=etiqueta)
    ax_re.set_ylabel("er'  (parte real)")
    ax_re.set_title(titulo)
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_re.legend()

    if er_teorico is not None:
        ax_im.plot(frecs / 1e9, er_teorico.imag, 'k--', linewidth=2, label="Teorico (Debye)")
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
    fig.savefig(archivo_salida, dpi=150)
    print(f"  Figura guardada: {archivo_salida}")
    plt.close(fig)
    return archivo_salida


def graficar_Gn(frecs, Gn, archivo_salida, log_x=True):
    """
    Grafica la conductancia normalizada Gn(f) calculada a partir de los 4
    patrones de calibracion (ver `funciones.calcular_Gn`). Es un
    diagnostico de la CALIBRACION, no de un material en particular: Gn no
    depende del DUT, asi que esta misma curva aplica a todos los
    materiales analizados con el metodo completo en esta corrida.

    Al estar relacionada con G0/(j*w*C0) -- una propiedad fisica continua
    de la sonda -- Gn(f) deberia verse suave. Un salto brusco o un pico
    aislado suele indicar un problema con la medicion de alguno de los 4
    patrones, tipicamente el alcohol isopropilico (el unico patron que
    entra en el calculo de Gn y en ningun otro lado del metodo
    simplificado, por lo que un problema ahi puede pasar desapercibido si
    solo se mira el chequeo de calibracion del agua).
    """
    fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_re.plot(frecs / 1e9, Gn.real, linewidth=1.6, color='tab:purple')
    ax_re.set_ylabel("Re(Gn)")
    ax_re.set_title("Conductancia normalizada Gn(f) \u2014 diagnostico de calibracion")
    ax_re.minorticks_on()
    ax_re.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_re.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    ax_im.plot(frecs / 1e9, Gn.imag, linewidth=1.6, color='tab:purple')
    ax_im.set_xlabel("Frecuencia (GHz)")
    ax_im.set_ylabel("Im(Gn)")
    ax_im.minorticks_on()
    ax_im.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_im.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    if log_x and frecs.size > 0 and np.all(frecs > 0):
        ax_re.set_xscale('log')
        ax_im.set_xscale('log')

    fig.tight_layout()
    fig.savefig(archivo_salida, dpi=150)
    print(f"  Figura guardada: {archivo_salida}")
    plt.close(fig)
    return archivo_salida


def calcular_error(er_medido, er_teorico):
    """Devuelve un dict con el error relativo (%) promedio y maximo de
    parte real e imaginaria, listo para tablas (consola o PDF)."""
    err_re, err_im = error_relativo_porcentual(er_medido, er_teorico)
    return {
        'err_re_medio': float(np.mean(np.abs(err_re))),
        'err_re_max': float(np.max(np.abs(err_re))),
        'err_im_medio': float(np.mean(np.abs(err_im))),
        'err_im_max': float(np.max(np.abs(err_im))),
    }


def reportar_error(nombre, er_medido, er_teorico):
    """Imprime el error relativo en consola y lo devuelve (ver `calcular_error`)."""
    e = calcular_error(er_medido, er_teorico)
    print(f"  {nombre}:")
    print(f"    error medio  er' = {e['err_re_medio']:6.2f}%   "
          f"error medio  er'' = {e['err_im_medio']:6.2f}%")
    print(f"    error maximo er' = {e['err_re_max']:6.2f}%   "
          f"error maximo er'' = {e['err_im_max']:6.2f}%")
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


def procesar_material(material, carpeta_datos, cache, frecs,
                       s11_corto, s11_aire, s11_agua, s11_isoprop_cal,
                       T_agua=None, T_isoprop=None, carpeta_salida=None,
                       f_min_ghz=None, f_max_ghz=None, log_x=None):
    """Procesa una entrada de MATERIALES: lee su .s1p, calcula la
    permitividad con los metodos indicados, grafica (contra la curva
    teorica si se conoce, o solo lo medido si no), exporta CSV y devuelve
    un dict con todo lo necesario para armar el informe PDF (o None si el
    archivo no se encontro, en cuyo caso se salta este material).

    `T_agua`, `T_isoprop`, `carpeta_salida`, `f_min_ghz`, `f_max_ghz`,
    `log_x`: si se omiten (None), se usan las constantes definidas mas
    arriba en este archivo (comportamiento identico al de antes).
    `ejecutar_analisis` los pasa explicitamente para poder usar valores
    distintos (p.ej. los que cargo la GUI) sin tocar esas constantes."""
    T_agua = TEMPERATURA_AGUA_C if T_agua is None else T_agua
    T_isoprop = TEMPERATURA_ALC_ISOPROP_CAL_C if T_isoprop is None else T_isoprop
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
            frecs, s11_dut, s11_agua, s11_aire, s11_corto,
            T_agua=T_agua)
    if 'completo' in metodos:
        resultados['Medido (completo, 4 patrones)'] = get_er_DUT_completo(
            frecs, s11_dut, s11_agua, s11_aire, s11_isoprop_cal, s11_corto,
            T_agua=T_agua, T_isoprop=T_isoprop, verbose=True)

    modelo = material.get('modelo') or None
    temperatura_material = float(material.get('temperatura', 25.0)) if modelo else None
    teorico_fn = resolver_modelo_teorico(modelo, temperatura_material)
    er_teo = teorico_fn(frecs) if teorico_fn is not None else None

    mascara = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
    f_r = frecs[mascara]
    resultados_r = {k: v[mascara] for k, v in resultados.items()}
    er_teo_r = er_teo[mascara] if er_teo is not None else None

    slug = _slug(nombre)
    titulo = f"{nombre}: medido vs. teorico" if er_teo_r is not None else \
             f"{nombre}: medido (sin modelo teorico de referencia)"
    ruta_figura = os.path.join(carpeta_salida, f"er_{slug}.png")

    graficar_comparacion(f_r, resultados_r, er_teo_r, titulo, ruta_figura, log_x=log_x)

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
    }


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
                                  'aire', 'agua', 'alc_isoprop'.
        'temperatura_agua_c'   : float, temperatura del agua de calibracion.
        'temperatura_isoprop_cal_c' : float, temperatura del alcohol
                                isopropilico usado como 4to patron de
                                calibracion (metodo completo). Si falta
                                esta clave (config viejo), se usa
                                TEMPERATURA_ALC_ISOPROP_CAL_C.
        'materiales'           : lista de dicts {nombre, archivo, modelo,
                                  temperatura, metodos} (ver formato de
                                  MATERIALES mas arriba).
        'carpeta_salida'       : str, carpeta donde se guardan figuras,
                                  CSVs e informe PDF.
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

    Retorna
    -------
    dict con 'chequeo_calibracion', 'resultados_materiales',
    'ruta_informe_pdf' y 'cancelado' (bool: True si se interrumpio a
    mitad de camino por el motivo de arriba).
    """
    if progreso is None:
        progreso = lambda paso, total, etiqueta=None: None
    if cancelado is None:
        cancelado = lambda: False

    carpeta_datos = config['carpeta_datos']
    carpeta_salida = config['carpeta_salida']
    os.makedirs(carpeta_salida, exist_ok=True)
    cache = {}

    T_agua = config['temperatura_agua_c']
    T_isoprop = config.get('temperatura_isoprop_cal_c', TEMPERATURA_ALC_ISOPROP_CAL_C)
    f_min_ghz = config['f_min_ghz']
    f_max_ghz = config['f_max_ghz']
    log_x = config.get('escala_log_frecuencia', ESCALA_LOG_FRECUENCIA)
    materiales = config['materiales']
    total_pasos = len(materiales) + 2  # calibracion + N materiales + informe PDF

    if cancelado():
        log("\n[cancelado] Analisis interrumpido antes de empezar.")
        return {'chequeo_calibracion': None, 'resultados_materiales': [],
                'ruta_informe_pdf': None, 'cancelado': True}

    progreso(0, total_pasos, "Leyendo patrones de calibraci\u00f3n")
    log("Leyendo patrones de calibracion...")
    calib = cargar_calibracion(carpeta_datos, config['archivos_calibracion'], cache)
    frecs = calib['agua']['Frec']

    s11_corto = calib['corto']['Complex']
    s11_aire = calib['aire']['Complex']
    s11_agua = calib['agua']['Complex']
    s11_isoprop_cal = calib['alc_isoprop']['Complex']

    # -----------------------------------------------------------------
    # Chequeo de calibracion: el agua ES uno de los patrones, asi que si
    # se la "mide" como si fuera un DUT mas, el resultado tiene que
    # coincidir casi exactamente con su propio modelo teorico
    # (get_er_agua). Si esto no da ~0% de error, hay un problema de
    # lectura/calibracion antes de analizar cualquier material real.
    # -----------------------------------------------------------------
    log("\nChequeo de calibracion (agua medida vs. teorica, debe dar ~0% de error)...")
    er_agua_teo = get_er_agua(frecs, T_agua)
    er_agua_simpl = get_er_DUTm(frecs, s11_agua, s11_agua, s11_aire, s11_corto,
                                 T_agua=T_agua)
    er_agua_compl = get_er_DUT_completo(frecs, s11_agua, s11_agua, s11_aire,
                                         s11_isoprop_cal, s11_corto,
                                         T_agua=T_agua, T_isoprop=T_isoprop)

    mascara = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
    ruta_figura_agua = os.path.join(carpeta_salida, "chequeo_calibracion_agua.png")
    graficar_comparacion(
        frecs[mascara],
        {"Medido (simplificado)": er_agua_simpl[mascara],
         "Medido (completo)": er_agua_compl[mascara]},
        er_agua_teo[mascara],
        "Agua destilada: chequeo de calibracion (medido vs. teorico)",
        ruta_figura_agua,
        log_x=log_x,
    )
    chequeo_calibracion = {
        'figura': ruta_figura_agua,
        'errores': {
            "Metodo simplificado": reportar_error(
                "Metodo simplificado", er_agua_simpl[mascara], er_agua_teo[mascara]),
            "Metodo completo": reportar_error(
                "Metodo completo    ", er_agua_compl[mascara], er_agua_teo[mascara]),
        },
    }

    # -----------------------------------------------------------------
    # Diagnostico adicional: conductancia normalizada Gn(f), calculada
    # UNICAMENTE a partir de los 4 patrones de calibracion (no depende de
    # ningun material analizado). Tiene que verse suave; un salto brusco
    # o un pico aislado suele delatar un problema con la medicion de
    # alguno de los patrones (sobre todo el alcohol isopropilico, el
    # unico que interviene en este calculo). Ver funciones.calcular_Gn.
    # -----------------------------------------------------------------
    log("Calculando conductancia normalizada Gn(f) (diagnostico de calibracion)...")
    Gn_cal = calcular_Gn(frecs, s11_agua, s11_aire, s11_isoprop_cal, s11_corto,
                          T_agua=T_agua, T_isoprop=T_isoprop)
    ruta_figura_Gn = os.path.join(carpeta_salida, "chequeo_calibracion_Gn.png")
    graficar_Gn(frecs[mascara], Gn_cal[mascara], ruta_figura_Gn, log_x=log_x)
    chequeo_calibracion['figura_Gn'] = ruta_figura_Gn

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
            s11_corto, s11_aire, s11_agua, s11_isoprop_cal,
            T_agua=T_agua, T_isoprop=T_isoprop, carpeta_salida=carpeta_salida,
            f_min_ghz=f_min_ghz, f_max_ghz=f_max_ghz, log_x=log_x,
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
        'temperatura_agua': T_agua,
        'temperatura_isoprop': T_isoprop,
        'f_min_ghz': f_min_ghz,
        'f_max_ghz': f_max_ghz,
        'archivos_calibracion': config['archivos_calibracion'],
        'cancelado': interrumpido,
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
        'temperatura_agua_c': TEMPERATURA_AGUA_C,
        'temperatura_isoprop_cal_c': TEMPERATURA_ALC_ISOPROP_CAL_C,
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