"""
Patrones.py
-----------
Modelos teoricos (curvas de referencia) de permitividad compleja para los
liquidos patron utilizados en la calibracion de la sonda coaxial open-ended,
y para los materiales bajo ensayo (MUT) que se quieran comparar contra su
curva teorica.

Todas las funciones reciben la frecuencia en Hz (no en GHz) y devuelven un
array de numpy de numeros complejos con la convencion:

    er = er' - j*er''      (er'' > 0 representa perdidas)

Esta es la convencion que usan explicitamente los papers de referencia de
la catedra (Felicio 2016, Zechmeister 2019, Henze et al. ARGENCON 2024,
etc). Es IMPORTANTE que todos los patrones usen la misma convencion: si se
agrega un nuevo modelo teorico, la parte imaginaria debe salir negativa
para un material con perdidas (revisar el signo del termino 1j en la
formula de Debye correspondiente).

Dependencia con la temperatura (T)
-----------------------------------
Todos los patrones de liquidos puros (etanol, isopropanol, metanol, DMSO,
etilenglicol y 1-butanol/1-propanol) ahora aceptan un parametro `T` (en
grados Celsius). Los parametros de relajacion de estos modelos salen de
las tablas de la Seccion 7 del reporte:

    A P Gregory and R N Clarke, "Tables of the Complex Permittivity of
    Dielectric Reference Liquids at Frequencies up to 5 GHz",
    NPL Report MAT 23, enero 2012 (https://www.npl.co.uk).

Esas tablas solo dan los parametros ajustados en pasos de 5 °C (10, 15,
20, ..., 50 °C; DMSO arranca en 20 °C porque funde a 18 °C). Como el
reporte mismo indica que la interpolacion lineal en general alcanza
(Seccion 1.3), pero ademas para varios liquidos la ECUACION usada cambia
segun el rango de temperatura (ver notas 26 y 30), lo mas robusto es
aproximar la T pedida a la temperatura tabulada mas cercana (dentro del
rango recomendado para cada ecuacion) y usar esos parametros tal cual.
Esto es lo que hacen las funciones de este archivo: si la T pedida no
cae justo en un escalon de 5 °C, se usa el mas cercano disponible (y se
puede pasar verbose=True para ver cual se uso).

El modelo del agua (Liebe, Hufford y Manabe) es la excepcion: es una
formula analitica continua en T, no una tabla, asi que no hace falta
aproximar nada ahi.

Liquidos que NO salen de las tablas de relajacion del MAT 23
-------------------------------------------------------------
Tres liquidos del reporte no tienen parametros de relajacion ajustados en
la Seccion 7, y se modelan con otras fuentes (o con la permitividad
estatica de la Seccion 6, que es la tabla de mediciones en celda de
admitancia):

  - ACETONA: NPL no le pudo ajustar un Debye porque su relajacion queda
    muy por encima de los 5 GHz que median (nota 23). Se modela con un
    Debye simple cuyo es(T) sale de la tabla estatica de NPL y cuyos
    e_inf y fr salen de un ajuste propio sobre la figura 5 de Zarubina
    et al. (2020), que la midieron hasta 20 GHz con sonda coaxial. Ver
    el bloque de comentarios de `get_er_pat_acetona`.

  - CICLOHEXANO: molecula NO POLAR, no tiene relajacion dipolar, asi que
    no hay ni puede haber un Debye. er' es realmente constante con la
    frecuencia y lineal con la temperatura (NPL MAT 23 + ecuacion (9) de
    Kaatze 2007); er'' es minusculo y sale de la curva E de la figura 4
    de Afsar et al. (1980). Ver `get_er_pat_ciclohexano`.

  - FLUIDO DE SILICONA (polidimetilsiloxano Dow-Corning 200, 1 cSt):
    tambien no polar. No se consiguieron datos de perdidas, asi que
    queda como permitividad estatica constante con er'' = 0. Es el unico
    de los tres que sigue siendo un modelo puramente estatico.

Cada modelo declara metadata en INFO_PATRONES (tipo, si modela perdidas,
si esas perdidas son COMPARABLES contra una medicion con sonda coaxial,
rangos validos y un texto de advertencia), que la GUI muestra como cartel
y el informe PDF imprime como caja destacada.

IMPORTANTE -- cuando el error relativo de er'' no significa nada: hay dos
casos distintos en los que no tiene sentido comparar el er'' medido
contra el modelo.
  (a) El modelo devuelve er'' = 0 exactamente (silicona). Dividir por
      cero no esta definido.
  (b) El modelo devuelve un er'' fisicamente correcto pero ORDENES DE
      MAGNITUD por debajo de lo que puede resolver la sonda
      (ciclohexano: er'' ~ 1e-5 a 1e-4 en la banda del proyecto). El
      cociente existe, pero compara ruido contra un numero minusculo y
      da porcentajes gigantescos que no dicen nada.
En los dos casos INFO_PATRONES marca `perdidas_comparables=False`, y
tanto la consola como el informe PDF muestran "n/a" en esa columna en vez
de un numero enganoso.
"""
import warnings

import numpy as np

# Permitividad del aire a 20°C, 1013.2 mb y 50% HR (Medley, NPL 1991).
# Se usa como patron de circuito abierto en el algoritmo de calibracion.
ER_AIRE = 1.0006


def get_er_agua(frecs, T=25.0):
    """
    Modelo de Liebe, Hufford y Manabe (1991) para la permitividad compleja
    del agua destilada en funcion de la frecuencia y la temperatura.
    Valido hasta ~1 THz.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C). La permitividad del agua varia
        fuertemente con la temperatura (~0.4%/°C en la parte real), por lo
        que conviene pasar la temperatura real de la medicion.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    Theta = 1 - 300 / (273.15 + T)
    E_s = 77.66 - Theta * 103.3
    E_inf = E_s * 0.066
    yd = (20.27 + 146.5 * Theta + 314 * (Theta ** 2)) * 1e9  # GHz -> Hz
    # OJO: el signo del termino 1j define la convencion de la parte
    # imaginaria. Se usa "+1j" (en vez de "-1j" como en versiones previas
    # de este archivo) para que quede CONSISTENTE con los modelos de
    # alcohol etilico/isopropilico de mas abajo y con la convencion
    # er = er' - j*er'' (er''>0 para perdidas) que usan todos los papers
    # de la catedra (Felicio 2016, Zechmeister 2019, ARGENCON 2024, etc).
    # Mezclar convenciones distintas entre patrones rompe los algoritmos
    # de conversion que combinan varios patrones a la vez (metodo completo).
    E_c = E_inf + (E_s - E_inf) / (1 + 1j * frecs / yd)
    return E_c


# ===========================================================================
# Ecuaciones de relajacion genericas (NPL Report MAT 23, Seccion 2).
# Todas toman la frecuencia en Hz, pero los parametros de relajacion
# (fr, fr1, fr2) EXACTAMENTE como estan tabulados en el reporte, es decir
# en GHz (la conversion a Hz se hace adentro). Todas devuelven
# er = er' - j*er'' (er''>0 para perdidas), ya verificado numericamente
# contra las tablas del reporte.
# ===========================================================================

def _debye_simple(f_hz, es, e_inf, fr_ghz):
    """Debye simple (eqn. 1). Usado por metanol y DMSO."""
    fr = fr_ghz * 1e9
    return e_inf + (es - e_inf) / (1 + 1j * f_hz / fr)


def _debye_doble(f_hz, es, eh, fr1_ghz, e_inf, fr2_ghz):
    """Doble-Debye (eqn. 2). Usado por 1-butanol, 1-propanol y
    2-propanol (isopropilico) a bajas/medias temperaturas."""
    fr1 = fr1_ghz * 1e9
    fr2 = fr2_ghz * 1e9
    return (e_inf
            + (eh - e_inf) / (1 + 1j * f_hz / fr2)
            + (es - eh) / (1 + 1j * f_hz / fr1))


def _debye_gamma(f_hz, es, eh, fr_ghz, gamma):
    """Debye-Gamma (eqn. 3): relajacion simple + "cola" lineal que
    representa una segunda relajacion fuera de rango medido. Usado por
    etanol (a toda T) y por 1-butanol/1-propanol/2-propanol a altas
    temperaturas."""
    fr = fr_ghz * 1e9
    f_ghz = f_hz / 1e9
    return eh + (es - eh) / (1 + 1j * f_hz / fr) - 1j * gamma * f_ghz


def _davidson_cole(f_hz, es, e_inf, fr_ghz, beta):
    """Davidson-Cole (eqn. 4). Usado por etilenglicol."""
    fr = fr_ghz * 1e9
    return e_inf + (es - e_inf) / (1 + 1j * f_hz / fr) ** beta


def _fila_mas_cercana(tabla, T, nombre="", rango_recomendado=None, verbose=False):
    """Busca en `tabla` (dict {T_tabulada_C: {parametros...}}) la fila cuya
    temperatura este mas cerca de `T`. Si `T` cae fuera del rango
    tabulado/recomendado, avisa con un warning (pero igual devuelve el
    valor mas cercano disponible, como pidio el usuario)."""
    temps = np.array(sorted(tabla.keys()), dtype=float)
    T_sel = temps[np.argmin(np.abs(temps - T))]

    lo, hi = rango_recomendado if rango_recomendado else (temps.min(), temps.max())
    if not (lo <= T <= hi):
        warnings.warn(
            f"{nombre}: T={T}\u00b0C esta fuera del rango recomendado "
            f"({lo}-{hi}\u00b0C, NPL MAT 23). Se usan los parametros de "
            f"{T_sel}\u00b0C (el mas cercano disponible)."
        )
    if verbose:
        print(f"[Patrones] {nombre}: T pedida={T}\u00b0C -> se usa la fila "
              f"tabulada de {T_sel}\u00b0C.")
    return tabla[T_sel], T_sel


# ===========================================================================
# Etanol (alcohol etilico) -- NPL MAT 23, pag. 16 y 44-52.
# Solo se pudo ajustar de forma confiable con Debye-Gamma (eqn. 3) en todo
# el rango medido (ver nota 27 del reporte): no hay switch de ecuacion.
# ===========================================================================
_ETANOL_DEBYE_GAMMA = {
    10: dict(es=26.79, eh=4.624, fr=0.596, gamma=0.075),
    15: dict(es=25.95, eh=4.590, fr=0.700, gamma=0.071),
    20: dict(es=25.16, eh=4.531, fr=0.829, gamma=0.059),
    25: dict(es=24.43, eh=4.505, fr=0.964, gamma=0.056),
    30: dict(es=23.65, eh=4.471, fr=1.124, gamma=0.054),
    35: dict(es=22.88, eh=4.439, fr=1.303, gamma=0.053),
    40: dict(es=22.16, eh=4.410, fr=1.511, gamma=0.050),
    45: dict(es=21.45, eh=4.394, fr=1.745, gamma=0.049),
    50: dict(es=20.78, eh=4.378, fr=2.010, gamma=0.044),
}


def get_er_pat_alc_etilico(frecs, T=20.0, verbose=False):
    """
    Modelo Debye-Gamma para alcohol etilico (etanol), NPL MAT 23 (Gregory &
    Clarke). Los parametros se toman de la tabla de la pag. 16/44-52,
    aproximando a la temperatura tabulada (10-50 °C, pasos de 5 °C) mas
    cercana a `T`.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 20°C, la misma que se usaba antes en
        este archivo).
    verbose : bool
        Si es True, imprime que temperatura tabulada se termino usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    p, _ = _fila_mas_cercana(_ETANOL_DEBYE_GAMMA, T, nombre="Etanol",
                              rango_recomendado=(10, 50), verbose=verbose)
    return _debye_gamma(frecs, p['es'], p['eh'], p['fr'], p['gamma'])


# ===========================================================================
# 2-Propanol (alcohol isopropilico) -- NPL MAT 23, pag. 18 y 71-79.
# Nota 30: no hay medicion de permitividad estatica en celda de admitancia
# (no hay liquido de referencia para "anclar" el ajuste a baja frecuencia),
# por eso el reporte lo marca como "es fitted" en ambas tablas.
# Doble-Debye recomendado 10-25 °C; Debye-Gamma recomendado 30-50 °C.
# ===========================================================================
_ISOPROP_DOBLE_DEBYE = {
    10: dict(es=21.73, eh=3.573, fr1=0.217, fr2=5.037, e_inf=3.045),
    15: dict(es=20.89, eh=3.566, fr1=0.277, fr2=5.545, e_inf=3.035),
    20: dict(es=20.11, eh=3.557, fr1=0.351, fr2=5.721, e_inf=3.057),
    25: dict(es=19.30, eh=3.551, fr1=0.443, fr2=5.999, e_inf=3.065),
}
_ISOPROP_DEBYE_GAMMA = {
    30: dict(es=18.37, eh=3.466, fr=0.565, gamma=0.052),
    35: dict(es=17.65, eh=3.462, fr=0.702, gamma=0.047),
    40: dict(es=16.93, eh=3.458, fr=0.870, gamma=0.042),
    45: dict(es=16.21, eh=3.454, fr=1.072, gamma=0.038),
    50: dict(es=15.50, eh=3.451, fr=1.315, gamma=0.035),
}
_ISOPROP_UMBRAL_C = 27.5  # punto medio entre 25 (limite doble-Debye) y 30 (limite Debye-Gamma)


def get_er_pat_alc_isopropilico(frecs, T=25.0, verbose=False):
    """
    Modelo para alcohol isopropilico (2-propanol) de alta pureza, NPL
    MAT 23 (Gregory & Clarke). Usa doble-Debye (eqn. 2) para T <= 27.5 °C
    (rango recomendado 10-25 °C) y Debye-Gamma (eqn. 3) para T > 27.5 °C
    (rango recomendado 30-50 °C), ver nota 30 del reporte. Dentro de cada
    tramo se aproxima a la temperatura tabulada mas cercana.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C, la misma que se usaba antes en
        este archivo).
    verbose : bool
        Si es True, imprime que temperatura tabulada y que ecuacion se
        terminaron usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    if T <= _ISOPROP_UMBRAL_C:
        p, _ = _fila_mas_cercana(_ISOPROP_DOBLE_DEBYE, T,
                                  nombre="Alcohol isopropilico (doble-Debye)",
                                  rango_recomendado=(10, 25), verbose=verbose)
        return _debye_doble(frecs, p['es'], p['eh'], p['fr1'], p['e_inf'], p['fr2'])
    else:
        p, _ = _fila_mas_cercana(_ISOPROP_DEBYE_GAMMA, T,
                                  nombre="Alcohol isopropilico (Debye-Gamma)",
                                  rango_recomendado=(30, 50), verbose=verbose)
        return _debye_gamma(frecs, p['es'], p['eh'], p['fr'], p['gamma'])


# ===========================================================================
# Metanol -- NPL MAT 23, pag. 17 y 53-61. Debye simple (eqn. 1) en todo el
# rango medido.
# ===========================================================================
_METANOL_DEBYE = {
    10: dict(es=35.74, e_inf=5.818, fr=2.262),
    15: dict(es=34.68, e_inf=5.698, fr=2.532),
    20: dict(es=33.64, e_inf=5.654, fr=2.822),
    25: dict(es=32.66, e_inf=5.563, fr=3.141),
    30: dict(es=31.69, e_inf=5.450, fr=3.490),
    35: dict(es=30.78, e_inf=5.388, fr=3.862),
    40: dict(es=29.85, e_inf=5.251, fr=4.283),
    45: dict(es=28.95, e_inf=5.107, fr=4.738),
    50: dict(es=28.19, e_inf=5.224, fr=5.175),
}


def get_er_pat_metanol(frecs, T=25.0, verbose=False):
    """
    Modelo Debye simple para metanol de alta pureza, NPL MAT 23 (Gregory &
    Clarke). Parametros de la tabla de la pag. 17/53-61, aproximando a la
    temperatura tabulada mas cercana (10-50 °C, pasos de 5 °C).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C).
    verbose : bool
        Si es True, imprime que temperatura tabulada se termino usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    p, _ = _fila_mas_cercana(_METANOL_DEBYE, T, nombre="Metanol",
                              rango_recomendado=(10, 50), verbose=verbose)
    return _debye_simple(frecs, p['es'], p['e_inf'], p['fr'])


# ===========================================================================
# Dimetil sulfoxido (DMSO) -- NPL MAT 23, pag. 16 y 28-34. Debye simple
# (eqn. 1). Funde a 18 °C, por eso el reporte no tiene datos por debajo de
# 20 °C (ver nota 4).
# ===========================================================================
_DMSO_DEBYE = {
    20: dict(es=47.13, e_inf=6.802, fr=7.555),
    25: dict(es=46.49, e_inf=6.501, fr=8.323),
    30: dict(es=45.86, e_inf=6.357, fr=9.077),
    35: dict(es=45.19, e_inf=5.984, fr=9.924),
    40: dict(es=44.53, e_inf=5.828, fr=10.733),
    45: dict(es=43.86, e_inf=5.637, fr=11.588),
    50: dict(es=43.19, e_inf=5.410, fr=12.477),
}


def get_er_pat_dmso(frecs, T=25.0, verbose=False):
    """
    Modelo Debye simple para dimetil sulfoxido (DMSO), NPL MAT 23 (Gregory
    & Clarke). Parametros de la tabla de la pag. 16/28-34, aproximando a
    la temperatura tabulada mas cercana. OJO: el DMSO funde a 18 °C, asi
    que el reporte solo cubre 20-50 °C; para T menor a 20 °C se usa igual
    la fila de 20 °C (con un warning), pero en la practica el DMSO estaria
    solido a esa temperatura.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C).
    verbose : bool
        Si es True, imprime que temperatura tabulada se termino usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    p, _ = _fila_mas_cercana(_DMSO_DEBYE, T, nombre="DMSO",
                              rango_recomendado=(20, 50), verbose=verbose)
    return _debye_simple(frecs, p['es'], p['e_inf'], p['fr'])


# ===========================================================================
# Etilenglicol (etanodiol) -- NPL MAT 23, pag. 16 y 35-43. Davidson-Cole
# (eqn. 4) en todo el rango medido (es el unico liquido de la lista que no
# se ajusta bien con ningun tipo de Debye, ver Seccion 2).
# ===========================================================================
_ETILENGLICOL_DAVIDSON_COLE = {
    10: dict(es=44.31, e_inf=4.543, fr=0.599, beta=0.833),
    15: dict(es=43.08, e_inf=4.568, fr=0.764, beta=0.841),
    20: dict(es=41.89, e_inf=4.745, fr=0.962, beta=0.856),
    25: dict(es=40.75, e_inf=4.700, fr=1.190, beta=0.859),
    30: dict(es=39.66, e_inf=4.712, fr=1.451, beta=0.864),
    35: dict(es=38.60, e_inf=4.739, fr=1.756, beta=0.871),
    40: dict(es=37.63, e_inf=4.840, fr=2.101, beta=0.879),
    45: dict(es=36.63, e_inf=4.782, fr=2.484, beta=0.880),
    50: dict(es=35.71, e_inf=4.856, fr=2.912, beta=0.885),
}


def get_er_pat_etilenglicol(frecs, T=25.0, verbose=False):
    """
    Modelo Davidson-Cole para etilenglicol (etanodiol), NPL MAT 23 (Gregory
    & Clarke). Parametros de la tabla de la pag. 16/35-43, aproximando a
    la temperatura tabulada mas cercana (10-50 °C, pasos de 5 °C).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C).
    verbose : bool
        Si es True, imprime que temperatura tabulada se termino usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    p, _ = _fila_mas_cercana(_ETILENGLICOL_DAVIDSON_COLE, T, nombre="Etilenglicol",
                              rango_recomendado=(10, 50), verbose=verbose)
    return _davidson_cole(frecs, p['es'], p['e_inf'], p['fr'], p['beta'])


# ===========================================================================
# 1-Butanol (butan-1-ol) -- NPL MAT 23, pag. 15 y 19-27. Doble-Debye (eqn.
# 2) recomendado 10-40 °C; Debye-Gamma (eqn. 3) recomendado 45-50 °C
# (ver nota 26: a temperaturas altas la 2da relajacion queda fuera del
# rango medido y el doble-Debye deja de ajustar bien).
# ===========================================================================
_BUTANOL_DOBLE_DEBYE = {
    10: dict(es=19.54, eh=3.528, fr1=0.167, fr2=4.878, e_inf=2.914),
    15: dict(es=18.86, eh=3.527, fr1=0.207, fr2=5.259, e_inf=2.914),
    20: dict(es=18.19, eh=3.505, fr1=0.257, fr2=5.865, e_inf=2.900),
    25: dict(es=17.53, eh=3.506, fr1=0.318, fr2=5.979, e_inf=2.921),
    30: dict(es=16.89, eh=3.501, fr1=0.393, fr2=6.433, e_inf=2.923),
    35: dict(es=16.26, eh=3.493, fr1=0.483, fr2=7.234, e_inf=2.902),
    40: dict(es=15.65, eh=3.484, fr1=0.592, fr2=8.139, e_inf=2.883),
}
_BUTANOL_DEBYE_GAMMA = {
    45: dict(es=15.03, eh=3.418, fr=0.728, gamma=0.051),
    50: dict(es=14.44, eh=3.416, fr=0.883, gamma=0.047),
}
_BUTANOL_UMBRAL_C = 42.5  # punto medio entre 40 (limite doble-Debye) y 45 (limite Debye-Gamma)


def get_er_pat_butanol(frecs, T=25.0, verbose=False):
    """
    Modelo para 1-butanol (butan-1-ol) de alta pureza, NPL MAT 23 (Gregory
    & Clarke). Usa doble-Debye (eqn. 2) para T <= 42.5 °C (rango
    recomendado 10-40 °C) y Debye-Gamma (eqn. 3) para T > 42.5 °C (rango
    recomendado 45-50 °C), ver nota 26 del reporte. Dentro de cada tramo se
    aproxima a la temperatura tabulada mas cercana.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C).
    verbose : bool
        Si es True, imprime que temperatura tabulada y que ecuacion se
        terminaron usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    if T <= _BUTANOL_UMBRAL_C:
        p, _ = _fila_mas_cercana(_BUTANOL_DOBLE_DEBYE, T,
                                  nombre="1-Butanol (doble-Debye)",
                                  rango_recomendado=(10, 40), verbose=verbose)
        return _debye_doble(frecs, p['es'], p['eh'], p['fr1'], p['e_inf'], p['fr2'])
    else:
        p, _ = _fila_mas_cercana(_BUTANOL_DEBYE_GAMMA, T,
                                  nombre="1-Butanol (Debye-Gamma)",
                                  rango_recomendado=(45, 50), verbose=verbose)
        return _debye_gamma(frecs, p['es'], p['eh'], p['fr'], p['gamma'])


# ===========================================================================
# 1-Propanol (propan-1-ol) -- NPL MAT 23, pag. 17 y 62-70. Doble-Debye
# (eqn. 2) recomendado 10-30 °C; Debye-Gamma (eqn. 3) recomendado 35-50 °C
# (mismo motivo que 1-butanol, nota 26).
# ===========================================================================
_PROPANOL_DOBLE_DEBYE = {
    10: dict(es=22.61, eh=3.862, fr1=0.268, fr2=6.871, e_inf=3.126),
    15: dict(es=21.88, eh=3.844, fr1=0.328, fr2=7.302, e_inf=3.121),
    20: dict(es=21.15, eh=3.821, fr1=0.398, fr2=7.629, e_inf=3.142),
    25: dict(es=20.42, eh=3.804, fr1=0.485, fr2=8.290, e_inf=3.129),
    30: dict(es=19.75, eh=3.791, fr1=0.586, fr2=8.860, e_inf=3.105),
}
_PROPANOL_DEBYE_GAMMA = {
    35: dict(es=19.07, eh=3.712, fr=0.709, gamma=0.055),
    40: dict(es=18.40, eh=3.697, fr=0.852, gamma=0.051),
    45: dict(es=17.76, eh=3.692, fr=1.019, gamma=0.049),
    50: dict(es=17.11, eh=3.684, fr=1.218, gamma=0.047),
}
_PROPANOL_UMBRAL_C = 32.5  # punto medio entre 30 (limite doble-Debye) y 35 (limite Debye-Gamma)


def get_er_pat_propanol(frecs, T=25.0, verbose=False):
    """
    Modelo para 1-propanol (propan-1-ol) de alta pureza, NPL MAT 23
    (Gregory & Clarke). Usa doble-Debye (eqn. 2) para T <= 32.5 °C (rango
    recomendado 10-30 °C) y Debye-Gamma (eqn. 3) para T > 32.5 °C (rango
    recomendado 35-50 °C), ver nota 26 del reporte. Dentro de cada tramo se
    aproxima a la temperatura tabulada mas cercana.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25°C).
    verbose : bool
        Si es True, imprime que temperatura tabulada y que ecuacion se
        terminaron usando.

    Retorna
    -------
    ndarray complejo con er'(f) - j*er''(f)
    """
    frecs = np.asarray(frecs, dtype=float)
    if T <= _PROPANOL_UMBRAL_C:
        p, _ = _fila_mas_cercana(_PROPANOL_DOBLE_DEBYE, T,
                                  nombre="1-Propanol (doble-Debye)",
                                  rango_recomendado=(10, 30), verbose=verbose)
        return _debye_doble(frecs, p['es'], p['eh'], p['fr1'], p['e_inf'], p['fr2'])
    else:
        p, _ = _fila_mas_cercana(_PROPANOL_DEBYE_GAMMA, T,
                                  nombre="1-Propanol (Debye-Gamma)",
                                  rango_recomendado=(35, 50), verbose=verbose)
        return _debye_gamma(frecs, p['es'], p['eh'], p['fr'], p['gamma'])


# ===========================================================================
# LIQUIDOS SIN MODELO DE RELAJACION -- permitividad "estatica" (constante)
# ===========================================================================
# Los valores de abajo son las mediciones en CELDA DE ADMITANCIA de la
# Seccion 6 del reporte NPL MAT 23 (pag. 14). Esa celda es un capacitor de
# placas paralelas con separacion variable por micrometro: se mide la
# capacidad con y sin liquido entre los electrodos y de la relacion sale la
# permitividad relativa. Al medir a varias separaciones y quedarse con la
# pendiente se cancelan las capacidades parasitas de borde, que en un
# capacitor de placas fijo serian un error sistematico grande.
#
# Se las llama permitividad "estatica" aunque no se midieron en DC: el
# reporte las tomo entre 200 kHz y 1 MHz (nota 17). Por debajo de eso
# aparece polarizacion de electrodos -- acumulacion de iones contra las
# placas -- que infla la permitividad aparente; y a esas frecuencias la
# parte real todavia no se aparta de forma apreciable de su valor nominal
# en DC.
#
# OJO con los rangos de temperatura, que NO son iguales entre si ni
# coinciden con los 10-50 °C de los liquidos con Debye:
#   - Acetona        : 5-50 °C (45 y 50 °C dudosos, ver _ACETONA_DUDOSAS).
#   - Ciclohexano    : 10-50 °C. No hay medicion a 5 °C porque el
#                      ciclohexano funde a 6.5 °C (a 5 °C estaria solido).
#   - Silicona 1 cSt : 5-35 °C. No hay mediciones a 40, 45 ni 50 °C.
# ===========================================================================

def _interpolar_en_T(tabla, T, nombre="", rango_recomendado=None, verbose=False):
    """Interpola LINEALMENTE en temperatura dentro de `tabla`
    (dict {T_tabulada_C: valor}) y devuelve (valor, T_pedida_acotada).

    A diferencia de `_fila_mas_cercana` (que usan los modelos de Debye),
    aca SI se interpola en vez de saltar al escalon de 5 °C mas cercano.
    El motivo es que en los modelos de relajacion la ECUACION misma puede
    cambiar segun el tramo de temperatura (notas 26 y 30 del reporte), asi
    que mezclar parametros de dos filas distintas no tiene sentido fisico;
    aca, en cambio, lo unico que hay es un numero que varia de forma suave
    y practicamente lineal con T, y el propio reporte avala interpolar
    (Seccion 1.3). En una temperatura tabulada exacta la interpolacion
    devuelve el valor tabulado tal cual, asi que no se pierde nada.

    Fuera del rango tabulado se avisa con un warning y se satura en el
    valor del extremo mas cercano (np.interp ya hace ese recorte solo),
    que es el mismo criterio de "usar el mas cercano disponible" que sigue
    el resto de este archivo.
    """
    temps = np.array(sorted(tabla.keys()), dtype=float)
    valores = np.array([tabla[t] for t in sorted(tabla.keys())], dtype=float)

    lo, hi = rango_recomendado if rango_recomendado else (temps.min(), temps.max())
    if not (lo <= T <= hi):
        warnings.warn(
            f"{nombre}: T={T}°C esta fuera del rango medido "
            f"({lo}-{hi}°C, NPL MAT 23, Seccion 6). Se satura en el "
            f"extremo mas cercano ({lo if T < lo else hi}°C)."
        )

    valor = float(np.interp(float(T), temps, valores))
    if verbose:
        print(f"[Patrones] {nombre}: T pedida={T}°C -> permitividad "
              f"estatica interpolada = {valor:.4f}.")
    return valor


def _permitividad_estatica(frecs, tabla, T, nombre, rango_recomendado,
                            tan_delta=0.0, verbose=False):
    """Devuelve un array complejo CONSTANTE en frecuencia, con la
    permitividad estatica interpolada a la temperatura `T`.

    `tan_delta` (tangente de perdidas) permite agregar una parte
    imaginaria si en algun momento se consigue un dato de perdidas para
    ese liquido: er = es * (1 - j*tan_delta), que respeta la convencion
    er = er' - j*er'' del resto del archivo. Por defecto es 0, o sea
    er'' = 0 exactamente. Ver la advertencia de cada modelo en
    INFO_PATRONES para saber si ese 0 significa "sin perdidas reales"
    (ciclohexano, silicona) o "perdidas no modeladas" (acetona).
    """
    frecs = np.asarray(frecs, dtype=float)
    es = _interpolar_en_T(tabla, T, nombre=nombre,
                           rango_recomendado=rango_recomendado, verbose=verbose)
    valor = complex(es) * (1.0 - 1j * float(tan_delta))
    return np.full(frecs.shape, valor, dtype=complex)


# --- Acetona -- NPL MAT 23, Seccion 6 (pag. 14). SOLO permitividad
# estatica: la relajacion de la acetona esta MUY por encima de los 5 GHz
# que llegaba a medir el equipo, asi que el reporte no publica ningun
# ajuste de Debye para ella (nota 23).
_ACETONA_ESTATICA = {
    5: 22.80, 10: 22.21, 15: 21.65, 20: 21.13, 25: 20.59,
    30: 20.10, 35: 19.60, 40: 19.12, 45: 18.68, 50: 18.19,
}
# El reporte muestra en italica las mediciones a 45 y 50 °C porque pudo
# haber burbujeo dentro de la celda (la acetona hierve a 56 °C).
_ACETONA_DUDOSAS = (45.0, 50.0)

# --- Ciclohexano -- NPL MAT 23, Seccion 6. Molecula NO POLAR: no tiene
# relajacion dipolar, por eso no hay -- ni puede haber -- un modelo de
# Debye. NPL lo midio justamente como control del propio banco, y los
# valores coinciden con Kienitz & Marsh (referencia [8] del reporte), que
# es el dato clasico de ciclohexano como liquido dielectrico de
# referencia. No hay valor a 5 °C: funde a 6.5 °C.
_CICLOHEXANO_ESTATICA = {
    10: 2.040, 15: 2.033, 20: 2.024, 25: 2.015,
    30: 2.008, 35: 2.000, 40: 1.992, 45: 1.984, 50: 1.975,
}

# --- Fluido de silicona Dow-Corning 200, 1 cSt (polidimetilsiloxano) --
# NPL MAT 23, Seccion 6. Igual que el ciclohexano: no polar, sin
# relajacion, muy bajas perdidas. Solo hay mediciones de 5 a 35 °C.
_SILICONA_ESTATICA = {
    5: 2.351, 10: 2.336, 15: 2.322, 20: 2.307, 25: 2.294,
    30: 2.279, 35: 2.265,
}


# ---------------------------------------------------------------------------
# Parametros de relajacion de la ACETONA, ajustados sobre las curvas medidas
# de la figura 5 de:
#     A Yu Zarubina, S G Kibets, A A Politiko, V N Semenenko, K M Baskov y
#     V A Chistyaev, "Complex permittivity of organic solvents at microwave
#     frequencies", IOP Conf. Ser.: Mater. Sci. Eng. 862 (2020) 062085,
#     doi:10.1088/1757-899X/862/6/062085.
# Midieron con sonda coaxial (DAK + VNA Rohde & Schwarz ZVA24) entre 0.2 y
# 20 GHz a 23 °C. El ajuste (Debye simple, 3 parametros libres, sobre las
# dos curvas de acetona digitalizadas de la figura 5) da:
#
#       es = 19.94     e_inf = 5.14     fr = 46.6 GHz   (tau = 3.42 ps)
#       rms: 0.51 en er', 0.34 en er''
#
# La frecuencia de relajacion es el parametro ROBUSTO del ajuste: un
# bootstrap sobre submuestras da fr = 46.6 +- 1.0 GHz, y todas las
# variantes probadas (es libre, es anclado, ajuste solo a los 3 puntos
# exactos de la Table 2 del paper) caen en 45-47 GHz. es y e_inf, en
# cambio, quedan fuertemente correlacionados entre si: con datos que solo
# llegan a 20 GHz nunca se pasa de f/fr = 0.43, asi que la curva todavia
# no "dobla" lo suficiente como para separarlos.
# ---------------------------------------------------------------------------
_ACETONA_FIT_ZARUBINA = dict(es=19.94, e_inf=5.14, fr=46.6, T_medicion=23.0)

# Lo que se usa en el modelo de abajo NO es exactamente ese ajuste: se
# conservan fr y e_inf, pero es se toma de la tabla de celda de admitancia
# del NPL MAT 23 (_ACETONA_ESTATICA), por tres motivos:
#
#  1. El es de NPL es TRAZABLE (+-0.04) y el del ajuste sale de una sonda
#     coaxial sin trazabilidad declarada.
#  2. El paper mide a UNA sola temperatura (23 °C); la tabla de NPL da la
#     dependencia con T entre 5 y 50 °C, que es la que necesita este
#     proyecto.
#  3. Los dos numeros no coinciden: 19.94 (ajuste) contra 20.81 (NPL a
#     23 °C), un -4.1%. La discrepancia NO es ruido, es una firma de la
#     sonda del paper: entre 2 y 10 GHz su er' se aplana en ~19.1-19.5,
#     y una relajacion en 46 GHz no puede hacer caer er' mas de 0.03 en
#     ese tramo (a 5 GHz, f/fr = 0.1). Es decir, esa caida de ~1.5
#     unidades no es fisica; es la sonda perdiendo exactitud lejos del
#     centro de su banda. El mismo efecto se ve en er'' a baja
#     frecuencia: el paper da 1.06 a 1 GHz y 0.38 a 0.2 GHz, una relacion
#     de 2.8 cuando un Debye exige 5.
#
# Consecuencia practica de la mezcla: como Delta = es - e_inf pasa de
# 14.80 (ajuste puro) a 15.67 (es de NPL a 23 °C), el er'' que predice
# este modelo queda ~6% por encima del ajuste puro. Esta dentro de la
# incertidumbre de todo el ejercicio, y el paper mismo no es confiable en
# er'' por debajo de unos 8 GHz.


def get_er_pat_acetona(frecs, T=25.0, verbose=False):
    """
    Modelo de DEBYE simple para la acetona.

        er(f) = e_inf + (es(T) - e_inf) / (1 + j*f/fr)

    Origen de cada parametro (ver el bloque de comentarios de arriba):

      - es(T) : tabla de celda de admitancia del NPL MAT 23, Seccion 6,
                interpolada linealmente en T (trazable, +-0.04, 5-50 °C).
      - e_inf : 5.14, del ajuste a la figura 5 de Zarubina et al. (2020).
      - fr    : 46.6 GHz (tau = 3.42 ps), del mismo ajuste. Es el
                parametro robusto: fr = 46.6 +- 1.0 GHz (bootstrap).

    Este modelo REEMPLAZA a la version anterior de permitividad estatica
    constante. La diferencia importante para este proyecto es que ahora
    er'' NO es cero: en la banda tipica de la sonda vale del orden de 0.2
    a 2, que es perfectamente medible.

    LIMITACIONES, en orden de importancia:

      - fr NO depende de la temperatura en este modelo (el paper mide a
        una sola T, 23 °C). La frecuencia de relajacion de un liquido
        real sube al calentarse, asi que lejos de 23 °C el er'' va a
        tener un error sistematico. La parte real es mucho menos
        sensible, porque en la banda de este proyecto er' ~ es(T), y esa
        si sigue la temperatura.
      - Valido hasta ~20 GHz, que es donde llegan los datos. Por encima,
        e_inf = 5.14 no es el limite optico real de la acetona (n^2 ~
        1.85): es un parametro EFECTIVO que absorbe los procesos rapidos
        que quedan fuera del rango medido. No usar este modelo para
        estimar er' arriba de 20 GHz.
      - e_inf y es estan correlacionados en el ajuste; ver el comentario
        de arriba.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25). Rango de es: 5-50 °C.
    verbose : bool
        Si es True, imprime los parametros que se terminaron usando.

    Retorna
    -------
    ndarray complejo con er' - j*er''.
    """
    frecs = np.asarray(frecs, dtype=float)
    if _ACETONA_DUDOSAS[0] <= float(T) <= _ACETONA_DUDOSAS[1]:
        warnings.warn(
            f"Acetona: las mediciones de permitividad estatica de NPL a "
            f"{_ACETONA_DUDOSAS[0]:.0f} y {_ACETONA_DUDOSAS[1]:.0f} \u00b0C figuran "
            f"en italica en el reporte porque pudo haber burbujeo dentro de "
            f"la celda (la acetona hierve a 56 \u00b0C). El es a T={T}\u00b0C sale "
            f"de esa zona y puede ser menos confiable."
        )
    es = _interpolar_en_T(_ACETONA_ESTATICA, T,
                           nombre="Acetona (es, NPL MAT 23)",
                           rango_recomendado=(5, 50), verbose=verbose)
    p = _ACETONA_FIT_ZARUBINA
    if verbose:
        print(f"[Patrones] Acetona: es={es:.3f} (NPL a {T}\u00b0C), "
              f"e_inf={p['e_inf']}, fr={p['fr']} GHz "
              f"(ajuste Zarubina 2020 a {p['T_medicion']}\u00b0C).")
    return _debye_simple(frecs, es, p['e_inf'], p['fr'])


# ---------------------------------------------------------------------------
# CICLOHEXANO. Modelo armado con tres fuentes que coinciden entre si:
#
#  [1] NPL MAT 23, Seccion 6 (tabla _CICLOHEXANO_ESTATICA de mas arriba):
#      permitividad estatica trazable de 10 a 50 °C, en pasos de 5 °C.
#  [2] U Kaatze, "Reference liquids for the calibration of dielectric
#      sensors and measurement instruments", Meas. Sci. Technol. 18 (2007)
#      967-976, doi:10.1088/0957-0233/18/4/002.
#  [3] M N Afsar et al., "A Comparison of Dielectric Measurement Methods
#      for Liquids in the Frequency Range 1 GHz to 4 THz", IEEE Trans.
#      Instrum. Meas. IM-29 (1980) 283-288.
#
# PARTE REAL. Un ajuste lineal por minimos cuadrados sobre los 9 puntos de
# NPL da:
#
#       es(T) = 2.0565 - 1.620e-3 * T[°C]
#             = 2.040 - 1.620e-3 * (T - 10 °C)
#
# con residuo maximo 0.001 en todo 10-50 °C. Eso reproduce exactamente la
# ecuacion (9) de Kaatze [2], que da -1.6e-3/°C con error < 0.001 (no es
# casualidad: Kaatze ajusto sobre los mismos datos de NPL).
#
# Que ese valor estatico se pueda usar TAMBIEN en microondas -- que es lo
# que hace falta aca -- lo respaldan las otras dos fuentes:
#   - Kaatze, ecuacion (10): er'(v,T) = es(T) y er''(v,T) = 0 valen "en el
#     rango de frecuencias de interes" con error relativo < 0.0005, hasta
#     la region submilimetrica.
#   - Afsar [3], figura 1: promediando TODAS las mediciones entre 1 y
#     300 GHz de siete laboratorios distintos obtiene er' = 2.0126 +-
#     0.0056 a 25 °C, y el texto dice explicitamente que no hay evidencia
#     de dispersion por debajo de 300 GHz.
#
# Se digitalizaron los puntos de esa figura 1 y se ajusto er' = a + b*f
# hasta 100 GHz, sumando el valor estatico de NPL como punto en f = 0:
#
#       b = -9.0e-6 +- 7.0e-5 por GHz   ->  t = -0.13
#
# es decir, la pendiente NO es significativa (|b|*100 GHz = 0.0009, ocho
# veces menor que la dispersion experimental de 0.0076). El "modelo lineal
# en frecuencia" se reduce entonces, dentro del error de medicion, a una
# CONSTANTE. Por eso el modelo de abajo deja er' plano en frecuencia y
# lineal en temperatura.
#
# PARTE IMAGINARIA. Sale de la curva E (ciclohexano puro) de la figura 4
# de Afsar [3], que es log-log de er'' contra frecuencia a 25 °C. La curva
# digitalizada es una recta casi perfecta en log-log:
#
#       log10(er'') = -12.081 + 0.7905 * log10(f[Hz])
#
# con residuo rms de 0.012 decadas (~3%). En la forma que se usa aca:
#
#       er''(f) = 1.08e-5 * (f / 1 GHz)^0.791
#
# OJO con la escala: esto da 6.7e-5 a 10 GHz, 4.1e-4 a 100 GHz y recien
# llega a 1e-3 cerca de 300 GHz (el final de la curva medida). Es decir,
# er'' se mantiene por debajo de 1e-3 en TODO el rango util, y la tangente
# de perdidas correspondiente (er''/er' ~ 2e-4 a 100 GHz) esta varios
# ordenes de magnitud por debajo de lo que puede resolver una sonda
# coaxial open-ended. Para el proyecto, a los efectos practicos el
# ciclohexano sigue siendo un material sin perdidas; el modelo se agrega
# por completitud fisica y para no tener que dividir por cero.
# ---------------------------------------------------------------------------
_CICLOHEXANO_ER1_A = 2.05649      # ordenada al origen del ajuste lineal en T
_CICLOHEXANO_ER1_B = -1.6200e-3   # pendiente [1/°C]
_CICLOHEXANO_ER2_A = 1.080e-5     # er'' a 1 GHz (Afsar fig. 4, curva E, 25 °C)
_CICLOHEXANO_ER2_N = 0.7905       # exponente de la ley de potencias


def get_er_pat_ciclohexano(frecs, T=25.0, verbose=False):
    """
    Modelo del CICLOHEXANO, combinando NPL MAT 23 + Kaatze (2007) + Afsar
    et al. (1980). Ver el bloque de comentarios de arriba para el detalle
    de como se obtuvo cada pieza.

        er'(T)  = 2.0565 - 1.620e-3 * T[°C]     (constante en frecuencia)
        er''(f) = 1.08e-5 * (f / 1 GHz)^0.791   (a 25 °C)

    El ciclohexano es NO POLAR: no tiene relajacion dipolar, asi que er'
    es realmente constante con la frecuencia (verificado hasta 300 GHz por
    Afsar, y hasta la region submilimetrica segun Kaatze). No es una
    aproximacion de conveniencia como lo era el modelo estatico de la
    acetona: aca la constante ES el comportamiento fisico.

    Las perdidas son minusculas (er'' < 1e-3 hasta 300 GHz) y provienen de
    absorcion inducida por colisiones, no de relajacion dipolar. En la
    practica estan MUY por debajo de lo que puede medir una sonda coaxial
    open-ended, asi que no tiene sentido comparar el er'' medido contra
    esta referencia (ver INFO_PATRONES['ciclohexano']['perdidas_
    comparables'], que por eso vale False).

    LIMITACIONES:
      - er'(T) esta validado entre 10 y 50 °C (el ciclohexano funde a
        6.5 °C, por eso no hay dato a 5 °C).
      - er''(f) sale de mediciones a 25 °C unicamente: no lleva
        dependencia con la temperatura.
      - er''(f) esta medido entre ~10 y 265 GHz; por debajo de eso la ley
        de potencias es una extrapolacion (que tiende a 0 en f = 0, que
        es el comportamiento fisicamente esperado).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    T : float
        Temperatura en °C (default 25). Rango validado: 10-50 °C.
    verbose : bool
        Si es True, imprime los valores que se terminaron usando.

    Retorna
    -------
    ndarray complejo con er' - j*er''.
    """
    frecs = np.asarray(frecs, dtype=float)
    if not (10.0 <= float(T) <= 50.0):
        warnings.warn(
            f"Ciclohexano: T={T}\u00b0C esta fuera del rango validado "
            f"(10-50 \u00b0C; funde a 6.5 \u00b0C). Se extrapola la recta "
            f"er'(T), que fuera de ese rango no esta respaldada por datos."
        )
    er1 = _CICLOHEXANO_ER1_A + _CICLOHEXANO_ER1_B * float(T)
    f_ghz = np.maximum(frecs, 0.0) / 1e9
    er2 = _CICLOHEXANO_ER2_A * (f_ghz ** _CICLOHEXANO_ER2_N)
    if verbose:
        print(f"[Patrones] Ciclohexano: er'={er1:.4f} a T={T}\u00b0C "
              f"(constante en f); er'' de {er2.min():.2e} a {er2.max():.2e} "
              f"(Afsar 1980, fig. 4 curva E).")
    return er1 - 1j * er2


def get_er_pat_silicona(frecs, T=25.0, tan_delta=0.0, verbose=False):
    """
    Permitividad del FLUIDO DE SILICONA Dow-Corning 200 de 1 cSt
    (polidimetilsiloxano), NPL MAT 23 (Gregory & Clarke), Seccion 6.

    Mismo caso que el ciclohexano: fluido no polar, sin relajacion
    dipolar en RF/microondas, permitividad practicamente constante (~2.3)
    y perdidas despreciables. No hay ni puede haber modelo de Debye.

    OJO con el rango de temperatura: el reporte solo publica mediciones
    de 5 a 35 °C (no hay 40, 45 ni 50 °C), bastante mas corto que el de
    los demas liquidos. Este grado de silicona es muy volatil, lo que
    explica que no se haya medido mas arriba.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz (solo dan la forma del array de salida).
    T : float
        Temperatura en °C (default 25 °C). Rango medido: 5-35 °C.
    tan_delta : float
        Tangente de perdidas (default 0, fisicamente razonable aca).
    verbose : bool
        Si es True, imprime la permitividad que se termino usando.

    Retorna
    -------
    ndarray complejo con er' - j*er'' (constante en frecuencia).
    """
    return _permitividad_estatica(frecs, _SILICONA_ESTATICA, T,
                                   nombre="Fluido de silicona 1 cSt (permitividad estatica)",
                                   rango_recomendado=(5, 35),
                                   tan_delta=tan_delta, verbose=verbose)


# Diccionario de conveniencia para elegir un patron teorico por nombre,
# util en scripts que iteran sobre varios materiales. Todas las funciones
# (salvo el agua) aceptan (frecs, T=..., verbose=...).
PATRONES_TEORICOS = {
    'agua': get_er_agua,
    'alcohol_etilico': get_er_pat_alc_etilico,
    'alcohol_isopropilico': get_er_pat_alc_isopropilico,
    'metanol': get_er_pat_metanol,
    'dmso': get_er_pat_dmso,
    'etilenglicol': get_er_pat_etilenglicol,
    'butanol': get_er_pat_butanol,
    'propanol': get_er_pat_propanol,
    # Liquidos SIN modelo de relajacion (permitividad estatica constante).
    # Ver INFO_PATRONES mas abajo: cada uno lleva su texto de advertencia.
    'acetona': get_er_pat_acetona,
    'ciclohexano': get_er_pat_ciclohexano,
    'silicona': get_er_pat_silicona,
}

# Etiquetas "lindas" (nombre para mostrar) de cada patron teorico. Vive
# aca -- junto a PATRONES_TEORICOS -- para ser la UNICA fuente de verdad:
# tanto gui_funciones.py (combobox de la GUI) como analisis_permitividad.py
# / reporte_pdf.py (informe PDF) importan este diccionario, asi se evita
# que la GUI muestre un nombre para un modelo y el informe PDF muestre
# otro distinto para el mismo modelo.
ETIQUETAS_PATRONES = {
    'agua': "Agua (Liebe-Hufford-Manabe)",
    'alcohol_etilico': "Alcohol etilico (etanol)",
    'alcohol_isopropilico': "Alcohol isopropilico (2-propanol)",
    'metanol': "Metanol",
    'dmso': "Dimetil sulfoxido (DMSO)",
    'etilenglicol': "Etilenglicol (etanodiol)",
    'butanol': "1-Butanol",
    'propanol': "1-Propanol",
    'acetona': "Acetona (Debye, Zarubina 2020)",
    'ciclohexano': "Ciclohexano (Kaatze 2007 + Afsar 1980)",
    'silicona': "Fluido de silicona 1 cSt (solo ε estatica)",
}


# ===========================================================================
# Metadata de cada modelo: que tipo es, que modela y que NO, y el texto de
# advertencia que la GUI muestra como cartel y el informe PDF como caja
# destacada.
# ===========================================================================
# Claves de cada entrada:
#   'tipo'            : 'relajacion' (tiene un modelo de Debye/Davidson-Cole
#                       ajustado y por lo tanto varia con la frecuencia) o
#                       'estatico' (permitividad constante en frecuencia).
#   'modela_perdidas' : True si el er'' que devuelve el modelo es un valor
#                       fisico calculado; False si es 0 por no estar
#                       modelado. Lo usan analisis_permitividad.py y
#                       reporte_pdf.py para no mostrar un "error relativo
#                       de er''" que no significaria nada.
#   'perdidas_reales_despreciables' : True solo para los no polares, donde
#                       er''=0, ademas de no estar modelado, es
#                       fisicamente correcto. Sirve para distinguir el
#                       caso "constante de verdad" del caso "constante por
#                       conveniencia" (acetona).
#   'rango_temperatura_c' : (min, max) realmente medido por NPL.
#   'rango_frecuencia_ghz': (min, max) en el que el modelo es confiable, o
#                       None si no aplica / no hay limite practico.
#   'nivel'           : 'info' | 'aviso' | 'critico'. Define el color del
#                       cartel en la GUI y de la caja en el PDF.
#   'advertencia'     : texto para mostrarle al usuario. None = sin cartel.
# ===========================================================================
_SIN_ADVERTENCIA = dict(
    tipo='relajacion', modela_perdidas=True, perdidas_comparables=True,
    perdidas_reales_despreciables=False,
    rango_temperatura_c=(10, 50), rango_frecuencia_ghz=(0.03, 5),
    etiqueta_curva="Teorico (Debye)",
    nivel='info', advertencia=None,
)

INFO_PATRONES = {
    'agua': dict(_SIN_ADVERTENCIA,
                 rango_temperatura_c=(0, 100), rango_frecuencia_ghz=(0, 1000),
                 etiqueta_curva="Teorico (Liebe-Hufford-Manabe)"),
    'alcohol_etilico': dict(_SIN_ADVERTENCIA),
    'alcohol_isopropilico': dict(_SIN_ADVERTENCIA),
    'metanol': dict(_SIN_ADVERTENCIA, rango_frecuencia_ghz=(0.03, 10)),
    'dmso': dict(_SIN_ADVERTENCIA,
                 rango_temperatura_c=(20, 50), rango_frecuencia_ghz=(0.03, 10)),
    'etilenglicol': dict(_SIN_ADVERTENCIA, rango_frecuencia_ghz=(0.03, 10)),
    'butanol': dict(_SIN_ADVERTENCIA),
    'propanol': dict(_SIN_ADVERTENCIA),

    'acetona': dict(
        tipo='relajacion',
        modela_perdidas=True,
        perdidas_comparables=True,
        perdidas_reales_despreciables=False,
        rango_temperatura_c=(5, 50),
        rango_frecuencia_ghz=(0.2, 20),
        etiqueta_curva="Teorico (Debye, ajuste Zarubina 2020)",
        nivel='aviso',
        advertencia=(
            "Modelo de Debye ARMADO CON DOS FUENTES, no publicado como tal "
            "en ningun lado: es(T) sale de la tabla de permitividad estatica "
            "del NPL MAT 23 (trazable, \u00b10.04, 5-50 \u00b0C), mientras que "
            "e_inf = 5.14 y fr = 46.6 GHz (tau = 3.4 ps) salen de un ajuste "
            "propio sobre las curvas de acetona de la figura 5 de Zarubina "
            "et al. (2020), medidas con sonda coaxial entre 0.2 y 20 GHz a "
            "23 \u00b0C.\n\n"
            "\u2022 fr es el parametro solido del ajuste (46.6 \u00b1 1.0 GHz por "
            "bootstrap, y todas las variantes probadas caen en 45-47 GHz). "
            "e_inf y es, en cambio, quedan correlacionados entre si porque "
            "los datos no pasan de f/fr = 0.43.\n"
            "\u2022 fr NO varia con la temperatura en este modelo: el paper mide "
            "a una sola T. Lejos de 23 \u00b0C, er'' va a tener un error "
            "sistematico (er' aguanta mejor, porque en esta banda er' ~ es(T) "
            "y eso si sigue la temperatura).\n"
            "\u2022 Valido hasta ~20 GHz. Por encima, e_inf = 5.14 no es el "
            "limite optico real de la acetona (n^2 ~ 1.85): es un valor "
            "EFECTIVO que absorbe los procesos rapidos fuera de rango.\n"
            "\u2022 El er' medido por Zarubina entre 2 y 10 GHz queda ~4% por "
            "debajo del valor trazable de NPL, y esa caida no puede ser "
            "fisica para una relajacion en 46 GHz: es la sonda del paper "
            "perdiendo exactitud lejos del centro de su banda. Por eso el "
            "modelo toma es de NPL y no del ajuste."
        ),
    ),

    'ciclohexano': dict(
        tipo='no_polar',
        modela_perdidas=True,
        perdidas_comparables=False,
        perdidas_reales_despreciables=True,
        rango_temperatura_c=(10, 50),
        rango_frecuencia_ghz=(0, 300),
        etiqueta_curva="Teorico (Kaatze 2007 + Afsar 1980)",
        nivel='info',
        advertencia=(
            "El ciclohexano es NO POLAR: no tiene relajacion dipolar, asi "
            "que no existe -- ni puede existir -- un modelo de Debye. Aca la "
            "permitividad constante NO es una aproximacion de conveniencia, "
            "es el comportamiento fisico real.\n\n"
            "\u2022 er'(T) = 2.0565 - 1.620e-3*T[\u00b0C], ajustado sobre los 9 "
            "puntos del NPL MAT 23 con residuo < 0.001. Reproduce la "
            "ecuacion (9) de Kaatze (2007). Validado 10-50 \u00b0C (funde a "
            "6.5 \u00b0C).\n"
            "\u2022 Constante en frecuencia: Afsar et al. (1980) promedian "
            "er' = 2.0126 \u00b1 0.0056 entre 1 y 300 GHz sin evidencia de "
            "dispersion, y Kaatze lo extiende hasta la region "
            "submilimetrica con error < 0.05%.\n"
            "\u2022 er''(f) = 1.08e-5*(f/GHz)^0.791, de la curva E de la figura "
            "4 de Afsar (25 \u00b0C). Da 4.1e-4 a 100 GHz y recien llega a "
            "1e-3 cerca de 300 GHz: esta MUY por debajo de lo que puede "
            "resolver una sonda coaxial, asi que el error relativo de er'' "
            "contra esta referencia se informa como n/a.\n"
            "\u2022 Como patron de calibracion: Kaatze (2007) lo recomienda "
            "explicitamente como uno de los cuatro liquidos de referencia, "
            "justamente porque hace falta un patron de BAJA permitividad "
            "para medir bien muestras de baja permitividad (la capacidad "
            "efectiva de la sonda depende de la permitividad de la muestra). "
            "Como 3er patron de una calibracion de solo 3, en cambio, queda "
            "mal condicionado frente al aire."
        ),
    ),

    'silicona': dict(
        tipo='estatico',
        modela_perdidas=False,
        perdidas_comparables=False,
        perdidas_reales_despreciables=True,
        rango_temperatura_c=(5, 35),
        rango_frecuencia_ghz=None,
        etiqueta_curva="Teorico (\u03b5 estatica NPL, sin relajacion)",
        nivel='aviso',
        advertencia=(
            "Unico de los tres liquidos sin tabla de relajacion que sigue "
            "siendo un modelo puramente ESTATICO. El fluido de silicona "
            "Dow-Corning 200 de 1 cSt (polidimetilsiloxano) es no polar: "
            "igual que el ciclohexano, no tiene relajacion dipolar en RF ni "
            "en microondas, asi que su permitividad (~2.3) es realmente "
            "constante. La diferencia es que para el ciclohexano hay datos "
            "publicados de er'' (Afsar 1980) y para la silicona no, asi que "
            "aca er'' se devuelve como 0.\n\n"
            "\u2022 OJO con la temperatura: NPL solo publica mediciones de 5 a "
            "35 \u00b0C (no hay 40, 45 ni 50 \u00b0C), bastante menos rango que el "
            "resto de los liquidos. Fuera de ahi el valor se satura en el "
            "extremo mas cercano.\n"
            "\u2022 Como el er'' de referencia es 0, el error relativo de er'' "
            "se informa como n/a.\n"
            "\u2022 Las perdidas reales son despreciables igual, asi que el 0 "
            "es una buena aproximacion en la practica."
        ),
    ),
}


def info_patron(clave):
    """Metadata de un modelo (ver INFO_PATRONES). Devuelve un dict con
    valores por defecto conservadores si la clave no esta registrada, asi
    quien lo consuma nunca tiene que chequear por None."""
    if not clave:
        return dict(_SIN_ADVERTENCIA)
    return INFO_PATRONES.get(clave, dict(_SIN_ADVERTENCIA))


def advertencia_patron(clave):
    """Texto de advertencia de un modelo, o None si no tiene. Es lo que la
    GUI muestra como cartel y el informe PDF como caja destacada."""
    return info_patron(clave).get('advertencia')


def nivel_advertencia_patron(clave):
    """'info' | 'aviso' | 'critico' -- define el color del cartel."""
    return info_patron(clave).get('nivel', 'info')


def es_modelo_estatico(clave):
    """True si el modelo devuelve una parte REAL constante en frecuencia
    (o sea, no tiene relajacion ajustada). Cubre tanto los liquidos no
    polares (ciclohexano: constante por fisica) como los que solo tienen
    permitividad estatica tabulada (silicona)."""
    return info_patron(clave).get('tipo') in ('estatico', 'no_polar')


def modela_perdidas(clave):
    """True si el modelo calcula un er'' fisico; False si devuelve 0 por
    no tener datos de perdidas."""
    return bool(info_patron(clave).get('modela_perdidas', True))


def perdidas_comparables(clave):
    """True si tiene sentido calcular el error relativo del er'' MEDIDO
    contra el de este modelo.

    Es mas restrictivo que `modela_perdidas`, y la diferencia importa:
    el ciclohexano SI modela perdidas (er'' ~ 1e-5 a 1e-4 en la banda del
    proyecto, de la curva E de Afsar), pero esos valores estan ordenes de
    magnitud por debajo de lo que puede resolver una sonda coaxial
    open-ended. Dividir el ruido de la medicion por un numero tan chico
    da porcentajes enormes que no dicen absolutamente nada, asi que en
    ese caso el error de er'' se informa como n/a igual que cuando la
    referencia es 0."""
    return bool(info_patron(clave).get('perdidas_comparables', True))


def etiqueta_curva_teorica(clave):
    """Texto de leyenda para la curva teorica de este modelo en los
    graficos. No todos son un Debye, asi que rotularlos a todos como
    "Teorico (Debye)" seria enganoso."""
    if not clave:
        return "Teorico"
    return info_patron(clave).get('etiqueta_curva', "Teorico (Debye)")


def resumen_corto_patron(clave):
    """Una linea para mostrar al lado del nombre del modelo en tablas y
    listados (o None si es un modelo de relajacion estandar del MAT 23,
    donde no hace falta aclarar nada)."""
    info = info_patron(clave)
    tipo = info.get('tipo')
    t_min, t_max = info.get('rango_temperatura_c', (None, None))
    if tipo == 'no_polar':
        base = "no polar: \u03b5' constante en f, perdidas despreciables"
    elif tipo == 'estatico':
        base = "\u03b5 estatica constante, er'' NO modelado"
    elif clave == 'acetona':
        base = "Debye ajustado sobre datos de 2020 (fr fijo, 23 \u00b0C)"
    else:
        return None
    if t_min is not None:
        return f"{base} \u2014 validado {t_min:.0f}-{t_max:.0f} \u00b0C"
    return base


def etiqueta_patron(clave):
    """Clave de PATRONES_TEORICOS -> etiqueta linda para mostrar.

    Si `clave` es None/"" (sin modelo asignado), devuelve None. Si es una
    clave valida sin etiqueta especifica definida arriba (p.ej. un modelo
    nuevo que se agrego a PATRONES_TEORICOS pero todavia no tiene entrada
    en ETIQUETAS_PATRONES), devuelve la clave tal cual, para que nunca
    falte texto para mostrar."""
    if not clave:
        return None
    return ETIQUETAS_PATRONES.get(clave, clave)