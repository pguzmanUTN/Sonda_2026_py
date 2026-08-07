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
}


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
