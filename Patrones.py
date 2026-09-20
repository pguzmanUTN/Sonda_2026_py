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

Liquidos SIN modelo de relajacion (permitividad "estatica")
------------------------------------------------------------
No todos los liquidos del reporte NPL MAT 23 tienen un modelo de Debye.
La Seccion 6 del reporte (tabla de mediciones en celda de admitancia)
publica la permitividad ESTATICA -- es decir, de baja frecuencia -- de
varios liquidos para los que NO hay parametros de relajacion ajustados,
por dos motivos bien distintos:

  (a) La relajacion existe pero cae MUY por encima del rango medido.
      Es el caso de la ACETONA: su frecuencia de relajacion esta varias
      decenas de GHz por encima de los 5 GHz que llegaba a medir el
      equipo de NPL, asi que no se pudo ajustar ningun Debye (nota 23
      del reporte). Dentro del rango de este proyecto la parte REAL
      practicamente no baja todavia, pero la parte IMAGINARIA NO es
      cero: es la cola de subida de esa relajacion lejana.

  (b) No hay relajacion dipolar en absoluto, porque la molecula es NO
      POLAR. Es el caso del CICLOHEXANO y del FLUIDO DE SILICONA
      (polidimetilsiloxano Dow-Corning 200, 1 cSt). Su permitividad
      viene casi entera de la polarizacion electronica, que no se relaja
      hasta el optico/infrarrojo: son realmente constantes y de perdidas
      despreciables en todo RF y microondas.

Estos modelos se implementan aca igual que los demas (misma firma
`get_er_pat_xxx(frecs, T=..., verbose=...)`, mismo diccionario
PATRONES_TEORICOS), pero devuelven una permitividad CONSTANTE en
frecuencia. Para que nadie los use sin darse cuenta de esa limitacion,
cada modelo declara metadata en INFO_PATRONES (tipo, si modela perdidas,
rangos validos y un texto de advertencia), que la GUI muestra como cartel
de advertencia y el informe PDF imprime como caja destacada.

IMPORTANTE -- er'' de los modelos estaticos: se devuelve exactamente 0
salvo que se pase `tan_delta`. Para ciclohexano y silicona eso es
fisicamente correcto (perdidas despreciables). Para la acetona NO lo es:
ahi el 0 significa "no modelado", no "sin perdidas". Por eso las
funciones de error relativo del proyecto (funciones.error_relativo_
porcentual) devuelven NaN en vez de infinito cuando la referencia tiene
er''=0, y tanto la consola como el PDF muestran "n/a" en esa columna.
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


def get_er_pat_acetona(frecs, T=25.0, tan_delta=0.0, verbose=False):
    """
    Permitividad de la ACETONA, NPL MAT 23 (Gregory & Clarke), Seccion 6.

    ATENCION: esto NO es un modelo de Debye, es la permitividad ESTATICA
    (de baja frecuencia) puesta como constante en toda la banda. El
    reporte no publica parametros de relajacion para la acetona porque su
    frecuencia de relajacion queda muy por encima de los 5 GHz que
    llegaba a medir el equipo (nota 23).

    Consecuencias practicas, que conviene tener MUY presentes:

      - La parte REAL es una buena aproximacion mientras se trabaje
        bastante por debajo de la relajacion (o sea, en el rango tipico
        de este proyecto, 0.5-6 GHz): ahi er' todavia casi no bajo. Pero
        el modelo no captura esa caida, asi que cuanto mas alto se suba
        en frecuencia, mas optimista va a ser.
      - La parte IMAGINARIA se devuelve como 0, y eso es FALSO: la
        acetona tiene perdidas apreciables en GHz (es la cola de subida
        de su relajacion). El 0 significa "no modelado", no "sin
        perdidas". No tiene sentido calcular error relativo de er''
        contra esta referencia.

    Si se consigue un dato de perdidas de la literatura, se puede pasar
    `tan_delta` para que el modelo devuelva er = es*(1 - j*tan_delta).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz (solo se usan para darle forma al array: el
        valor devuelto es el mismo en todas).
    T : float
        Temperatura en °C (default 25 °C). Rango medido: 5-50 °C; se
        interpola linealmente entre los escalones tabulados de 5 °C.
    tan_delta : float
        Tangente de perdidas a aplicar (default 0 = sin perdidas
        modeladas). Ver arriba.
    verbose : bool
        Si es True, imprime la permitividad estatica que se termino
        usando.

    Retorna
    -------
    ndarray complejo con er' - j*er'' (constante en frecuencia).
    """
    if _ACETONA_DUDOSAS[0] <= float(T) <= _ACETONA_DUDOSAS[1]:
        warnings.warn(
            f"Acetona: las mediciones de NPL a {_ACETONA_DUDOSAS[0]:.0f} y "
            f"{_ACETONA_DUDOSAS[1]:.0f} °C figuran en italica en el reporte "
            f"porque pudo haber burbujeo dentro de la celda de medicion (la "
            f"acetona hierve a 56 °C). El valor a T={T}°C sale de esa "
            f"zona y puede ser menos confiable."
        )
    return _permitividad_estatica(frecs, _ACETONA_ESTATICA, T,
                                   nombre="Acetona (permitividad estatica)",
                                   rango_recomendado=(5, 50),
                                   tan_delta=tan_delta, verbose=verbose)


def get_er_pat_ciclohexano(frecs, T=25.0, tan_delta=0.0, verbose=False):
    """
    Permitividad del CICLOHEXANO, NPL MAT 23 (Gregory & Clarke), Seccion 6.

    El ciclohexano es una molecula NO POLAR: no tiene relajacion dipolar,
    asi que no existe modelo de Debye que ajustarle. Su permitividad
    (~2.0) viene casi entera de la polarizacion electronica, que no se
    relaja hasta el optico/infrarrojo. Es decir: aca la constante NO es
    una aproximacion de conveniencia, es el comportamiento fisico real en
    todo RF y microondas, con perdidas despreciables (er'' ~ 0).

    Por su permitividad baja y sus perdidas casi nulas es un buen control
    de banco -- de hecho NPL lo midio justamente para validar su celda
    contra Kienitz & Marsh -- pero por lo mismo es un MAL candidato a
    patron de calibracion de la sonda coaxial: al estar tan cerca del
    aire (er ~ 1) aporta muy poca informacion nueva y deja el sistema de
    ecuaciones de calibracion mal condicionado.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz (solo dan la forma del array de salida).
    T : float
        Temperatura en °C (default 25 °C). Rango medido: 10-50 °C -- no
        hay valor a 5 °C porque el ciclohexano funde a 6.5 °C.
    tan_delta : float
        Tangente de perdidas (default 0). Para el ciclohexano el 0 es
        fisicamente correcto: tan_delta real < 1e-4.
    verbose : bool
        Si es True, imprime la permitividad que se termino usando.

    Retorna
    -------
    ndarray complejo con er' - j*er'' (constante en frecuencia).
    """
    return _permitividad_estatica(frecs, _CICLOHEXANO_ESTATICA, T,
                                   nombre="Ciclohexano (permitividad estatica)",
                                   rango_recomendado=(10, 50),
                                   tan_delta=tan_delta, verbose=verbose)


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
    'acetona': "Acetona (solo ε estatica)",
    'ciclohexano': "Ciclohexano (no polar, ε constante)",
    'silicona': "Fluido de silicona 1 cSt (no polar, ε constante)",
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
    tipo='relajacion', modela_perdidas=True,
    perdidas_reales_despreciables=False,
    rango_temperatura_c=(10, 50), rango_frecuencia_ghz=(0.03, 5),
    nivel='info', advertencia=None,
)

INFO_PATRONES = {
    'agua': dict(_SIN_ADVERTENCIA,
                 rango_temperatura_c=(0, 100), rango_frecuencia_ghz=(0, 1000)),
    'alcohol_etilico': dict(_SIN_ADVERTENCIA),
    'alcohol_isopropilico': dict(_SIN_ADVERTENCIA),
    'metanol': dict(_SIN_ADVERTENCIA, rango_frecuencia_ghz=(0.03, 10)),
    'dmso': dict(_SIN_ADVERTENCIA,
                 rango_temperatura_c=(20, 50), rango_frecuencia_ghz=(0.03, 10)),
    'etilenglicol': dict(_SIN_ADVERTENCIA, rango_frecuencia_ghz=(0.03, 10)),
    'butanol': dict(_SIN_ADVERTENCIA),
    'propanol': dict(_SIN_ADVERTENCIA),

    'acetona': dict(
        tipo='estatico',
        modela_perdidas=False,
        perdidas_reales_despreciables=False,
        rango_temperatura_c=(5, 50),
        rango_frecuencia_ghz=(0, 5),
        nivel='critico',
        advertencia=(
            "Este modelo NO es una ecuacion de Debye: es la permitividad "
            "ESTATICA (de baja frecuencia) de la Seccion 6 del reporte NPL "
            "MAT 23, puesta como CONSTANTE en toda la banda. NPL no publica "
            "parametros de relajacion para la acetona porque su frecuencia "
            "de relajacion queda muy por encima de los 5 GHz que llegaba a "
            "medir el equipo (nota 23 del reporte).\n\n"
            "• La parte REAL (er') es una aproximacion razonable mientras "
            "se trabaje bastante por debajo de la relajacion, o sea en el "
            "rango tipico de este proyecto: ahi er' todavia casi no bajo. "
            "El modelo no captura esa caida, asi que cuanto mas se suba en "
            "frecuencia, mas optimista va a ser.\n"
            "• La parte IMAGINARIA (er'') se devuelve como 0, y eso es "
            "FALSO. La acetona SI tiene perdidas apreciables en GHz (es la "
            "cola de subida de su relajacion lejana). El 0 significa 'no "
            "modelado', no 'sin perdidas': el error relativo de er'' contra "
            "esta referencia no significa nada y se informa como n/a."
        ),
    ),

    'ciclohexano': dict(
        tipo='estatico',
        modela_perdidas=False,
        perdidas_reales_despreciables=True,
        rango_temperatura_c=(10, 50),
        rango_frecuencia_ghz=None,
        nivel='aviso',
        advertencia=(
            "Este modelo NO es una ecuacion de Debye, y no por una "
            "limitacion del equipo de medicion sino por fisica: el "
            "ciclohexano es una molecula NO POLAR, no tiene relajacion "
            "dipolar y por lo tanto no hay nada que ajustar. Su "
            "permitividad (~2.0) viene casi entera de la polarizacion "
            "electronica, que no se relaja hasta el optico/infrarrojo.\n\n"
            "• Aca la constante NO es una aproximacion de conveniencia: es "
            "el comportamiento real en todo RF y microondas. er'' ~ 0 "
            "tambien es correcto (tan_delta < 1e-4).\n"
            "• Rango de temperatura medido: 10-50 °C. No hay valor a 5 °C "
            "porque el ciclohexano funde a 6.5 °C.\n"
            "• Como el er'' de referencia es 0, el error relativo de er'' se "
            "informa como n/a (dividir por cero no significa nada).\n"
            "• Es un buen control de banco, pero MAL patron de calibracion "
            "de la sonda: al estar tan cerca del aire (er ~ 1) aporta poca "
            "informacion nueva y deja mal condicionado el sistema de "
            "ecuaciones de calibracion."
        ),
    ),

    'silicona': dict(
        tipo='estatico',
        modela_perdidas=False,
        perdidas_reales_despreciables=True,
        rango_temperatura_c=(5, 35),
        rango_frecuencia_ghz=None,
        nivel='aviso',
        advertencia=(
            "Este modelo NO es una ecuacion de Debye. El fluido de silicona "
            "Dow-Corning 200 de 1 cSt (polidimetilsiloxano) es no polar: "
            "igual que el ciclohexano, no tiene relajacion dipolar en RF ni "
            "en microondas, asi que su permitividad (~2.3) es realmente "
            "constante y sus perdidas despreciables (er'' ~ 0 correcto).\n\n"
            "• OJO con la temperatura: NPL solo publica mediciones de 5 a "
            "35 °C (no hay 40, 45 ni 50 °C), bastante menos rango que el "
            "resto de los liquidos. Fuera de ahi el valor se satura en el "
            "extremo mas cercano.\n"
            "• Como el er'' de referencia es 0, el error relativo de er'' se "
            "informa como n/a.\n"
            "• Mismo reparo que el ciclohexano como patron de calibracion: "
            "su permitividad esta demasiado cerca de la del aire."
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
    """True si el modelo devuelve permitividad CONSTANTE en frecuencia (no
    tiene ecuacion de relajacion ajustada)."""
    return info_patron(clave).get('tipo') == 'estatico'


def modela_perdidas(clave):
    """True si el er'' que devuelve el modelo es un valor fisico
    calculado; False si es 0 por no estar modelado. Lo usan el pipeline de
    analisis y el informe PDF para no publicar un error relativo de er''
    que no significaria nada."""
    return bool(info_patron(clave).get('modela_perdidas', True))


def resumen_corto_patron(clave):
    """Una linea para mostrar al lado del nombre del modelo en tablas y
    listados (o None si es un modelo de relajacion normal, donde no hace
    falta aclarar nada)."""
    info = info_patron(clave)
    if info.get('tipo') != 'estatico':
        return None
    t_min, t_max = info.get('rango_temperatura_c', (None, None))
    if info.get('perdidas_reales_despreciables'):
        base = "ε constante (no polar, sin relajacion)"
    else:
        base = "ε estatica constante, er'' NO modelado"
    if t_min is not None:
        return f"{base} — medido {t_min:.0f}-{t_max:.0f} °C"
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