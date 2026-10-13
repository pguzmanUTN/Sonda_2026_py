"""
funciones.py
------------
Algoritmos de conversion de coeficiente de reflexion S11 (medido con la
sonda coaxial open-ended) a permitividad dielectrica compleja del material
bajo ensayo (MUT), siguiendo el modelo de admitancia de la sonda:

    Y = j*w*er*C0 + er^(5/2)*G0

Se implementan los dos metodos descriptos en la bibliografia de la catedra
(Higa/Cismondi/Grass 2016; Henze et al. 2024 - ARGENCON):

  * `get_er_DUTm`          -> Metodo "simplificado" (Gn = 0), usa 3 patrones
                               de calibracion: corto, aire (abierto) y un
                               3er liquido patron ("patron 3"). Es una
                               formula algebraica cerrada, valida sobre
                               todo a frecuencias mas bajas / materiales
                               de bajas perdidas.

  * `get_er_DUT_completo`  -> Metodo "completo" (con conductancia normalizada
                               Gn != 0), usa 4 patrones de calibracion:
                               corto, aire, "patron 3" y un 4to liquido
                               patron ("patron 4"). Resuelve, para cada
                               frecuencia, la ecuacion polinomica de 5to
                               orden en raiz(er) que aparece en el modelo
                               de admitancia completo. Es mas preciso en un
                               rango de frecuencias mas amplio.

Este modulo es AGNOSTICO de que liquidos especificos se usan como patron 3
y patron 4 (tradicionalmente agua destilada y alcohol isopropilico, pero
puede ser cualquier par de liquidos con un modelo teorico conocido): en vez
de recibir una temperatura y calcular la permitividad teorica internamente,
estas funciones reciben directamente el array Er_patron3/Er_patron4 ya
evaluado (ver `analisis_permitividad.py`, que resuelve que modelo y
temperatura usar para cada patron a partir de la configuracion, via
`Patrones.PATRONES_TEORICOS`). Esto mantiene a este archivo enfocado
unicamente en el algebra S11 -> er, sin acoplarlo a los modelos de
liquidos concretos de Patrones.py.

Ambos metodos devuelven er = er' - j*er'' (parte imaginaria positiva para
perdidas) evaluado en las mismas frecuencias que los datos de entrada.
"""
import numpy as np

from Patrones import ER_AIRE


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def S11_to_Y11(frec, S11, z0=50):
    """
    Convierte un coeficiente de reflexion S11 a admitancia Y11 normalizada
    a la impedancia de referencia z0 (50 ohm por defecto).
    """
    y0 = 1 / z0
    Y11 = y0 * ((1 - S11) / (1 + S11))
    return {'Frec': frec, 'Complex': Y11}


def convertir_a_rectangular(modulo_db, fase_grados):
    """
    Convierte un par (modulo en dB, fase en grados) a un numero complejo en
    coordenadas rectangulares. Util si en algun momento se cargan datos de
    forma manual en formato dB/fase en lugar de leerlos con skrf.
    """
    modulo = 10 ** (modulo_db / 20)
    fase_radianes = np.radians(fase_grados)
    return modulo * np.exp(1j * fase_radianes)


# ---------------------------------------------------------------------------
# Metodo simplificado (3 patrones: corto, aire, patron3) -- Gn = 0
# ---------------------------------------------------------------------------
def get_er_DUTm(frecs, S11_medido, S11_patron3, S11_aire, S11_corto,
                 Er_patron3, Er_aire=ER_AIRE):
    """
    Calcula la permitividad relativa compleja del DUT a partir del modelo
    simplificado (conductancia normalizada Gn = 0), usando como patrones de
    calibracion el cortocircuito (referencia Y->inf), el aire/circuito
    abierto (er ~= 1) y un 3er liquido patron ("patron 3": tradicionalmente
    agua destilada, pero puede ser cualquier liquido con permitividad
    conocida -- ver `Patrones.PATRONES_TEORICOS`).

    Es una formula algebraica cerrada (no iterativa), equivalente a la
    ecuacion (18) de Higa/Cismondi/Grass (2016) y de Henze et al. (2024).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz (deben coincidir con las de todos los S11).
    S11_medido : array_like complejo
        S11 medido del material bajo ensayo (DUT).
    S11_patron3, S11_aire, S11_corto : array_like complejo
        S11 medidos de los patrones de calibracion.
    Er_patron3 : array_like complejo
        Permitividad relativa teorica del 3er patron, YA EVALUADA en
        `frecs` a la temperatura real de calibracion (p.ej.
        `Patrones.get_er_agua(frecs, T)` si el patron 3 es agua, o
        `Patrones.PATRONES_TEORICOS[modelo](frecs, T=temperatura)` para
        cualquier otro liquido).
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).

    Retorna
    -------
    ndarray complejo: er'(f) - j*er''(f) del DUT.
    """
    frecs = np.asarray(frecs, dtype=float)
    S11_medido = np.asarray(S11_medido, dtype=complex)
    S11_patron3 = np.asarray(S11_patron3, dtype=complex)
    S11_aire = np.asarray(S11_aire, dtype=complex)
    S11_corto = np.asarray(S11_corto, dtype=complex)
    Er_patron3 = np.asarray(Er_patron3, dtype=complex)

    # Ecuacion (18) de Higa/Cismondi/Grass (2016): coeficientes de la
    # combinacion lineal de las permitividades conocidas del patron 3
    # (Er_patron3) y del aire (Er_aire), en funcion de los S11 medidos.
    coef_patron3 = ((S11_medido - S11_aire) * (S11_corto - S11_patron3)) / \
                   ((S11_medido - S11_corto) * (S11_patron3 - S11_aire))

    coef_aire = ((S11_medido - S11_patron3) * (S11_aire - S11_corto)) / \
                ((S11_medido - S11_corto) * (S11_patron3 - S11_aire))

    return -coef_patron3 * Er_patron3 - coef_aire * Er_aire


# ---------------------------------------------------------------------------
# Metodo completo (4 patrones: corto, aire, agua, alcohol isopropilico)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Metodo completo (4 patrones: corto, aire, patron3, patron4)
# ---------------------------------------------------------------------------
def calcular_Gn(frecs, S11_patron3, S11_aire, S11_patron4, S11_corto,
                 Er_patron3, Er_patron4, Er_aire=ER_AIRE):
    """
    Calcula la conductancia normalizada Gn(f) del modelo de admitancia
    completo (ecuacion (17) de Higa/Cismondi/Grass 2016 y de Henze et al.
    2024/ARGENCON), a partir UNICAMENTE de los 4 patrones de calibracion
    (corto, aire, patron 3, patron 4). Notar que Gn NO depende del S11 del
    DUT: es una propiedad de la sonda/calibracion, la misma para todos los
    materiales que se analicen con esa calibracion.

    `get_er_DUT_completo` llama a esta funcion internamente para armar el
    polinomio de 5to orden, pero se expone tambien aca aparte para poder
    usarla como DIAGNOSTICO de la calibracion (ver
    `analisis_permitividad.graficar_Gn`): como Gn esta relacionada con
    G0/(j*w*C0) -- una propiedad fisica de la sonda que varia en forma
    continua con la frecuencia -- la curva Gn(f) resultante tiene que
    verse suave. Un salto brusco o un pico aislado suele ser sintoma de
    un problema con la medicion de alguno de los 4 patrones, tipicamente
    el patron 4 (tradicionalmente alcohol isopropilico), que es el UNICO
    que entra en el calculo de Gn y en ningun otro lado del metodo
    simplificado -- asi que un problema ahi puede pasar desapercibido si
    solo se mira el chequeo de calibracion del patron 3.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    S11_patron3, S11_aire, S11_patron4, S11_corto : array_like complejo
        S11 medidos de los 4 patrones de calibracion.
    Er_patron3, Er_patron4 : array_like complejo
        Permitividad relativa teorica de los patrones 3 y 4, YA EVALUADA
        en `frecs` a la temperatura real de calibracion de cada uno (ver
        `Patrones.PATRONES_TEORICOS`).
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).

    Retorna
    -------
    ndarray complejo: Gn(f), mismo largo que `frecs`.
    """
    r_patron3 = np.asarray(S11_patron3, dtype=complex)
    r_aire = np.asarray(S11_aire, dtype=complex)
    r_patron4 = np.asarray(S11_patron4, dtype=complex)
    r_corto = np.asarray(S11_corto, dtype=complex)
    Er_patron3 = np.asarray(Er_patron3, dtype=complex)
    Er_patron4 = np.asarray(Er_patron4, dtype=complex)

    # Gn = -(D41*D32*e4 + D42*D13*e3 + D43*D21*e2) /
    #       (D41*D32*e4^2.5 + D42*D13*e3^2.5 + D43*D21*e2^2.5)
    # con patron1=corto, patron2=aire (e2=Er_aire), patron3=Er_patron3,
    # patron4=Er_patron4.
    d41 = r_patron4 - r_corto
    d32 = r_patron3 - r_aire
    d42 = r_patron4 - r_aire
    d13 = r_corto - r_patron3
    d43 = r_patron4 - r_patron3
    d21 = r_aire - r_corto

    num_Gn = (d41 * d32 * Er_patron4
              + d42 * d13 * Er_patron3
              + d43 * d21 * Er_aire)
    den_Gn = (d41 * d32 * (Er_patron4 ** 2.5)
              + d42 * d13 * (Er_patron3 ** 2.5)
              + d43 * d21 * (Er_aire ** 2.5))
    return -num_Gn / den_Gn


def _resolver_raiz_fisica(coeficientes, semilla):
    """
    Resuelve Gn*u^5 + u^2 + c0 = 0 (con u = sqrt(er)) y devuelve, entre las
    5 raices er = u^2, la mas cercana a `semilla`, priorizando ademas
    soluciones con parte real positiva (fisicamente validas para un
    material pasivo).

    La eleccion de una buena `semilla` es critica: el polinomio tiene 5
    raices y varias pueden estar cerca entre si. Ver `get_er_DUT_completo`
    para la estrategia de "marcha en frecuencia" usada para construir una
    semilla confiable en cada punto.
    """
    raices_u = np.roots(coeficientes)
    raices_er = raices_u ** 2

    candidatas = raices_er[np.real(raices_er) > 0]
    if candidatas.size == 0:
        candidatas = raices_er

    idx = np.argmin(np.abs(candidatas - semilla))
    return candidatas[idx]


def get_er_DUT_completo(frecs, S11_medido, S11_patron3, S11_aire,
                         S11_patron4, S11_corto, Er_patron3, Er_patron4,
                         Er_aire=ER_AIRE, verbose=False,
                         estrategia_semilla='minimo_gn', Gn=None):
    """
    Calcula la permitividad relativa compleja del DUT a partir del modelo
    de admitancia "completo" (con conductancia de radiacion normalizada Gn),
    usando 4 patrones de calibracion: cortocircuito, aire, "patron 3" y
    "patron 4" (tradicionalmente agua destilada y alcohol isopropilico,
    pero puede ser cualquier par de liquidos con permitividad conocida).
    Si `Gn` se pasa explicitamente (p.ej. precalculado o acondicionado con
    `acondicionar_gn`), se utiliza directamente ese vector en lugar de
    recalcularlo.


    Para cada frecuencia se arma la ecuacion polinomica de 5to orden en
    u = sqrt(er):

        Gn*u^5 + u^2 + c0 = 0

    y se resuelve exactamente con `numpy.roots` (en lugar de un solver
    iterativo real como `scipy.optimize.fsolve`, que no admite residuos
    complejos). Entre las 5 raices se elige la fisicamente correcta
    "marchando" en frecuencia: se arranca en el punto donde |Gn(f)| es
    MINIMO en todo el barrido (ahi el metodo simplificado, que asume
    Gn=0, da la mejor semilla posible) y desde ahi se avanza hacia ambos
    lados -- frecuencias mas altas y mas bajas -- usando en cada paso
    como semilla la solucion ya aceptada en el punto vecino, aprovechando
    que la permitividad de un material real varia en forma continua con
    la frecuencia. Con `verbose=True` se imprime en que frecuencia se
    encontro el minimo de |Gn| (donde arranca la marcha) y un aviso si en
    algun tramo la curva resultante muestra un salto brusco (posible
    eleccion de rama incorrecta), para que se pueda revisar esa zona.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    S11_medido : array_like complejo
        S11 medido del material bajo ensayo (DUT).
    S11_patron3, S11_aire, S11_patron4, S11_corto : array_like complejo
        S11 medidos de los 4 patrones de calibracion.
    Er_patron3, Er_patron4 : array_like complejo
        Permitividad relativa teorica de los patrones 3 y 4, YA EVALUADA
        en `frecs` a la temperatura real de calibracion de cada uno (ver
        `Patrones.PATRONES_TEORICOS`). Cuidado: como Er_patron4 entra en
        el calculo de Gn (la conductancia normalizada), una temperatura
        mal puesta ahi afecta el resultado de TODOS los materiales
        analizados con el metodo completo, no solo el del propio patron 4.
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).
    estrategia_semilla : {'minimo_gn', 'frecuencia_maxima'}
        Donde arranca la "marcha" en frecuencia (ver mas abajo):

        - 'minimo_gn' (default): arranca en el punto donde |Gn(f)| es
          MINIMO en todo el barrido -- ahi el metodo simplificado (que
          asume Gn=0) da la mejor semilla posible, sea cual sea la
          frecuencia donde ocurra -- y marcha desde ahi hacia AMBOS lados
          (frecuencias mas altas y mas bajas).
        - 'frecuencia_maxima': arranca siempre en la frecuencia MAS ALTA
          del barrido (equivalente a marchar en una sola direccion,
          descendente). Es el comportamiento de versiones anteriores de
          esta funcion, valido solo si C0/G0 son aproximadamente
          constantes en todo el barrido (ahi |Gn| decrece en forma
          monotona con la frecuencia, y el maximo es tambien el minimo de
          |Gn|). En patrones reales eso no siempre se cumple -- la sonda
          deja de comportarse como un capacitor+conductancia ideales y
          empieza a irradiar de forma mas compleja a medida que la
          longitud de onda se acerca al tamano de la sonda -- por eso
          'minimo_gn' es el default; esta opcion se deja disponible para
          comparar ambas estrategias o por si en algun caso puntual se
          prefiere forzar el comportamiento clasico.

    Retorna
    -------
    ndarray complejo: er'(f) - j*er''(f) del DUT.
    """
    frecs = np.asarray(frecs, dtype=float)
    S11_medido = np.asarray(S11_medido, dtype=complex)
    r_patron3 = np.asarray(S11_patron3, dtype=complex)
    r_aire = np.asarray(S11_aire, dtype=complex)
    r_corto = np.asarray(S11_corto, dtype=complex)
    Er_patron3 = np.asarray(Er_patron3, dtype=complex)

    # Conductancia normalizada Gn. Si no se paso explicitamente, se
    # calcula a partir del patron 4.
    if Gn is None:
        Gn = calcular_Gn(frecs, S11_patron3, S11_aire, S11_patron4, S11_corto,
                          Er_patron3, Er_patron4, Er_aire=Er_aire)
    else:
        Gn = np.asarray(Gn, dtype=complex)

    # --- Coeficientes X, Z de la ecuacion principal (idem metodo simple) ---
    d32 = r_patron3 - r_aire
    d13 = r_corto - r_patron3
    d21 = r_aire - r_corto

    dm2 = S11_medido - r_aire
    dm1 = S11_medido - r_corto
    dm3 = S11_medido - r_patron3

    X = (dm2 * d13) / (dm1 * d32)
    Z = (dm3 * d21) / (dm1 * d32)

    # Semilla candidata en cada frecuencia: prediccion del metodo
    # simplificado (Gn=0), evaluada en todo el barrido.
    semillas = get_er_DUTm(frecs, S11_medido, S11_patron3, S11_aire, S11_corto,
                            Er_patron3, Er_aire=Er_aire)

    # El polinomio de 5to orden en sqrt(er) tiene 5 raices, y cuando Gn no
    # es despreciable varias pueden quedar cerca entre si. Usar siempre la
    # semilla del metodo simplificado para elegir la raiz es poco confiable
    # justo en el regimen donde Gn importa (que es donde el simplificado
    # mas se degrada). En cambio, se "marcha" en frecuencia: la permitividad
    # de un material real varia en forma continua con la frecuencia, asi
    # que salvo en el punto de arranque se usa como semilla la solucion ya
    # aceptada en el punto vecino.
    #
    # El punto de arranque importa, y tiene que ser donde el metodo
    # simplificado (Gn=0) sea la mejor aproximacion posible al completo --
    # es decir, donde |Gn(f)| sea MINIMO. Una version anterior asumia que
    # eso pasa siempre en la frecuencia mas alta del barrido (razonable si
    # Gn ~ G0/(j*w*C0) con C0/G0 constantes, ya que ahi |Gn| decrece en
    # forma monotona con la frecuencia) -- pero en patrones reales C0 y G0
    # no son necesariamente constantes en todo el barrido (la sonda deja de
    # comportarse como un capacitor+conductancia ideales y empieza a
    # irradiar de forma mas compleja a medida que la longitud de onda se
    # acerca al tamano de la sonda), asi que |Gn(f)| puede tener su minimo
    # en cualquier punto -- a veces en el medio del barrido, no en un
    # extremo (se puede confirmar mirando el diagnostico de
    # `analisis_permitividad.graficar_Gn`). Por eso ahora se busca
    # explicitamente el minimo de |Gn(f)| en TODO el barrido, se arranca
    # ahi (con la semilla del metodo simplificado, la mejor posible en ese
    # punto), y se marcha hacia AMBOS lados en frecuencia -- hacia arriba y
    # hacia abajo -- usando siempre la solucion del vecino inmediato ya
    # resuelto como semilla del siguiente.
    # Termino independiente del polinomio Gn*u^5 + u^2 + c0 = 0, en todo el
    # barrido de una vez. Corregido: antes el aire entraba como `1.0 + Gn`,
    # o sea con er = 1 exacto, mientras que calcular_Gn y get_er_DUTm usan
    # Er_aire = 1.0006. Esa inconsistencia dejaba un error sistematico de
    # hasta ~0.03% aun con datos perfectos.
    coef_X = Er_patron3 + Gn * (Er_patron3 ** 2.5)
    coef_1 = Er_aire + Gn * (Er_aire ** 2.5)
    c0 = X * coef_X + Z * coef_1

    return _marchar_en_frecuencia(frecs, Gn, c0, semillas, estrategia_semilla,
                                  verbose=verbose, etiqueta="get_er_DUT_completo")


def _marchar_en_frecuencia(frecs, Gn, c0, semillas, estrategia_semilla='minimo_gn',
                           verbose=False, etiqueta="marcha"):
    """
    Resuelve, frecuencia por frecuencia, el polinomio Gn*u^5 + u^2 + c0 = 0
    (u = sqrt(er)) eligiendo la raiz fisica por "marcha en frecuencia":
    arranca donde |Gn| es minimo (ahi la semilla del modelo con Gn = 0 es
    la mejor posible) y avanza hacia ambos lados usando como semilla la
    solucion ya aceptada en el punto vecino. Ver el docstring de
    `get_er_DUT_completo` para la justificacion completa.

    Lo comparten `get_er_DUT_completo` (4 patrones, solucion exacta) y
    `get_er_DUT_minimos_cuadrados` (N patrones, calibracion ajustada): la
    unica diferencia entre los dos metodos esta en como se obtienen Gn y
    c0, no en como se resuelve el polinomio.
    """
    frecs = np.asarray(frecs, dtype=float)
    idx_gn_minimo = int(np.argmin(np.abs(Gn)))
    orden = np.argsort(frecs)  # indices en orden ASCENDENTE de frecuencia

    if estrategia_semilla == 'minimo_gn':
        pos_inicio = int(np.where(orden == idx_gn_minimo)[0][0])
    elif estrategia_semilla == 'frecuencia_maxima':
        # El ultimo en orden ASCENDENTE es la frecuencia mas alta. Con
        # pos_inicio ahi, el bucle "hacia arriba" de mas abajo no tiene
        # nada que recorrer (ya es el ultimo indice), asi que el resultado
        # es exactamente el comportamiento clasico: una sola marcha
        # descendente arrancando en la frecuencia mas alta.
        pos_inicio = len(orden) - 1
    else:
        raise ValueError(
            f"estrategia_semilla invalida: {estrategia_semilla!r} "
            f"(opciones validas: 'minimo_gn', 'frecuencia_maxima')")

    if verbose:
        print(f"  [{etiqueta}] estrategia='{estrategia_semilla}': "
              f"la marcha arranca en f={frecs[orden[pos_inicio]] / 1e9:.4f} GHz "
              f"(|Gn| ahi = {abs(Gn[orden[pos_inicio]]):.4g}; el minimo real de "
              f"|Gn| en todo el barrido es {abs(Gn[idx_gn_minimo]):.4g}, en "
              f"f={frecs[idx_gn_minimo] / 1e9:.4f} GHz).")

    def _resolver_en(n, semilla):
        # Gn*u^5 + 0*u^4 + 0*u^3 + 1*u^2 + 0*u + c0 = 0
        return _resolver_raiz_fisica([Gn[n], 0, 0, 1, 0, c0[n]], semilla)

    Er_out = np.zeros_like(frecs, dtype=complex)

    n0 = orden[pos_inicio]
    Er_out[n0] = _resolver_en(n0, semillas[n0])

    semilla_actual = Er_out[n0]
    for pos in range(pos_inicio + 1, len(orden)):  # hacia frecuencias mas altas
        n = orden[pos]
        Er_out[n] = _resolver_en(n, semilla_actual)
        semilla_actual = Er_out[n]

    semilla_actual = Er_out[n0]
    for pos in range(pos_inicio - 1, -1, -1):  # hacia frecuencias mas bajas
        n = orden[pos]
        Er_out[n] = _resolver_en(n, semilla_actual)
        semilla_actual = Er_out[n]

    if verbose:
        _diagnosticar_saltos(frecs, Er_out)

    return Er_out


# ---------------------------------------------------------------------------
# Metodo por CUADRADOS MINIMOS (N >= 4 patrones: calibracion redundante)
# ---------------------------------------------------------------------------
# Motivacion: Kaatze (2007), "Reference liquids for the calibration of
# dielectric sensors and measurement instruments", Meas. Sci. Technol. 18
# 967, recomienda calibraciones REDUNDANTES, con mas patrones que los
# estrictamente necesarios:
#   - pag. 968: "Redundant calibrations with more than two reference
#     liquids are therefore being recommended."
#   - pag. 969: "...in order to improve the accuracy of the measurements
#     by redundant reference routines..."
#   - pag. 974 (conclusiones): cuatro liquidos de referencia -- aire,
#     agua, ciclohexano y metanol a 25 °C -- mas DMSO "for redundancy and
#     consistency checks".
# OJO: Kaatze NO dice como resolver el sistema redundante (no aparecen
# "least squares" ni "fit" en todo el paper; remite a sus referencias
# [15] Evans & Michelson 1995 y [33] Folgero 1996). Resolverlo por
# cuadrados minimos es la eleccion de ESTA implementacion: es la forma
# estandar de resolver un sistema sobredeterminado, y minimizando en el
# espacio de S11 es ademas el estimador de maxima verosimilitud si el
# ruido del VNA es gaussiano y parejo entre mediciones.
#
# Modelo (el mismo que el metodo completo, sin simplificar nada):
#
#     F(er) = er + Gn*er^(5/2) = (a*rho + b) / (c*rho + 1)
#
# donde rho es el S11 medido. (a, b, c) es la transformacion bilineal que
# absorbe la caja de error del VNA/cable y la capacidad C0 de la sonda --
# es la ecuacion (3) de Kaatze -- y Gn es la conductancia de radiacion
# normalizada. Son 4 incognitas complejas por frecuencia: con 4 patrones
# el sistema tiene solucion exacta (y reproduce get_er_DUT_completo); con
# 5 o mas queda sobredeterminado y se ajusta.
#
# El cortocircuito es el caso limite F -> inf, que en la bilineal implica
# c*rho_corto + 1 = 0, o sea rho_corto = -1/c.
# ---------------------------------------------------------------------------

def _rho_modelo(p, er):
    """S11 que predice el modelo para un patron de permitividad `er`
    (None = cortocircuito), dados p = (a, b, c, Gn) complejos. Devuelve
    (rho, derivadas respecto de a, b, c, Gn). El modelo es holomorfo en los
    4 parametros, asi que la derivada compleja alcanza para Gauss-Newton."""
    a, b, c, Gn = p
    if er is None:  # cortocircuito
        rho = -1.0 / c
        cero = np.zeros_like(rho)
        return rho, (cero, cero, 1.0 / c ** 2, cero)
    e25 = er ** 2.5
    F = er + Gn * e25
    D = c * F - a
    rho = (b - F) / D
    d_a = (b - F) / D ** 2
    d_b = 1.0 / D
    d_c = -(b - F) * F / D ** 2
    d_Gn = (a - b * c) / D ** 2 * e25
    return rho, (d_a, d_b, d_c, d_Gn)


def calibrar_minimos_cuadrados(frecs, patrones, max_iter=60, tol=1e-12, verbose=False):
    """
    Calibracion redundante de la sonda por cuadrados minimos.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    patrones : list[dict]
        Uno por patron medido, con las claves:
          'nombre' : str (para diagnosticos)
          'S11'    : array complejo medido, en las mismas `frecs`
          'er'     : array complejo con la permitividad teorica del patron
                     YA evaluada en `frecs` y a su temperatura, o None si
                     el patron es el CORTOCIRCUITO
          'peso'   : float opcional (default 1). Peso relativo de ese
                     patron en el ajuste.
        Hacen falta al menos 4 patrones (4 incognitas complejas). Tipico:
        corto, aire, agua, metanol, ciclohexano, ...
    max_iter, tol :
        Control del Levenberg-Marquardt (se hace una corrida por
        frecuencia, todas en paralelo).
    verbose : bool
        Imprime un resumen de la convergencia y de los residuos.

    Retorna
    -------
    dict con:
      'a', 'b', 'c', 'Gn' : arrays complejos, uno por frecuencia
      'nombres'           : lista de nombres de los patrones
      'residuos'          : dict {nombre: array |rho_med - rho_modelo|}
      'residuo_rms'       : array, rms sobre patrones, por frecuencia
      'grados_libertad'   : 2*N - 8 (reales). 0 = solucion exacta, sin
                            redundancia (y residuos nulos)
      'iteraciones'       : array de iteraciones usadas por frecuencia
      'convergio'         : array bool por frecuencia
    """
    frecs = np.asarray(frecs, dtype=float)
    nf = frecs.size
    npat = len(patrones)
    if npat < 4:
        raise ValueError(
            f"La calibracion por cuadrados minimos necesita al menos 4 patrones "
            f"(hay {npat}): son 4 incognitas complejas por frecuencia.")

    nombres = [pt.get('nombre', f"patron {k + 1}") for k, pt in enumerate(patrones)]
    rho = np.array([np.asarray(pt['S11'], dtype=complex) for pt in patrones])      # (npat, nf)
    ers = [None if pt.get('er') is None else np.asarray(pt['er'], dtype=complex)
           for pt in patrones]
    w = np.sqrt(np.array([float(pt.get('peso', 1.0)) for pt in patrones]))          # (npat,)

    # --- 1) Arranque con Gn = 0: bilineal lineal en (a, b, c) -------------
    #   dielectrico: a*rho + b - er*rho*c = er      corto: rho*c = -1
    A = np.zeros((nf, npat, 3), dtype=complex)
    y = np.zeros((nf, npat), dtype=complex)
    for k in range(npat):
        if ers[k] is None:
            A[:, k, 2] = rho[k]
            y[:, k] = -1.0
        else:
            A[:, k, 0] = rho[k]
            A[:, k, 1] = 1.0
            A[:, k, 2] = -ers[k] * rho[k]
            y[:, k] = ers[k]
    A *= w[None, :, None]
    y *= w[None, :]
    p = np.zeros((nf, 4), dtype=complex)
    for n in range(nf):
        sol, *_ = np.linalg.lstsq(A[n], y[n], rcond=None)
        p[n, :3] = sol
    # p[:, 3] (Gn) arranca en 0

    # --- 2) Levenberg-Marquardt en paralelo sobre todas las frecuencias ---
    def _residuos_y_jac(pp):
        r = np.zeros((nf, npat), dtype=complex)
        J = np.zeros((nf, npat, 4), dtype=complex)
        for k in range(npat):
            rm, dm = _rho_modelo((pp[:, 0], pp[:, 1], pp[:, 2], pp[:, 3]), ers[k])
            r[:, k] = w[k] * (rho[k] - rm)
            J[:, k, :] = w[k] * np.stack(np.broadcast_arrays(*dm), axis=-1)
        return r, J

    lam = np.full(nf, 1e-3)
    iteraciones = np.zeros(nf, dtype=int)
    convergio = np.zeros(nf, dtype=bool)
    r, J = _residuos_y_jac(p)
    costo = np.sum(np.abs(r) ** 2, axis=1)
    for it in range(max_iter):
        activos = ~convergio
        if not np.any(activos):
            break
        # escalado de columnas (Marquardt): hace comparables a (a, b, c)
        # con Gn, que tienen escalas MUY distintas.
        esc = np.sqrt(np.sum(np.abs(J) ** 2, axis=1))            # (nf, 4)
        esc[esc == 0] = 1.0
        Js = J / esc[:, None, :]
        JhJ = np.einsum('nki,nkj->nij', Js.conj(), Js)
        Jhr = np.einsum('nki,nk->ni', Js.conj(), r)
        M = JhJ + lam[:, None, None] * np.eye(4)[None]
        try:
            # el [..., None] hace falta en numpy >= 2: sin eso, solve()
            # interpreta el vector apilado (nf, 4) como una matriz
            paso_s = np.linalg.solve(M, Jhr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            paso_s = np.array([np.linalg.lstsq(M[n], Jhr[n], rcond=None)[0] for n in range(nf)])
        paso = paso_s / esc
        p_nuevo = p + np.where(activos[:, None], paso, 0)
        r_n, J_n = _residuos_y_jac(p_nuevo)
        costo_n = np.sum(np.abs(r_n) ** 2, axis=1)
        mejora = (costo_n <= costo) & activos
        p[mejora] = p_nuevo[mejora]
        r[mejora] = r_n[mejora]; J[mejora] = J_n[mejora]
        rel = np.abs(costo[mejora] - costo_n[mejora]) / np.maximum(costo[mejora], 1e-300)
        tam = np.linalg.norm(paso_s[mejora], axis=1)
        costo[mejora] = costo_n[mejora]
        lam[mejora] = np.maximum(lam[mejora] / 3.0, 1e-12)
        lam[activos & ~mejora] *= 4.0
        iteraciones[activos] += 1
        idx = np.where(mejora)[0]
        listo = (rel < tol) | (tam < 1e-10) | (costo[idx] < 1e-28)
        convergio[idx[listo]] = True
        convergio |= lam > 1e10   # estancado: ya no puede mejorar

    # --- 3) Diagnostico: residuo de cada patron --------------------------
    residuos = {}
    for k in range(npat):
        rm, _ = _rho_modelo((p[:, 0], p[:, 1], p[:, 2], p[:, 3]), ers[k])
        residuos[nombres[k]] = np.abs(rho[k] - rm)
    residuo_rms = np.sqrt(np.mean(np.array(list(residuos.values())) ** 2, axis=0))

    if verbose:
        print(f"  [cuadrados minimos] {npat} patrones, {2 * npat - 8} grados de libertad; "
              f"convergio en {int(np.sum(convergio))}/{nf} frecuencias "
              f"(max {int(iteraciones.max())} iteraciones).")
        for nom, res in residuos.items():
            print(f"      residuo rms de '{nom}': {np.sqrt(np.mean(res ** 2)):.2e} (en S11)")

    return {
        'a': p[:, 0], 'b': p[:, 1], 'c': p[:, 2], 'Gn': p[:, 3],
        'nombres': nombres, 'residuos': residuos, 'residuo_rms': residuo_rms,
        'grados_libertad': 2 * npat - 8,
        'iteraciones': iteraciones, 'convergio': convergio,
    }


def reajustar_abc_con_gn_fijo(frecs, patrones, Gn_fijo, calibracion_base=None, verbose=False):
    """
    Dado un vector `Gn_fijo` (por ejemplo el `Gn` acondicionado mediante
    tapering, transicion cuadratica o cero clamping), reajusta los
    coeficientes bilineales (a, b, c):
      - En las frecuencias donde `Gn_fijo` coincide con `calibracion_base['Gn']`
        (zona no modificada por el acondicionamiento), conserva exactamente
        los `(a, b, c)` de `calibracion_base`.
      - En las frecuencias donde `Gn` fue modificado por el acondicionamiento,
        resuelve el sistema bilineal lineal por cuadrados minimos para (a, b, c)
        con F_k = er_k + Gn_fijo * er_k^(5/2), evitando que un reajuste no lineal
        en bajas frecuencias desplace el polo c hacia la region dielectrica.
    """
    frecs = np.asarray(frecs, dtype=float)
    Gn_fijo = np.asarray(Gn_fijo, dtype=complex)
    nf = frecs.size
    npat = len(patrones)

    nombres = [pt.get('nombre', f"patron {k + 1}") for k, pt in enumerate(patrones)]
    rho = np.array([np.asarray(pt['S11'], dtype=complex) for pt in patrones])
    ers = [None if pt.get('er') is None else np.asarray(pt['er'], dtype=complex)
           for pt in patrones]
    w = np.sqrt(np.array([float(pt.get('peso', 1.0)) for pt in patrones]))

    A = np.zeros((nf, npat, 3), dtype=complex)
    y = np.zeros((nf, npat), dtype=complex)
    for k in range(npat):
        if ers[k] is None:
            A[:, k, 2] = rho[k]
            y[:, k] = -1.0
        else:
            Fk = ers[k] + Gn_fijo * (ers[k] ** 2.5)
            A[:, k, 0] = rho[k]
            A[:, k, 1] = 1.0
            A[:, k, 2] = -Fk * rho[k]
            y[:, k] = Fk
    A *= w[None, :, None]
    y *= w[None, :]
    abc = np.zeros((nf, 3), dtype=complex)
    if calibracion_base is not None and 'Gn' in calibracion_base:
        gn_orig = np.asarray(calibracion_base['Gn'], dtype=complex)
        sin_cambio = np.abs(Gn_fijo - gn_orig) <= 1e-15 * np.maximum(np.abs(gn_orig), 1.0)
    else:
        sin_cambio = np.zeros(nf, dtype=bool)

    for n in range(nf):
        if sin_cambio[n]:
            abc[n, 0] = calibracion_base['a'][n]
            abc[n, 1] = calibracion_base['b'][n]
            abc[n, 2] = calibracion_base['c'][n]
        else:
            sol, *_ = np.linalg.lstsq(A[n], y[n], rcond=None)
            abc[n] = sol

    residuos = {}
    for k in range(npat):
        rm, _ = _rho_modelo((abc[:, 0], abc[:, 1], abc[:, 2], Gn_fijo), ers[k])
        residuos[nombres[k]] = np.abs(rho[k] - rm)
    residuo_rms = np.sqrt(np.mean(np.array(list(residuos.values())) ** 2, axis=0))

    if verbose:
        n_reaj = int(np.sum(~sin_cambio))
        print(f"  [cuadrados minimos - reajuste (a,b,c) con Gn acondicionado] "
              f"reajustado linealmente en {n_reaj}/{nf} frecuencias.")

    return {
        'a': abc[:, 0], 'b': abc[:, 1], 'c': abc[:, 2], 'Gn': Gn_fijo,
        'nombres': nombres, 'residuos': residuos, 'residuo_rms': residuo_rms,
        'grados_libertad': 2 * npat - 8,
        'iteraciones': np.zeros(nf, dtype=int),
        'convergio': np.ones(nf, dtype=bool),
    }


def get_er_DUT_minimos_cuadrados(frecs, S11_medido, calibracion, verbose=False,
                                 estrategia_semilla='minimo_gn'):
    """
    Permitividad del DUT a partir de una calibracion hecha con
    `calibrar_minimos_cuadrados` (o `reajustar_abc_con_gn_fijo`).

    Con (a, b, c, Gn) ya ajustados, el S11 medido da directamente
    F = (a*rho + b)/(c*rho + 1), y er sale de resolver er + Gn*er^(5/2) = F,
    o sea el mismo polinomio en u = sqrt(er) que el metodo completo:
    Gn*u^5 + u^2 - F = 0, con la misma marcha en frecuencia para elegir la
    raiz fisica. La semilla de arranque es F misma (la solucion con Gn = 0).

    Con exactamente 4 patrones (corto, aire, patron 3, patron 4) el
    resultado coincide con `get_er_DUT_completo`.
    """
    frecs = np.asarray(frecs, dtype=float)
    rho = np.asarray(S11_medido, dtype=complex)
    a, b, c, Gn = (calibracion[k] for k in ('a', 'b', 'c', 'Gn'))
    F = (a * rho + b) / (c * rho + 1.0)
    return _marchar_en_frecuencia(frecs, Gn, -F, F, estrategia_semilla,
                                  verbose=verbose, etiqueta="minimos_cuadrados")


def validacion_cruzada_minimos_cuadrados(frecs, patrones, cfg_acond=None):
    """
    Chequeo de CONSISTENCIA de una calibracion redundante, "dejando uno
    afuera": para cada patron liquido se calibra con TODOS LOS DEMAS y se
    mide el patron que quedo afuera como si fuera una muestra desconocida.

    Si `cfg_acond` tiene `'habilitado': True`, aplica el mismo acondicionamiento
    sobre el `Gn` obtenido en cada sub-calibracion y reajusta linealmente/LM
    `(a, b, c)` antes de predecir la permitividad del patron excluido.

    Retorna
    -------
    dict {nombre: {'er_predicho', 'er_teorico', 'modelo', 'residuo_resto'}}
    """
    resultado = {}
    if len(patrones) < 5:
        return resultado
    acond_activo = bool(cfg_acond and cfg_acond.get('habilitado', False))
    for k, pt in enumerate(patrones):
        if pt.get('er') is None or pt.get('es_aire'):
            continue
        resto = [q for j, q in enumerate(patrones) if j != k]
        cal = calibrar_minimos_cuadrados(frecs, resto)
        if acond_activo:
            tipo_acond = cfg_acond.get('tipo', 'transicion_polinomica_cero')
            modo_acond = cfg_acond.get('modo', 'automatico')
            f1_acond = float(cfg_acond.get('f1_ghz', 0.0) or 0.0) * 1e9
            f2_acond = float(cfg_acond.get('f2_ghz', 0.5) or 0.5) * 1e9
            umbral_acond = float(cfg_acond.get('umbral_pct', 30.0) or 30.0)
            gn_ac, _ = acondicionar_gn(
                frecs, cal['Gn'], modo=modo_acond, tipo=tipo_acond,
                f1=f1_acond, f2=f2_acond, umbral_pct=umbral_acond)
            cal = reajustar_abc_con_gn_fijo(frecs, resto, gn_ac, calibracion_base=cal)
        er_pred = get_er_DUT_minimos_cuadrados(frecs, pt['S11'], cal)
        er_teo = np.asarray(pt['er'], dtype=complex)
        resultado[pt.get('nombre', f"patron {k + 1}")] = {
            'er_predicho': er_pred,
            'er_teorico': er_teo,
            'modelo': pt.get('modelo'),
            'residuo_resto': float(np.sqrt(np.mean(cal['residuo_rms'] ** 2))),
        }
    return resultado



def _diagnosticar_saltos(frecs, Er_out, salto_relativo_max=0.5):
    """
    Heuristica de diagnostico: si entre dos puntos de frecuencia
    consecutivos la curva resultante 'salta' mas de un `salto_relativo_max`
    (50% por defecto) respecto de su propia magnitud, es un sintoma tipico
    de haber elegido la raiz equivocada del polinomio de 5to orden en ese
    tramo (rama incorrecta). Se imprime un aviso con la frecuencia
    sospechosa. No corrige nada automaticamente: es para que el usuario
    revise esa zona (por ejemplo, cerca de una resonancia de la sonda o
    donde la medicion tenga mucho ruido).
    """
    orden = np.argsort(frecs)
    f_ord = frecs[orden]
    er_ord = Er_out[orden]
    if len(er_ord) < 3:
        return
    saltos = np.abs(np.diff(er_ord))
    escala = np.maximum(np.abs(er_ord[:-1]), np.abs(er_ord[1:]))
    escala[escala == 0] = 1.0
    salto_rel = saltos / escala

    sospechosos = np.where(salto_rel > salto_relativo_max)[0]
    for i in sospechosos:
        print(f"[aviso] posible salto de rama del polinomio entre "
              f"{f_ord[i]/1e9:.3f} GHz y {f_ord[i+1]/1e9:.3f} GHz "
              f"(cambio relativo {salto_rel[i]*100:.0f}%, revisar esa zona)")


# Alias por compatibilidad con el nombre usado en versiones anteriores del
# proyecto (la implementacion original con scipy.fsolve no funcionaba con
# numeros complejos: tiraba TypeError). Se deja el alias para no romper
# notebooks/scripts previos que la llamaban `get_er_setup`.
get_er_setup = get_er_DUT_completo


def error_absoluto(Er_medido, Er_teorico):
    """
    Error absoluto de la parte real e imaginaria de la permitividad
    medida respecto de la curva teorica, punto a punto:
      err_real = |Re(Er_medido) - Re(Er_teorico)|
      err_imag = |Im(Er_medido) - Im(Er_teorico)|

    Retorna
    -------
    (error_real, error_imag) : tupla de ndarrays float con el error absoluto
    (adimensional, en las mismas unidades de la permitividad).
    """
    Er_medido = np.asarray(Er_medido, dtype=complex)
    Er_teorico = np.asarray(Er_teorico, dtype=complex)

    err_real = np.abs(np.real(Er_medido) - np.real(Er_teorico))
    err_imag = np.abs(np.imag(Er_medido) - np.imag(Er_teorico))

    return err_real, err_imag


def error_relativo_porcentual(Er_medido, Er_teorico):
    """
    Error relativo porcentual de la parte real e imaginaria de la
    permitividad medida respecto de la curva teorica, punto a punto.

    Cuidado con el denominador: desde que existen los modelos de
    permitividad ESTATICA de `Patrones.py` (acetona, ciclohexano, fluido
    de silicona -- liquidos para los que NPL no publica un ajuste de
    relajacion), la curva teorica puede tener er'' identicamente 0. En
    ese caso el error relativo de la parte imaginaria no esta definido:
    antes esto daba `inf`/`-inf` (o `nan` donde medido y teorico eran
    ambos 0) y despues se propagaba a las tablas del informe como
    "inf%", que parece un error de calculo cuando en realidad la
    pregunta misma no tiene sentido.

    Ahora, donde el denominador es 0, se devuelve `np.nan` de forma
    explicita y sin emitir el RuntimeWarning de division por cero de
    numpy. Los consumidores (`analisis_permitividad.calcular_error`,
    que promedia con nanmean/nanmax, y `reporte_pdf`, que imprime "n/a")
    ya saben interpretar ese NaN.

    Retorna
    -------
    (error_real, error_imag) en %, mismos largos que las entradas. Un
    NaN en `error_imag` significa "no aplica: la referencia no modela
    perdidas", no "fallo el calculo".
    """
    Er_medido = np.asarray(Er_medido, dtype=complex)
    Er_teorico = np.asarray(Er_teorico, dtype=complex)

    ref_real = np.real(Er_teorico)
    ref_imag = np.imag(Er_teorico)

    # `where=` evita el warning de division por cero y deja el valor de
    # salida en lo que haya inicializado `out` (NaN) en esas posiciones.
    err_real = np.full(ref_real.shape, np.nan, dtype=float)
    np.divide(100 * (np.real(Er_medido) - ref_real), ref_real,
              out=err_real, where=(ref_real != 0))

    err_imag = np.full(ref_imag.shape, np.nan, dtype=float)
    np.divide(100 * (np.imag(Er_medido) - ref_imag), ref_imag,
              out=err_imag, where=(ref_imag != 0))

    return err_real, err_imag


# ===========================================================================
# ACONDICIONAMIENTO DE Gn (EN TESTEO / DESARROLLO)
# ===========================================================================
# Herramientas para acondicionar la conductancia normalizada Gn(f) a bajas
# frecuencias. Diseñado para mitigar divergencias o ruido de medicion sin
# alterar la zona de alta frecuencia donde el modelo de radiacion es critico.
#
# Metodos disponibles:
#   - 'cero_clamping'             : anula a 0 por debajo de una frecuencia.
#   - 'transicion_polinomica_cero': transicion cuadratica suave (f^2) entre
#                                   f1 (donde Gn=0) y f2 (Gn original en f2),
#                                   garantizando monotonicidad y sin sobrepicos.
#   - 'tapering'                  : desvanecimiento suave a 0 multiplicando
#                                   por una ventana coseno alzada en [f1, f2].
# ===========================================================================

def _detectar_frecuencia_disparo(frecs, gn, umbral_pct=30.0):
    """
    Recorre el barrido en frecuencia de forma DESCENDENTE (desde la frecuencia
    mas alta hacia la mas baja) y detecta el primer salto relativo |Delta(Gn)| / |Gn|
    que supere `umbral_pct`.

    Retorna
    -------
    (f_disparo, idx_disparo) o (None, None) si no se detecto ninguna variacion
    superior al umbral.
    """
    frecs = np.asarray(frecs, dtype=float)
    gn = np.asarray(gn, dtype=complex)
    umbral_frac = float(umbral_pct) / 100.0

    for i in range(len(frecs) - 1, 0, -1):
        denominador = float(np.abs(gn[i]))
        if denominador == 0.0:
            denominador = 1e-15  # Resguardo unico si el punto de referencia vale exactamente cero

        salto_relativo = float(np.abs(gn[i - 1] - gn[i])) / denominador
        if salto_relativo > umbral_frac:
            return frecs[i], i

    return None, None


def _aplicar_cero_clamping(frecs, gn, f_corte):
    """
    Anula abruptamente Gn(f) a 0 para toda frecuencia f <= f_corte.
    """
    gn_mod = np.copy(gn)
    mascara_corte = frecs <= f_corte
    gn_mod[mascara_corte] = 0.0 + 0.0j
    return gn_mod


def _aplicar_transicion_polinomica_cero(frecs, gn, f1, f2):
    """
    Reemplaza Gn en el intervalo [f1, f2] por una ley de potencia cuadratica
    (polinomio de grado 2, proporcional a f^2), modelando el comportamiento
    asintotico de la radiacion de la apertura coaxial a baja frecuencia:
      - En f <= f1 : Gn = 0 (cero absoluto)
      - En f1 < f < f2 : Gn(f) = Gn(f2) * ((f - f1)/(f2 - f1))^2
        (crecimiento cuadratico monótono, derivada cero en f1, sin sobrepicos)
      - En f >= f2 : Gn original

    Garantiza monotonicidad estricta en el modulo (|Gn| <= |Gn(f2)|) y
    elimina cualquier sobrepico u oscilacion artificial debida a ruido.
    """
    frecs = np.asarray(frecs, dtype=float)
    gn_mod = np.copy(gn)

    if f2 <= f1:
        return gn_mod

    # 1. Zona f <= f1 (cero absoluto)
    gn_mod[frecs <= f1] = 0.0 + 0.0j

    # 2. Localizar punto de empalme f2
    idx2 = int(np.argmin(np.abs(frecs - f2)))
    val_f2 = gn[idx2]

    # 3. Transicion cuadratica suave (proporcional a f^2) en (f1, f2)
    mascara_transicion = (frecs > f1) & (frecs < f2)
    if np.any(mascara_transicion):
        delta_f = f2 - f1
        tau = (frecs[mascara_transicion] - f1) / delta_f  # Normalizado en (0, 1)
        gn_mod[mascara_transicion] = val_f2 * (tau ** 2)

    return gn_mod


def _aplicar_tapering(frecs, gn, f1, f2):
    """
    Aplica una ventana suave de desvanecimiento (coseno alzado / semi-Hann)
    sobre Gn(f) entre f1 y f2:
      - f <= f1     : w = 0 -> Gn = 0
      - f1 < f < f2 : w = 0.5 * (1 - cos(pi * (f - f1) / (f2 - f1)))
      - f >= f2     : w = 1 -> Gn original
    """
    frecs = np.asarray(frecs, dtype=float)
    gn_mod = np.copy(gn)

    if f2 <= f1:
        return gn_mod

    gn_mod[frecs <= f1] = 0.0 + 0.0j

    mascara_transicion = (frecs > f1) & (frecs < f2)
    if np.any(mascara_transicion):
        tau = (frecs[mascara_transicion] - f1) / (f2 - f1)
        w = 0.5 * (1.0 - np.cos(np.pi * tau))
        gn_mod[mascara_transicion] = gn[mascara_transicion] * w

    return gn_mod


def acondicionar_gn(frecs, gn, modo='automatico', tipo='transicion_polinomica_cero',
                     f1=None, f2=None, umbral_pct=30.0):
    """
    Acondiciona el vector de conductancia normalizada Gn(f) segun el metodo
    y modo seleccionados.

    Parametros
    ----------
    frecs : array_like float
        Vector de frecuencias en Hz.
    gn : array_like complex
        Vector complejo de Gn(f) obtenido con `calcular_Gn`.
    modo : {'automatico', 'manual'}, opcional (default 'automatico')
        - 'automatico': busca de forma descendente en frecuencia la primera
          discontinuidad |Delta(Gn)| / |Gn| > umbral_pct para determinar la
          frecuencia de corte/empalme. En metodos de transicion, toma
          f1 como la frecuencia minima del archivo (frecs[0]).
        - 'manual': utiliza las frecuencias explicitas pasadas en `f1` y `f2`.
    tipo : {'cero_clamping', 'transicion_polinomica_cero', 'tapering'}, opcional
        - 'cero_clamping': anula Gn = 0 por debajo de la frecuencia de corte.
        - 'transicion_polinomica_cero': sustituye el tramo [f1, f2] por una
          ley de potencia cuadratica (proporcional a f^2), suave y monotona.
        - 'tapering': modula suavemente el Gn existente mediante ventana de
          coseno alzado hacia 0 en [f1, f2].
    f1 : float, opcional
        Frecuencia inicial donde Gn=0 (Hz). En automatico se fija en la frecuencia minima del archivo.
    f2 : float, opcional
        Frecuencia final de empalme con el Gn original (Hz), o frecuencia de corte
        para 'cero_clamping'.
    umbral_pct : float, opcional (default 30.0)
        Porcentaje de variacion relativa para el detector automatico de disparo.

    Retorna
    -------
    gn_acondicionado : ndarray complex
        Vector de Gn acondicionado.
    info : dict
        Metadatos de la operacion ('tipo', 'modo', 'disparo_detectado',
        'f1_aplicada', 'f2_aplicada', 'f_corte_aplicada', 'aviso').
    """
    frecs = np.asarray(frecs, dtype=float)
    gn = np.asarray(gn, dtype=complex)

    tipos_validos = ('cero_clamping', 'transicion_polinomica_cero', 'tapering')
    if tipo not in tipos_validos:
        raise ValueError(f"Tipo '{tipo}' no valido. Debe ser uno de {tipos_validos}.")

    modos_validos = ('automatico', 'manual')
    if modo not in modos_validos:
        raise ValueError(f"Modo '{modo}' no valido. Debe ser uno de {modos_validos}.")

    info = {
        'tipo': tipo,
        'modo': modo,
        'disparo_detectado': None,
        'f1_aplicada': None,
        'f2_aplicada': None,
        'f_corte_aplicada': None,
        'aviso': None,
    }

    if modo == 'automatico':
        f_disp, idx_disp = _detectar_frecuencia_disparo(frecs, gn, umbral_pct=umbral_pct)
        if f_disp is None:
            info['disparo_detectado'] = False
            info['aviso'] = (
                f"[aviso] No se detecto ninguna variacion relativa mayor al "
                f"{umbral_pct:.1f}% en Gn. Se mantuvo la curva original intacta."
            )
            print(info['aviso'])
            return np.copy(gn), info

        info['disparo_detectado'] = True

        if tipo == 'cero_clamping':
            f_corte = f_disp
            info['f_corte_aplicada'] = f_corte
            gn_out = _aplicar_cero_clamping(frecs, gn, f_corte)
        else:
            # En automatico: f1 se fija en la frecuencia minima del archivo (frecs[0])
            # y f2 en la frecuencia de disparo detectada
            f1_eff = float(frecs[0])
            f2_eff = f_disp
            info['f1_aplicada'] = f1_eff
            info['f2_aplicada'] = f2_eff
            if tipo == 'transicion_polinomica_cero':
                gn_out = _aplicar_transicion_polinomica_cero(frecs, gn, f1_eff, f2_eff)
            else:
                gn_out = _aplicar_tapering(frecs, gn, f1_eff, f2_eff)

        return gn_out, info

    else:  # modo == 'manual'
        info['disparo_detectado'] = True
        if tipo == 'cero_clamping':
            f_corte = f2 if f2 is not None else (f1 if f1 is not None else frecs[0])
            info['f_corte_aplicada'] = float(f_corte)
            gn_out = _aplicar_cero_clamping(frecs, gn, f_corte)
        else:
            f1_eff = 0.0 if f1 is None else float(f1)
            f2_eff = float(frecs[-1]) if f2 is None else float(f2)

            if f2_eff <= f1_eff:
                info['aviso'] = (
                    f"[aviso] f2 ({f2_eff/1e9:.3f} GHz) debe ser mayor que f1 "
                    f"({f1_eff/1e9:.3f} GHz). Se mantuvo la curva original intacta."
                )
                print(info['aviso'])
                return np.copy(gn), info

            info['f1_aplicada'] = f1_eff
            info['f2_aplicada'] = f2_eff
            if tipo == 'transicion_polinomica_cero':
                gn_out = _aplicar_transicion_polinomica_cero(frecs, gn, f1_eff, f2_eff)
            else:
                gn_out = _aplicar_tapering(frecs, gn, f1_eff, f2_eff)

        return gn_out, info