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

  * `calibrar_minimos_cuadrados` + `get_er_DUT_minimos_cuadrados`
                            -> Calibracion REDUNDANTE (N >= 4 patrones: corto,
                               aire y cualquier cantidad de liquidos), como
                               recomienda Kaatze (2007). Mismo modelo que el
                               metodo completo, pero los 4 parametros de la
                               sonda por frecuencia se AJUSTAN por cuadrados
                               minimos en el espacio de S11 en vez de
                               despejarse de exactamente 4 patrones. Con 4
                               patrones reproduce el metodo completo. Ademas,
                               `validacion_cruzada_minimos_cuadrados` da un
                               chequeo de consistencia entre patrones.

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

    # FILTRO DE RAMA (corregido): antes se elevaban al cuadrado las 5 raices
    # y se elegia la mas cercana a la semilla. El problema es que el
    # polinomio tiene DOS raices casi iguales cerca de la fisica: u1 ~
    # +sqrt(er) y u2 ~ -sqrt(er). Con Gn = 0 darian exactamente el mismo er,
    # pero el termino Gn*u^5 cambia de signo entre las dos, asi que sus
    # cuadrados quedan separados en ~2*Gn*er^2.5 -- justo alrededor del
    # valor que da el metodo simplificado, que es la semilla. La eleccion
    # por cercania quedaba librada al azar, con errores sistematicos de
    # hasta ~0.1-0.5% (mas grandes cuanto mas irradia la sonda). Solo u1
    # es fisica: el modelo Y = jwC0*er + G0*er^(5/2) define er^(5/2) con la
    # raiz PRINCIPAL, que para un medio pasivo (Re(er) > 0) tiene Re(u) > 0.
    principales = raices_u[np.real(raices_u) > 0]
    if principales.size == 0:
        principales = raices_u
    raices_er = principales ** 2

    candidatas = raices_er[np.real(raices_er) > 0]
    if candidatas.size == 0:
        candidatas = raices_er

    idx = np.argmin(np.abs(candidatas - semilla))
    return candidatas[idx]


def get_er_DUT_completo(frecs, S11_medido, S11_patron3, S11_aire,
                         S11_patron4, S11_corto, Er_patron3, Er_patron4,
                         Er_aire=ER_AIRE, verbose=False,
                         estrategia_semilla='minimo_gn'):
    """
    Calcula la permitividad relativa compleja del DUT a partir del modelo
    de admitancia "completo" (con conductancia de radiacion normalizada Gn),
    usando 4 patrones de calibracion: cortocircuito, aire, "patron 3" y
    "patron 4" (tradicionalmente agua destilada y alcohol isopropilico,
    pero puede ser cualquier par de liquidos con permitividad conocida).

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

    # Conductancia normalizada Gn, a partir del patron 4. Se delega a
    # `calcular_Gn` (definida arriba) para no duplicar la formula: esa
    # misma funcion queda disponible aparte como diagnostico de
    # calibracion (ver su docstring y `analisis_permitividad.graficar_Gn`).
    Gn = calcular_Gn(frecs, S11_patron3, S11_aire, S11_patron4, S11_corto,
                      Er_patron3, Er_patron4, Er_aire=Er_aire)

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

    Metodo
    ------
    Para cada frecuencia se buscan (a, b, c, Gn) que minimizan

        sum_k  peso_k * |rho_medido_k - rho_modelo_k(a, b, c, Gn)|^2

    es decir, el error se mide en el ESPACIO DE S11, que es donde esta el
    ruido del VNA (y no en el espacio de permitividad, donde un residuo
    algebraico pesaria al agua ~50000 veces mas que al aire por el
    termino er^(5/2)).

    Arranque: primero se ajusta la bilineal con Gn = 0, que es un problema
    LINEAL en (a, b, c) y se resuelve exacto; despues Levenberg-Marquardt
    sobre los 4 parametros. Con Gn chico -- lo habitual -- converge en
    pocas iteraciones.

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


def get_er_DUT_minimos_cuadrados(frecs, S11_medido, calibracion, verbose=False,
                                 estrategia_semilla='minimo_gn'):
    """
    Permitividad del DUT a partir de una calibracion hecha con
    `calibrar_minimos_cuadrados`.

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


def validacion_cruzada_minimos_cuadrados(frecs, patrones):
    """
    Chequeo de CONSISTENCIA de una calibracion redundante, "dejando uno
    afuera": para cada patron liquido se calibra con TODOS LOS DEMAS y se
    calculan dos cosas.

      'err_medio_pct' / 'err_max_pct' : cuanto se equivoca esa calibracion
          al medir el patron que quedo afuera, como si fuera una muestra
          desconocida (en % de |er|). Es la version interpretable de los
          residuos: en vez de un numero en unidades de S11, un error en
          permitividad.
      'residuo_resto' : residuo rms (en S11) del ajuste SIN ese patron.
          Si el residuo global de la calibracion completa es alto pero al
          sacar un patron el de los demas baja al nivel de ruido, ese
          patron es sospechoso.

    Es el tipo de "consistency check" que sugiere Kaatze (2007, pag. 974)
    para el DMSO. Lo que PUEDE y lo que NO PUEDE hacer, verificado por
    simulacion (metanol contaminado con 2% de agua, 6 patrones):
      - DETECTAR una inconsistencia funciona bien: el residuo global sube
        ~2.5 veces respecto de una calibracion limpia.
      - SENALAR AL CULPABLE funciona solo a medias: sacar el metanol
        normaliza a los demas, pero sacar el agua tambien, porque sin el
        unico patron de alta permitividad el modelo tiene libertad para
        absorber la inconsistencia. Con pocos patrones de sobra el
        diagnostico acota los sospechosos, no los identifica. Mas
        redundancia (mas patrones) lo mejora.
      - El error "dejando afuera" el AGUA es naturalmente alto aun con
        todo limpio: sin ella no queda ningun patron de alta
        permitividad, y medir agua pasa a ser una extrapolacion. No es
        una senal de problema.

    Solo se evalua cada patron liquido (no el corto ni el aire, marcado
    con 'es_aire'), y solo si al sacarlo quedan al menos 4 patrones.

    Retorna
    -------
    dict {nombre: {'er_predicho', 'er_teorico', 'err_medio_pct',
                   'err_max_pct', 'residuo_resto'}}
    """
    resultado = {}
    if len(patrones) < 5:
        return resultado
    for k, pt in enumerate(patrones):
        if pt.get('er') is None or pt.get('es_aire'):
            continue
        resto = [q for j, q in enumerate(patrones) if j != k]
        cal = calibrar_minimos_cuadrados(frecs, resto)
        er_pred = get_er_DUT_minimos_cuadrados(frecs, pt['S11'], cal)
        er_teo = np.asarray(pt['er'], dtype=complex)
        rel = np.abs(er_pred - er_teo) / np.abs(er_teo)
        resultado[pt.get('nombre', f"patron {k + 1}")] = {
            'er_predicho': er_pred, 'er_teorico': er_teo,
            'err_medio_pct': float(100 * np.mean(rel)),
            'err_max_pct': float(100 * np.max(rel)),
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