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
                               de calibracion: corto, aire (abierto) y agua.
                               Es una formula algebraica cerrada, valida
                               sobre todo a frecuencias mas bajas / materiales
                               de bajas perdidas.

  * `get_er_DUT_completo`  -> Metodo "completo" (con conductancia normalizada
                               Gn != 0), usa 4 patrones de calibracion: corto,
                               aire, agua y alcohol isopropilico. Resuelve,
                               para cada frecuencia, la ecuacion polinomica de
                               5to orden en raiz(er) que aparece en el modelo
                               de admitancia completo. Es mas preciso en un
                               rango de frecuencias mas amplio.

Ambos metodos devuelven er = er' - j*er'' (parte imaginaria positiva para
perdidas) evaluado en las mismas frecuencias que los datos de entrada.
"""
import numpy as np

from Patrones import get_er_agua, get_er_pat_alc_isopropilico, ER_AIRE


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
# Metodo simplificado (3 patrones: corto, aire, agua) -- Gn = 0
# ---------------------------------------------------------------------------
def get_er_DUTm(frecs, S11_medido, S11_agua, S11_aire, S11_corto,
                 T_agua=25.0, Er_aire=ER_AIRE):
    """
    Calcula la permitividad relativa compleja del DUT a partir del modelo
    simplificado (conductancia normalizada Gn = 0), usando como patrones de
    calibracion el cortocircuito (referencia Y->inf), el aire/circuito
    abierto (er ~= 1) y el agua destilada (er segun modelo de Debye/Liebe).

    Es una formula algebraica cerrada (no iterativa), equivalente a la
    ecuacion (18) de Higa/Cismondi/Grass (2016) y de Henze et al. (2024).

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz (deben coincidir con las de todos los S11).
    S11_medido : array_like complejo
        S11 medido del material bajo ensayo (DUT).
    S11_agua, S11_aire, S11_corto : array_like complejo
        S11 medidos de los patrones de calibracion.
    T_agua : float
        Temperatura del agua destilada durante la calibracion (°C).
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).

    Retorna
    -------
    ndarray complejo: er'(f) - j*er''(f) del DUT.
    """
    frecs = np.asarray(frecs, dtype=float)
    S11_medido = np.asarray(S11_medido, dtype=complex)
    S11_agua = np.asarray(S11_agua, dtype=complex)
    S11_aire = np.asarray(S11_aire, dtype=complex)
    S11_corto = np.asarray(S11_corto, dtype=complex)

    Er_agua = get_er_agua(frecs, T_agua)

    # Ecuacion (18) de Higa/Cismondi/Grass (2016): coeficientes de la
    # combinacion lineal de las permitividades conocidas del agua (Er_agua)
    # y del aire (Er_aire), en funcion de los S11 medidos.
    coef_agua = ((S11_medido - S11_aire) * (S11_corto - S11_agua)) / \
                ((S11_medido - S11_corto) * (S11_agua - S11_aire))

    coef_aire = ((S11_medido - S11_agua) * (S11_aire - S11_corto)) / \
                ((S11_medido - S11_corto) * (S11_agua - S11_aire))

    return -coef_agua * Er_agua - coef_aire * Er_aire


# ---------------------------------------------------------------------------
# Metodo completo (4 patrones: corto, aire, agua, alcohol isopropilico)
# ---------------------------------------------------------------------------
def calcular_Gn(frecs, S11_agua, S11_aire, S11_isopropilico, S11_corto,
                 T_agua=25.0, T_isoprop=25.0, Er_aire=ER_AIRE):
    """
    Calcula la conductancia normalizada Gn(f) del modelo de admitancia
    completo (ecuacion (17) de Higa/Cismondi/Grass 2016 y de Henze et al.
    2024/ARGENCON), a partir UNICAMENTE de los 4 patrones de calibracion
    (corto, aire, agua, alcohol isopropilico). Notar que Gn NO depende del
    S11 del DUT: es una propiedad de la sonda/calibracion, la misma para
    todos los materiales que se analicen con esa calibracion.

    `get_er_DUT_completo` llama a esta funcion internamente para armar el
    polinomio de 5to orden, pero se expone tambien aca aparte para poder
    usarla como DIAGNOSTICO de la calibracion (ver
    `analisis_permitividad.graficar_Gn`): como Gn esta relacionada con
    G0/(j*w*C0) -- una propiedad fisica de la sonda que varia en forma
    continua con la frecuencia -- la curva Gn(f) resultante tiene que
    verse suave. Un salto brusco o un pico aislado suele ser sintoma de
    un problema con la medicion de alguno de los 4 patrones, tipicamente
    el 4to (alcohol isopropilico), que es el UNICO que entra en el
    calculo de Gn y en ningun otro lado del metodo simplificado -- asi
    que un problema ahi puede pasar desapercibido si solo se mira el
    chequeo de calibracion del agua.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    S11_agua, S11_aire, S11_isopropilico, S11_corto : array_like complejo
        S11 medidos de los 4 patrones de calibracion.
    T_agua, T_isoprop : float
        Temperatura del agua y del alcohol isopropilico durante la
        calibracion (°C).
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).

    Retorna
    -------
    ndarray complejo: Gn(f), mismo largo que `frecs`.
    """
    frecs = np.asarray(frecs, dtype=float)
    r_agua = np.asarray(S11_agua, dtype=complex)
    r_aire = np.asarray(S11_aire, dtype=complex)
    r_alcohol = np.asarray(S11_isopropilico, dtype=complex)
    r_corto = np.asarray(S11_corto, dtype=complex)

    Er_agua = get_er_agua(frecs, T_agua)
    Er_alcohol = get_er_pat_alc_isopropilico(frecs, T_isoprop)

    # Gn = -(D41*D32*e4 + D42*D13*e3 + D43*D21*e2) /
    #       (D41*D32*e4^2.5 + D42*D13*e3^2.5 + D43*D21*e2^2.5)
    # con patron1=corto, patron2=aire (e2=Er_aire), patron3=agua (e3=Er_agua),
    # patron4=alcohol isopropilico (e4=Er_alcohol).
    d41 = r_alcohol - r_corto
    d32 = r_agua - r_aire
    d42 = r_alcohol - r_aire
    d13 = r_corto - r_agua
    d43 = r_alcohol - r_agua
    d21 = r_aire - r_corto

    num_Gn = (d41 * d32 * Er_alcohol
              + d42 * d13 * Er_agua
              + d43 * d21 * Er_aire)
    den_Gn = (d41 * d32 * (Er_alcohol ** 2.5)
              + d42 * d13 * (Er_agua ** 2.5)
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


def get_er_DUT_completo(frecs, S11_medido, S11_agua, S11_aire,
                         S11_isopropilico, S11_corto, T_agua=25.0,
                         T_isoprop=25.0, Er_aire=ER_AIRE, verbose=False):
    """
    Calcula la permitividad relativa compleja del DUT a partir del modelo
    de admitancia "completo" (con conductancia de radiacion normalizada Gn),
    usando 4 patrones de calibracion: cortocircuito, aire, agua destilada y
    alcohol isopropilico.

    Para cada frecuencia se arma la ecuacion polinomica de 5to orden en
    u = sqrt(er):

        Gn*u^5 + u^2 + c0 = 0

    y se resuelve exactamente con `numpy.roots` (en lugar de un solver
    iterativo real como `scipy.optimize.fsolve`, que no admite residuos
    complejos). Entre las 5 raices se elige la fisicamente correcta
    "marchando" en frecuencia, de mayor a menor: se arranca en la
    frecuencia mas alta del barrido (donde el metodo simplificado da la
    mejor semilla, porque ahi Gn es mas chico) y en cada paso siguiente se
    usa como semilla la solucion aceptada en el punto anterior, aprovechando
    que la permitividad de un material real varia en forma continua con la
    frecuencia. Con `verbose=True` se imprime un aviso si en algun tramo la
    curva resultante muestra un salto brusco (posible eleccion de rama
    incorrecta), para que se pueda revisar esa zona.

    Parametros
    ----------
    frecs : array_like
        Frecuencias en Hz.
    S11_medido : array_like complejo
        S11 medido del material bajo ensayo (DUT).
    S11_agua, S11_aire, S11_isopropilico, S11_corto : array_like complejo
        S11 medidos de los 4 patrones de calibracion.
    T_agua : float
        Temperatura del agua destilada durante la calibracion (°C).
    T_isoprop : float
        Temperatura del alcohol isopropilico (4to patron) durante la
        calibracion (°C). Antes esta funcion llamaba a
        `get_er_pat_alc_isopropilico(frecs)` sin pasar temperatura, con lo
        cual siempre asumia el default de Patrones.py (25°C) sin importar
        la temperatura real de la calibracion. Como Er_alcohol entra en el
        calculo de Gn (la conductancia normalizada), una temperatura mal
        puesta ahi afecta el resultado de TODOS los materiales analizados
        con el metodo completo, no solo del alcohol isopropilico.
    Er_aire : float
        Permitividad relativa del aire (por defecto 1.0006).

    Retorna
    -------
    ndarray complejo: er'(f) - j*er''(f) del DUT.
    """
    frecs = np.asarray(frecs, dtype=float)
    S11_medido = np.asarray(S11_medido, dtype=complex)
    r_agua = np.asarray(S11_agua, dtype=complex)
    r_aire = np.asarray(S11_aire, dtype=complex)
    r_corto = np.asarray(S11_corto, dtype=complex)

    Er_agua = get_er_agua(frecs, T_agua)

    # Conductancia normalizada Gn, a partir del 4to patron (alcohol
    # isopropilico). Se delega a `calcular_Gn` (definida arriba) para no
    # duplicar la formula: esa misma funcion queda disponible aparte como
    # diagnostico de calibracion (ver su docstring y
    # `analisis_permitividad.graficar_Gn`).
    Gn = calcular_Gn(frecs, S11_agua, S11_aire, S11_isopropilico, S11_corto,
                      T_agua=T_agua, T_isoprop=T_isoprop, Er_aire=Er_aire)

    # --- Coeficientes X, Z de la ecuacion principal (idem metodo simple) ---
    d32 = r_agua - r_aire
    d13 = r_corto - r_agua
    d21 = r_aire - r_corto

    dm2 = S11_medido - r_aire
    dm1 = S11_medido - r_corto
    dm3 = S11_medido - r_agua

    X = (dm2 * d13) / (dm1 * d32)
    Z = (dm3 * d21) / (dm1 * d32)

    # Semilla inicial (solo para el primer punto de frecuencia): prediccion
    # del metodo simplificado (Gn=0).
    semillas = get_er_DUTm(frecs, S11_medido, S11_agua, S11_aire, S11_corto,
                            T_agua=T_agua, Er_aire=Er_aire)

    # El polinomio de 5to orden en sqrt(er) tiene 5 raices, y cuando Gn no
    # es despreciable varias pueden quedar cerca entre si. Usar siempre la
    # semilla del metodo simplificado para elegir la raiz es poco confiable
    # justo en el regimen donde Gn importa (que es donde el simplificado
    # mas se degrada). En cambio, se "marcha" en frecuencia: la permitividad
    # de un material real varia en forma continua con la frecuencia, asi
    # que a partir del segundo punto se usa como semilla la solucion ya
    # aceptada en el punto anterior.
    #
    # El punto de partida importa: Gn = G0/(j*w*C0) es mas chico (en modulo)
    # cuanto mayor es la frecuencia, asi que ahi el metodo simplificado
    # (que asume Gn=0) da la mejor semilla posible. Por eso se arranca la
    # marcha en la frecuencia MAS ALTA del barrido y se recorre en orden
    # DESCENDENTE hacia la frecuencia mas baja.
    orden = np.argsort(frecs)[::-1]

    Er_out = np.zeros_like(frecs, dtype=complex)
    semilla_actual = None
    for n in orden:
        coef_X = Er_agua[n] + Gn[n] * (Er_agua[n] ** 2.5)
        coef_1 = 1.0 + Gn[n]
        c0 = X[n] * coef_X + Z[n] * coef_1

        # Gn*u^5 + 0*u^4 + 0*u^3 + 1*u^2 + 0*u + c0 = 0
        coeficientes = [Gn[n], 0, 0, 1, 0, c0]

        if semilla_actual is None:
            semilla_actual = semillas[n]

        Er_out[n] = _resolver_raiz_fisica(coeficientes, semilla_actual)
        semilla_actual = Er_out[n]

    if verbose:
        _diagnosticar_saltos(frecs, Er_out)

    return Er_out


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

    Retorna
    -------
    (error_real, error_imag) en % , mismos largos que las entradas.
    """
    Er_medido = np.asarray(Er_medido, dtype=complex)
    Er_teorico = np.asarray(Er_teorico, dtype=complex)

    err_real = 100 * (np.real(Er_medido) - np.real(Er_teorico)) / np.real(Er_teorico)
    err_imag = 100 * (np.imag(Er_medido) - np.imag(Er_teorico)) / np.imag(Er_teorico)
    return err_real, err_imag
