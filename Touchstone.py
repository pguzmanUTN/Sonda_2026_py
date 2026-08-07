"""
Touchstone.py
-------------
Lectura y visualizacion de archivos Touchstone (.s1p) exportados por el VNA.

La lectura se apoya en `scikit-rf` (paquete `skrf`), que interpreta
correctamente el formato Touchstone estandar (lineas de comentario '!',
linea de opciones '#', y datos en Hz/GHz, modulo-fase o real-imaginario)
sin depender de la cantidad exacta de lineas de encabezado que use cada
VNA en particular. Esto reemplaza al parseo manual linea a linea, que
es fragil frente a archivos con distinto numero de comentarios.

Requiere: pip install scikit-rf --break-system-packages
"""
import numpy as np
import skrf as rf


def leer_s1p(path):
    """
    Lee un archivo Touchstone .s1p y devuelve un diccionario compatible con
    las funciones de conversion a permitividad de `funciones.py`.

    Parametros
    ----------
    path : str
        Ruta al archivo .s1p

    Retorna
    -------
    dict con las claves:
        'Frec'    : ndarray de frecuencias en Hz (float)
        'Complex' : ndarray de S11 complejo
        'Network' : objeto skrf.Network original (por si se necesita, p.ej.
                    para graficar con las utilidades propias de skrf)
        'Nombre'  : nombre de la red (toma el nombre de archivo sin extension)
    """
    ntw = rf.Network(path)
    return {
        'Frec': np.asarray(ntw.f, dtype=float),
        'Complex': np.asarray(ntw.s[:, 0, 0], dtype=complex),
        'Network': ntw,
        'Nombre': ntw.name,
    }


def resamplear(frecs_objetivo, datos):
    """
    Interpola (parte real e imaginaria por separado) el S11 de `datos` a las
    frecuencias de `frecs_objetivo`. Util cuando dos mediciones no comparten
    exactamente la misma grilla de frecuencias (distinto barrido del VNA).

    Parametros
    ----------
    frecs_objetivo : array_like
        Frecuencias (Hz) a las que se quiere remuestrear.
    datos : dict
        Diccionario con claves 'Frec' y 'Complex' (formato de `leer_s1p`).

    Retorna
    -------
    dict con 'Frec' = frecs_objetivo y 'Complex' interpolado.
    """
    frecs_objetivo = np.asarray(frecs_objetivo, dtype=float)
    frec_datos = np.asarray(datos['Frec'], dtype=float)
    s11_datos = np.asarray(datos['Complex'], dtype=complex)

    fuera_de_rango = (frecs_objetivo < frec_datos.min()) | (frecs_objetivo > frec_datos.max())
    if np.any(fuera_de_rango):
        n_fuera = int(np.sum(fuera_de_rango))
        print(f"  [aviso] {n_fuera} punto(s) de la grilla objetivo caen fuera del "
              f"rango medido ({frec_datos.min()/1e9:.3f}-{frec_datos.max()/1e9:.3f} GHz) "
              f"de '{datos.get('Nombre', '?')}'; se extrapola con el valor del borde "
              f"(puede no ser preciso).")

    parte_real = np.interp(frecs_objetivo, frec_datos, np.real(s11_datos))
    parte_imag = np.interp(frecs_objetivo, frec_datos, np.imag(s11_datos))

    return {
        'Frec': frecs_objetivo,
        'Complex': parte_real + 1j * parte_imag,
    }


def mismas_frecuencias(*dicts, tol_hz=1.0):
    """
    Chequea que todos los diccionarios (formato `leer_s1p`) compartan la
    misma grilla de frecuencias, dentro de una tolerancia en Hz. Devuelve
    True/False. Util para decidir si hace falta resamplear antes de operar
    con varios archivos juntos.
    """
    frecs_ref = np.asarray(dicts[0]['Frec'], dtype=float)
    for d in dicts[1:]:
        frecs = np.asarray(d['Frec'], dtype=float)
        if len(frecs) != len(frecs_ref) or np.any(np.abs(frecs - frecs_ref) > tol_hz):
            return False
    return True


def graficar_s11_db(datos_dict, ax=None, titulo="|S11| vs Frecuencia"):
    """
    Grafica |S11| en dB vs frecuencia para uno o varios patrones.

    Parametros
    ----------
    datos_dict : dict[str, dict]
        Diccionario {nombre: datos} donde cada `datos` tiene el formato de
        `leer_s1p` (claves 'Frec' y 'Complex').
    ax : matplotlib.axes.Axes, opcional
        Eje donde graficar. Si no se pasa, se crea una figura nueva.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    for nombre, datos in datos_dict.items():
        mag_db = 20 * np.log10(np.abs(datos['Complex']))
        ax.plot(datos['Frec'] / 1e9, mag_db, label=nombre, linewidth=1.6)

    ax.set_xlabel("Frecuencia (GHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title(titulo)
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    return ax


def _rango_fase_amigable(fase_min, fase_max):
    """
    Devuelve (limite_inferior, limite_superior) "lindos" (en grados) para
    el eje de fase, calculados por separado a partir del minimo y el
    maximo realmente observados en la fase YA DESENVUELTA (ver
    `graficar_s11_mag_fase`, que aplica `np.unwrap` antes de llamar a esta
    funcion -- sin desenvolver, un patron con fase cerca de +-180, tipico
    del cortocircuito, puede aparentar saltar de -179 a +179 de un punto
    a otro, aunque fisicamente es un cambio chico).

    Version anterior: redondeaba cada extremo "hacia afuera" entendiendo
    afuera como "lejos de 0". Eso esta bien cuando los dos extremos tienen
    signo distinto (agua real: -150 a +5), pero es un bug cuando los DOS
    extremos tienen el MISMO signo (cortocircuito real, casi siempre
    negativa, p.ej. -190 a -178): ahi "lejos de 0" empuja el limite
    superior en la direccion EQUIVOCADA (mas negativo en vez de menos
    negativo), y la curva terminaba pegada contra el borde del grafico.

    Ahora se buscan candidatos "lindos" (positivos y negativos) y se
    elige, para el limite inferior, el mayor candidato que sigue siendo
    <= el objetivo; para el superior, el menor candidato que sigue siendo
    >= el objetivo. Esto da el resultado correcto sin importar de que
    lado de 0 caiga cada extremo. Tampoco hay techo fijo en 180: si la
    fase desenvuelta se va mas alla (mucho retardo electrico en un
    barrido de varias decadas), se sigue escalonando en pasos de 90° en
    vez de cortar la curva.
    """
    pasos_base = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90,
                  120, 150, 180, 210, 240, 270, 300, 330, 360]
    candidatos = sorted({0.0} | {float(p) for p in pasos_base} | {-float(p) for p in pasos_base})

    margen = max(0.15 * (fase_max - fase_min), 2.0)
    objetivo_inf = fase_min - margen
    objetivo_sup = fase_max + margen

    def _piso(valor):
        """Mayor candidato <= valor (para el limite inferior)."""
        for c in reversed(candidatos):
            if c <= valor:
                return c
        paso = candidatos[0]
        while paso > valor:
            paso -= 90.0
        return paso

    def _techo(valor):
        """Menor candidato >= valor (para el limite superior)."""
        for c in candidatos:
            if c >= valor:
                return c
        paso = candidatos[-1]
        while paso < valor:
            paso += 90.0
        return paso

    return _piso(objetivo_inf), _techo(objetivo_sup)


def graficar_s11_mag_fase(datos_dict, titulo="S11: modulo y fase", archivo_salida=None,
                           log_x=True):
    """
    Grafica |S11| en dB y fase (grados) vs frecuencia, en dos subplots
    apilados con eje de frecuencias compartido -- el mismo estilo que
    `analisis_permitividad.graficar_comparacion` usa para er'/er''.

    Parametros
    ----------
    datos_dict : dict[str, dict]
        {nombre: datos}, cada `datos` con el formato de `leer_s1p` (o de
        `resamplear`): claves 'Frec' y 'Complex'. Admite una o varias
        curvas (una por entrada del dict), igual que `graficar_s11_db`.
    titulo : str
    archivo_salida : str, opcional
        Si se pasa, guarda la figura ahi (dpi=150) y la cierra (para no
        acumular figuras abiertas en una corrida con muchos materiales).
        Si se omite, devuelve la figura abierta sin guardar ni cerrar
        (para seguir editandola/mostrandola desde otro script, como
        `Touchstone.py` corrido standalone).
    log_x : bool
        Eje de frecuencias en escala logaritmica (default True) -- util
        para barridos de varias decadas (p.ej. 1 MHz a 2 GHz). Cae a
        lineal automaticamente si alguna frecuencia es <= 0 (el
        logaritmo no esta definido ahi).

    Retorna
    -------
    La ruta de `archivo_salida` si se guardo, o la Figure de matplotlib
    si no se guardo.
    """
    import matplotlib.pyplot as plt

    fig, (ax_mag, ax_fase) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    frecs_vistas = []
    fase_min = 0.0
    fase_max = 0.0
    for nombre, datos in datos_dict.items():
        frec = np.asarray(datos['Frec'], dtype=float)
        s11 = np.asarray(datos['Complex'], dtype=complex)
        frecs_vistas.append(frec)
        mag_db = 20 * np.log10(np.abs(s11))
        # np.angle() sola reporta la fase "envuelta" en (-180, 180]: un
        # patron con fase fisica cerca de ese borde (tipicamente el
        # cortocircuito, con fase ~180°) puede aparentar un salto de -179
        # a +179 de un punto a otro, cuando en realidad es un cambio
        # chico. np.unwrap corrige esos saltos artificiales de +-360°,
        # dejando una curva continua y un rango real para el auto-escalado
        # (ver _rango_fase_amigable).
        fase_deg = np.degrees(np.unwrap(np.angle(s11)))
        fase_min = min(fase_min, float(np.min(fase_deg)))
        fase_max = max(fase_max, float(np.max(fase_deg)))
        ax_mag.plot(frec / 1e9, mag_db, linewidth=1.6, label=nombre)
        ax_fase.plot(frec / 1e9, fase_deg, linewidth=1.6, label=nombre)

    ax_mag.set_ylabel("|S11| (dB)")
    ax_mag.set_title(titulo)
    ax_mag.minorticks_on()
    ax_mag.margins(y=0.15)
    ax_mag.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_mag.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_mag.legend()

    # Antes esto era siempre set_ylim(-180, 180): con la fase real de esta
    # sonda la curva quedaba aplastada contra el centro o, si la fase era
    # grande pero no simetrica (p.ej. +5 a -150), un rango +-simetrico
    # dejaba medio grafico vacio. Ahora se calculan el limite inferior y
    # superior por separado (ver _rango_fase_amigable), asi el eje se
    # ajusta a la forma real de la curva sea o no simetrica.
    limite_inferior, limite_superior = _rango_fase_amigable(fase_min, fase_max)
    ax_fase.set_xlabel("Frecuencia (GHz)")
    ax_fase.set_ylabel("Fase S11 (\u00b0)")
    ax_fase.set_ylim(limite_inferior, limite_superior)
    ax_fase.minorticks_on()
    ax_fase.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax_fase.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
    ax_fase.legend()

    frecs_concat = np.concatenate(frecs_vistas) if frecs_vistas else np.array([])
    if log_x and frecs_concat.size > 0 and np.all(frecs_concat > 0):
        ax_mag.set_xscale('log')
        ax_fase.set_xscale('log')  # sharex ya lo propaga, se deja explicito

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        plt.close(fig)
        return archivo_salida
    return fig


def graficar_smith(datos_dict, titulo="Diagrama de Smith - S11", archivo_salida=None):
    """
    Grafica S11 (una o varias curvas) sobre un diagrama de Smith, usando
    las utilidades de scikit-rf (`Network.plot_s_smith`) solo para dibujar
    la grilla de fondo -- cada curva despues se plotea a mano con
    matplotlib para tener control total sobre colores/leyenda/marcadores
    de inicio y fin del barrido. Es el mismo patron que ya se usa en
    `MATS.py` para explorar el .mat original.

    Parametros
    ----------
    datos_dict : dict[str, dict]
        {nombre: datos}, cada `datos` con el formato de `leer_s1p` (o de
        `resamplear`): claves 'Frec' y 'Complex'. La red de scikit-rf se
        arma aca mismo a partir de esos arrays (no hace falta que
        `datos` tenga la clave 'Network' de `leer_s1p`; por eso funciona
        igual con datos ya remuestreados con `resamplear`, que no la
        conservan).
    titulo : str
    archivo_salida : str, opcional
        Si se pasa, guarda la figura ahi (dpi=150) y la cierra. Si se
        omite, devuelve la figura abierta sin guardar ni cerrar.

    Retorna
    -------
    La ruta de `archivo_salida` si se guardo, o la Figure de matplotlib
    si no se guardo.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # La grilla de fondo del diagrama de Smith se dibuja una sola vez, a
    # partir de la primera curva (con color='none' la traza de esa red no
    # se ve; solo queda la grilla/etiquetas de scikit-rf).
    primer_datos = next(iter(datos_dict.values()))
    frec_ref = np.asarray(primer_datos['Frec'], dtype=float)
    s11_ref = np.asarray(primer_datos['Complex'], dtype=complex)
    freq_obj = rf.Frequency.from_f(frec_ref, unit='hz')
    ntw_grilla = rf.Network(frequency=freq_obj, s=s11_ref.reshape(-1, 1, 1))
    ntw_grilla.plot_s_smith(ax=ax, draw_labels=True, show_legend=False, color='none')

    for nombre, datos in datos_dict.items():
        s11 = np.asarray(datos['Complex'], dtype=complex)
        linea, = ax.plot(s11.real, s11.imag, linewidth=1.8, label=nombre)
        color = linea.get_color()
        ax.plot(s11[0].real, s11[0].imag, 'o', color=color, ms=6)
        ax.plot(s11[-1].real, s11[-1].imag, 's', color=color, ms=6)

    ax.set_title(titulo)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.02, 0.02, f"\u25cf {frec_ref.min()/1e9:.3g} GHz   \u25a0 {frec_ref.max()/1e9:.3g} GHz",
            transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

    fig.tight_layout()
    if archivo_salida is not None:
        fig.savefig(archivo_salida, dpi=150)
        plt.close(fig)
        return archivo_salida
    return fig


if __name__ == "__main__":
    # Ejemplo rapido de uso (ajustar nombres de archivo segun corresponda)
    import matplotlib.pyplot as plt

    archivos = {
        "Agua": "sonda4-agua.s1p",
        "Aire": "sonda4-aire.s1p",
        "Alcohol etilico": "sonda4-alc-etilico.s1p",
        "Alcohol isopropilico": "sonda4-alc-isoprop.s1p",
        "Corto": "sonda4-short.s1p",
    }

    datos = {nombre: leer_s1p(path) for nombre, path in archivos.items()}
    graficar_s11_db(datos)
    plt.tight_layout()
    plt.show()