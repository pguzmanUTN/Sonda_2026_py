"""
Grafico.py
----------
Valida el modelo teorico de agua (`Patrones.get_er_agua`) comparandolo
contra puntos de referencia publicados en:
    U. Kaatze, "Complex Permittivity of Water as a Function of Frequency
    and Temperature", J. Chem. Eng. Data, 1989.

Esta comparacion sirve para confirmar que el modelo teorico usado como
patron de calibracion (agua destilada) es correcto antes de usarlo en el
algoritmo de conversion de S11 a permitividad.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

from Patrones import get_er_agua

CARPETA_SALIDA = "./salidas"

# Los puntos de referencia de abajo corresponden a agua a 25°C (Kaatze,
# 1989). OJO: la temperatura del modelo teorico tiene que coincidir con la
# de los datos de referencia para que la comparacion tenga sentido.
TEMPERATURA_REF_C = 25.0

# Frecuencia [GHz]
f_ref = np.array([
    1.821, 2.000, 2.200, 2.610, 2.628, 2.800,
    3.417, 3.623, 3.733, 3.750, 3.922,
    5.300, 5.306, 5.433, 5.536, 5.638, 5.853,
    6.000, 6.145,
    6.300, 6.414, 6.729, 6.850, 6.958,
    7.267, 7.406, 7.681, 7.850, 7.900, 7.941,
    7.950, 8.244, 8.579, 8.979, 9.516,
    10.23, 10.45, 11.32,
    11.73, 12.00, 12.49, 12.50, 12.53,
    12.77, 13.14, 13.38, 13.82, 14.23,
    15.24, 15.63, 16.14, 16.60, 17.17,
    17.38, 17.43, 18.02, 18.48,
    19.02, 21.01, 23.53, 24.45, 26.43,
    26.64, 26.70, 26.79, 26.83, 27.61,
    28.13, 28.58, 36.56, 36.84, 37.81,
    37.97, 39.62, 52.25, 57.78
])

# Parte real ε' (a 25°C, Kaatze 1989)
eps_real_ref = np.array([
    78.0, 77.9, 77.5, 76.9, 77.1, 76.6,
    76.4, 75.8, 75.3, 75.6, 75.7,
    73.1, 73.1, 73.1, 72.7, 72.6, 72.3,
    71.9, 71.5,
    71.2, 70.9, 70.3, 69.7, 69.7,
    69.2, 68.8, 68.7, 67.9, 67.8, 67.8,
    67.6, 67.1, 66.8, 65.4, 63.8,
    62.0, 61.1, 59.5,
    58.5, 58.2, 56.6, 56.6, 56.6,
    55.8, 55.3, 54.0, 53.5, 52.6,
    50.0, 49.6, 48.2, 47.0, 45.4,
    45.5, 45.6, 44.4, 42.7,
    41.7, 38.0, 34.4, 32.8, 31.1,
    30.8, 31.0, 30.0, 31.0, 29.9,
    29.4, 29.2, 21.4, 21.3, 21.1,
    20.8, 20.6, 15.3, 11.5
])

# Parte imaginaria ε'' (magnitud de perdidas, positiva; a 25°C, Kaatze 1989)
eps_imag_ref = np.array([
    6.49, 7.57, 8.22, 9.77, 9.86, 10.6,
    12.9, 13.4, 14.5, 13.7, 14.3,
    18.7, 18.7, 19.3, 19.4, 19.8, 20.4,
    20.6, 21.1,
    21.6, 22.1, 22.7, 23.0, 23.3,
    24.2, 24.5, 25.3, 25.6, 25.7, 25.8,
    25.7, 26.0, 27.2, 27.7, 28.7,
    30.0, 30.2, 31.8,
    32.3, 33.1, 33.4, 33.3, 33.4,
    33.5, 34.2, 34.0, 34.5, 34.7,
    35.3, 35.8, 35.5, 35.9, 36.1,
    36.4, 36.1, 36.7, 36.6,
    36.2, 36.3, 35.8, 35.2, 34.8,
    34.6, 34.6, 34.3, 34.9, 34.1,
    34.1, 33.9, 30.1, 30.0, 29.6,
    29.5, 29.1, 23.9, 22.5
])

f_ref_hz = f_ref * 1e9

if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    freqs = np.arange(start=0.1e9, stop=60e9, step=10e6)
    agua_teo = get_er_agua(freqs, TEMPERATURA_REF_C)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    ax1.plot(freqs, np.real(agua_teo), label=f"Modelo teorico ({TEMPERATURA_REF_C:.0f}°C)")
    ax1.plot(f_ref_hz, eps_real_ref, 'o', markersize=4,
              label="Referencia Kaatze (1989), 25°C")
    ax1.set_xlabel("Frecuencia (Hz)")
    ax1.set_ylabel("er'  (parte real)")
    ax1.legend()
    ax1.minorticks_on()
    ax1.set_ylim([71, 79])
    ax1.set_xlim([0.1e9, 6e9])
    ax1.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    # OJO con el signo: get_er_agua devuelve er = er' - j*er'' (convencion
    # de la catedra), asi que la magnitud de perdidas positiva es -Im(er).
    ax2.plot(freqs, -np.imag(agua_teo), label=f"Modelo teorico ({TEMPERATURA_REF_C:.0f}°C)")
    ax2.plot(f_ref_hz, eps_imag_ref, 'o', markersize=4,
              label="Referencia Kaatze (1989), 25°C")
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("er''  (perdidas)")
    ax2.set_xlim([0.1e9, 6e9])
    ax2.set_ylim([0.250, 22])
    ax2.legend()
    ax2.minorticks_on()
    ax2.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)

    fig.tight_layout()
    ruta_salida = os.path.join(CARPETA_SALIDA, "validacion_modelo_agua.png")
    fig.savefig(ruta_salida, dpi=150)
    print(f"Figura guardada en: {ruta_salida}")
    plt.show()
