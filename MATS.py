"""
Script para abrir y explorar el archivo patrones_expo2.mat
Requiere: pip install scipy numpy matplotlib scikit-rf
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from matplotlib.collections import LineCollection

# ─── Cargar el archivo ────────────────────────────────────────────────────────
ruta = "patrones_expo2.mat"   # <-- cambiá la ruta si es necesario
mat = scipy.io.loadmat(ruta)

# ─── Filtrar variables internas de MATLAB (comienzan con '__') ────────────────
variables = {k: v for k, v in mat.items() if not k.startswith("__")}

print("=" * 60)
print(f"Archivo: {ruta}")
print(f"Variables encontradas: {len(variables)}")
print("=" * 60)

for nombre, datos in variables.items():
    if nombre in ("handles", "None"):
        print(f"\n[{nombre}]  →  estructura de interfaz gráfica MATLAB (ignorada)")
        continue
    print(f"\n[{nombre}]")
    print(f"  Tipo  : {datos.dtype}")
    print(f"  Shape : {datos.shape}  ({datos.size} elementos)")
    if np.iscomplexobj(datos):
        print(f"  Datos complejos (parte real + imaginaria)")
        print(f"  |S11| mín: {np.abs(datos).min():.4f}  |S11| máx: {np.abs(datos).max():.4f}")
    else:
        print(f"  Mín: {datos.min():.6g}   Máx: {datos.max():.6g}")
    flat = datos.flatten()
    n_show = min(5, len(flat))
    print(f"  Primeros {n_show} valores: {flat[:n_show]}")
    print(f"  Últimos  {n_show} valores: {flat[-n_show:]}")

print("\n" + "=" * 60)
print("RESUMEN DE MEDICIONES VNA")
print("=" * 60)

freq     = mat["FRECUENCIA"].flatten()
freq_ghz = freq / 1e9

print(f"\nFrecuencias: {len(freq)} puntos")
print(f"  Desde : {freq_ghz[0]:.3f} GHz")
print(f"  Hasta : {freq_ghz[-1]:.3f} GHz")
print(f"  Paso  : {(freq[1]-freq[0])/1e6:.1f} MHz")

patrones = {
    "Abierto"     : mat["S11_ABIERTO"].flatten(),
    "Corto"       : mat["S11_CORTO"].flatten(),
    "Agua"        : mat["S11_AGUA"].flatten(),
    "Isopropílico": mat["S11_ISOPROPILICO"].flatten(),
}

print("\nPatrones medidos:")
for nombre, s11 in patrones.items():
    mag_db = 20 * np.log10(np.abs(s11))
    fase   = np.angle(s11, deg=True)
    print(f"\n  {nombre}")
    print(f"    |S11| en dB : mín={mag_db.min():.2f} dB  máx={mag_db.max():.2f} dB")
    print(f"    Fase        : mín={fase.min():.1f}°    máx={fase.max():.1f}°")

# ─── Colores compartidos entre plots ─────────────────────────────────────────
colores = {
    "Abierto"     : "tab:red",
    "Corto"       : "tab:green",
    "Agua"        : "tab:blue",
    "Isopropílico": "tab:orange",
}

# ─── Plot 1: |S11| en dB vs Frecuencia ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 4))

for nombre, s11 in patrones.items():
    mag_db = 20 * np.log10(np.abs(s11))
    ax1.plot(freq_ghz, mag_db, label=nombre, color=colores[nombre], linewidth=1.8)

ax1.set_xlabel("Frecuencia (GHz)")
ax1.set_ylabel("|S11| (dB)")
ax1.set_title("|S11| vs Frecuencia")
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("s11_db.png", dpi=150, bbox_inches='tight')
print("\nGuardado: s11_db.png")

# ─── Plot 2: Carta de Smith ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 7))

# Grilla Smith con scikit-rf
freq_hz = rf.Frequency.from_f(freq, unit='hz')
nw      = rf.Network(frequency=freq_hz, s=patrones["Abierto"].reshape(-1, 1, 1))
nw.plot_s_smith(ax=ax2, draw_labels=True, show_legend=False, color='none')

for nombre, s11 in patrones.items():
    ax2.plot(s11.real, s11.imag, label=nombre, color=colores[nombre], linewidth=1.8)
    ax2.plot(s11[0].real,  s11[0].imag,  'o', color=colores[nombre], ms=6)  # inicio
    ax2.plot(s11[-1].real, s11[-1].imag, 's', color=colores[nombre], ms=6)  # fin

ax2.set_title("Carta de Smith — S11 (0.1–10 GHz)")
ax2.legend(loc='lower right')
ax2.text(0.02, 0.02, '● 0.1 GHz   ■ 10 GHz',
         transform=ax2.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig("smith_chart.png", dpi=150, bbox_inches='tight')
print("Guardado: smith_chart.png")

plt.show()