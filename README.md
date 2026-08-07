# Analizador de Permitividad — Sonda Coaxial Open-Ended

Software de procesamiento para el sistema de medición de permitividad
compleja descripto en:

> A. Henze, C. Kupzevich, R. Mascheroni, V. Zerpa, P. Della Roca, J.
> Arballo, *"Desarrollo de un sistema de medición de la permitividad
> compleja en alimentos para frecuencias de microondas mediante
> parámetros S"*, IEEE ARGENCON 2024.

A partir de los parámetros S11 medidos con un VNA y una sonda coaxial
tipo open-ended, calcula la permitividad relativa compleja
εr = εr′ − j·εr″ del material bajo ensayo (MUT), usando los dos métodos
de conversión descriptos en el paper:

- **Método simplificado** (3 patrones: corto, aire, agua — Gn = 0):
  fórmula algebraica cerrada, válida sobre todo para materiales de bajas
  pérdidas y/o frecuencias más bajas.
- **Método completo** (4 patrones: corto, aire, agua, alcohol
  isopropílico — con conductancia normalizada Gn): resuelve un polinomio
  de 5to orden por frecuencia, más preciso en un rango más amplio.

## Instalación

```bash
pip install -r requirements.txt
```

**Nota sobre tkinter:** la GUI usa `tkinter`, que es parte de la
librería estándar de Python (no se instala con `pip`). En Windows/macOS
ya viene con el instalador oficial de Python. En Linux hace falta
instalarlo aparte:

```bash
sudo apt install python3-tk
```

El resto del proyecto (script por consola, `funciones.py`, `Patrones.py`,
`Touchstone.py`, `Grafico.py`, `MATS.py`) no depende de `tkinter`.

## Uso

### Opción A — Interfaz gráfica (recomendada)

```bash
python main_gui.py
```

La GUI permite: cargar los `.s1p` de calibración y de cada material,
elegir el modelo teórico y la temperatura de cada uno, ver una vista
previa de S11 (módulo/fase y Smith) antes de correr el análisis, correr
todo en un hilo de fondo con barra de progreso (mostrando el paso actual)
y la posibilidad de cancelar a mitad de camino (se genera igual un
informe PDF parcial con lo ya calculado), copiar o guardar el log de la
consola, y guardar/cargar/reabrir configuraciones recientes como JSON
para no tener que repetir la carga a mano.

### Opción B — Script por consola

Editar las constantes al principio de `analisis_permitividad.py`
(`ARCHIVOS_CALIBRACION`, `MATERIALES`, temperaturas, rango de
frecuencias) y correr:

```bash
python analisis_permitividad.py
```

## Flujo de calibración

Se necesitan 4 patrones de referencia (Sección III del paper):

| Patrón | Rol |
|---|---|
| Cortocircuito | Y → ∞, simplifica el modelo matemático |
| Aire (circuito abierto) | εr ≈ 1.0006 |
| Agua destilada | Modelo teórico dependiente de temperatura |
| Alcohol isopropílico | 4to patrón, solo necesario para el método completo (permite calcular Gn) |

El agua se usa además como **chequeo de calibración**: al procesarla como
si fuera un material más, el resultado tiene que coincidir casi
exactamente con su propio modelo teórico. Si no da ~0 % de error, hay un
problema de lectura o calibración antes de analizar cualquier material
real — esto se corre automáticamente al principio de cada análisis.

## Salidas generadas

En la carpeta de salida (por defecto `./salidas`):

- `chequeo_calibracion_agua.png` — agua medida vs. teórica.
- `chequeo_calibracion_Gn.png` — diagnóstico de la conductancia
  normalizada Gn(f) (ver más abajo).
- `er_<material>.png`, `s11_<material>.png`, `smith_<material>.png` —
  por cada material: permitividad, S11 módulo/fase y diagrama de Smith.
- `tabla_<material>.csv` — permitividad calculada en todas las
  frecuencias.
- Un informe PDF único (`informe_permitividad.pdf` por defecto) con
  todo lo anterior más tablas de error resumidas.

### Diagnóstico: Gn(f)

`Gn` se calcula únicamente a partir de los 4 patrones de calibración (no
depende de ningún material medido), así que es el mismo para todos los
materiales analizados con el método completo en una corrida. Al estar
relacionada con G0/(jωC0) — una propiedad física continua de la sonda —
la curva Gn(f) debería verse suave. Un salto brusco o un pico aislado
suele delatar un problema con la medición de alguno de los 4 patrones,
típicamente el alcohol isopropílico (el único que interviene en este
cálculo y en ningún otro lado del método simplificado, por lo que un
problema ahí puede pasar desapercibido si solo se mira el chequeo de
calibración del agua). Conviene revisar esta figura **antes** de confiar
en los resultados del método completo.

## Convención de signos

Todo el proyecto usa la convención εr = εr′ − j·εr″ (con εr″ > 0 para
materiales con pérdidas), consistente con el paper y con la bibliografía
de referencia (Liebe-Hufford-Manabe 1991 para agua; NPL Report MAT 23
para los demás líquidos patrón).

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `Patrones.py` | Modelos teóricos de permitividad de los líquidos patrón (agua, alcoholes, DMSO, etilenglicol — NPL MAT 23) |
| `funciones.py` | Algoritmos de conversión S11 → εr (métodos simplificado y completo) |
| `Touchstone.py` | Lectura de `.s1p` y gráficos de S11 (módulo/fase, Smith) |
| `analisis_permitividad.py` | Pipeline completo: calibración, cálculo, gráficos, CSV, informe PDF |
| `reporte_pdf.py` | Armado del informe PDF a partir de los resultados del pipeline |
| `gui_funciones.py` | Lógica no visual de la GUI (config, hilo de fondo, validaciones) |
| `gui_permitividad.py` | Ventana Tkinter |
| `main_gui.py` | Punto de entrada de la GUI |
| `Grafico.py` | Valida el modelo teórico del agua contra datos de Kaatze (1989) |
| `MATS.py` | Script exploratorio para abrir el `.mat` original de mediciones |

## Referencias

- A. Henze et al., "Desarrollo de un sistema de medición de la
  permitividad compleja en alimentos para frecuencias de microondas
  mediante parámetros S", IEEE ARGENCON 2024.
- A. P. Gregory, R. N. Clarke, "Tables of the Complex Permittivity of
  Dielectric Reference Liquids at Frequencies up to 5 GHz", NPL Report
  MAT 23, 2012.
- H. J. Liebe, G. A. Hufford, T. Manabe, "A model for the complex
  permittivity of water at frequencies below 1 THz", Int. J. Infrared
  and Millimeter Waves, 1991.
- U. Kaatze, "Complex Permittivity of Water as a Function of Frequency
  and Temperature", J. Chem. Eng. Data, 1989.
