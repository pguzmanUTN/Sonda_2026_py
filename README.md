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

- **Método simplificado** (3 patrones: corto, aire, patrón 3 — Gn = 0):
  fórmula algebraica cerrada, válida sobre todo para materiales de bajas
  pérdidas y/o frecuencias más bajas.
- **Método completo** (4 patrones: corto, aire, patrón 3, patrón 4 — con
  conductancia normalizada Gn): resuelve un polinomio de 5to orden por
  frecuencia, más preciso en un rango más amplio.

**Patrón 3 y patrón 4** son, por defecto, agua destilada y alcohol
isopropílico (el par tradicional de la cátedra), pero pueden ser
**cualquier par de líquidos** con modelo teórico conocido (ver
`Patrones.PATRONES_TEORICOS`): no hace falta calibrar sí o sí con esos
dos, alcanza con elegir el líquido y la temperatura de cada uno en la
pestaña de Calibración.

## Instalación

```bash
pip install -r requirements.txt
```

**Nota sobre tkinter:** la GUI usa `tkinter`, parte de la librería
estándar de Python (no se instala con `pip`). En Linux hace falta
instalarlo aparte:

```bash
sudo apt install python3-tk
```

## Uso

### Opción A — Interfaz gráfica (recomendada)

```bash
python main_gui.py
```

La GUI tiene 6 pestañas:

1. **Calibración** — archivos de corto, aire, y los patrones 3/4 (cada
   uno con su líquido y temperatura elegibles). El método completo
   (Patrón 4 + cálculo de Gn) se puede **deshabilitar por completo** con
   un checkbox si solo interesa el método simplificado — en ese caso no
   hace falta cargar ni el archivo ni el modelo de Patrón 4 para nada.
   También se elige ahí el **punto de arranque de la marcha en
   frecuencia** del método completo: automático (donde |Gn(f)| es
   realmente mínimo en todo el barrido) o el clásico (siempre en la
   frecuencia más alta). Cada campo de archivo muestra un indicador ✓/✗
   que confirma si el archivo existe, actualizado en vivo mientras se
   escribe.
2. **Materiales** — lista de muestras a analizar. Se puede agregar de a
   una (con el diálogo de siempre), **agregar una carpeta entera** (crea
   un material por cada `.s1p` encontrado, sin duplicar los que ya
   estuvieran cargados), o **duplicar** una fila existente para muestras
   parecidas.
3. **Vista previa S11** — módulo/fase y diagrama de Smith de cualquier
   patrón o material, leyendo el `.s1p` directo, sin correr el análisis
   completo.
4. **Salida y ejecución** — corre el pipeline en un hilo de fondo (con
   barra de progreso, indicador del paso actual, botón para cancelar a
   mitad de camino, y consola con log copiable/guardable), y muestra los
   resultados.
5. **Comparar mediciones** — superpone en un mismo gráfico varias curvas
   ya calculadas (de un `tabla_<material>.csv`), útil para comparar la
   misma muestra medida en días distintos o distintos cortes de un
   material.
6. **Modelos teóricos** — grafica cualquier modelo de `Patrones.py`
   directamente, sin necesidad de ningún dato medido: elegís el modelo,
   la temperatura, el rango de frecuencias y la cantidad de puntos. Sirve
   para explorar cómo se ve un modelo, o para comparar el mismo líquido a
   distintas temperaturas (o líquidos distintos entre sí) superpuestos.
   El botón **"Usar Patrón 3/4 actuales"** carga automáticamente el
   modelo y la temperatura configurados en la pestaña de Calibración, sin
   tener que volver a tipearlos.

Tanto en la pestaña 5 como en la 6, además de agregar y editar, se puede
**duplicar** una curva para variar un solo parámetro (p.ej. la
temperatura) sin rehacer todo el diálogo.

Los gráficos de las pestañas 3, 4, 5 y 6 son **interactivos**: zoom (rueda
del mouse o herramienta de lupa), pan (arrastrar), botón "home" para
volver a la vista original, y lectura de las coordenadas del dato bajo
el cursor — la misma interacción que da el backend Qt de matplotlib,
pero embebida en Tkinter.

La configuración se guarda/carga como JSON (con historial de recientes
en el menú Archivo). Las configuraciones guardadas por versiones
anteriores de este programa (con los patrones fijos a agua/alcohol
isopropílico) se migran solas al abrirlas.

### Opción B — Script por consola

Editar las constantes al principio de `analisis_permitividad.py`
(`ARCHIVOS_CALIBRACION`, `PATRON3_MODELO_CAL`/`PATRON4_MODELO_CAL`,
`MATERIALES`, rango de frecuencias) y correr:

```bash
python analisis_permitividad.py
```

## Carpeta de salida

Cada corrida guarda sus archivos en:

```
<carpeta_salida>/<AAAA-MM-DD>/<HH-MM-SS>/
    chequeo_calibracion_patron3.png
    chequeo_calibracion_Gn.png       (solo si el metodo completo esta habilitado)
    er_<material>.png, s11_<material>.png, smith_<material>.png
    tabla_<material>.csv
    informe_permitividad.pdf
```

Es decir, una subcarpeta por **día** y por **hora** de cada corrida, para
no pisar resultados anteriores y tener un historial ordenado. Además,
los `.s1p` de **entrada** (calibración + cada material) se copian a la
carpeta del día (compartida entre todas las corridas de ese día, sin
duplicar si se corre varias veces con la misma medición), como registro
de con qué mediciones exactas se generó cada informe.

El botón "Abrir carpeta de salida" de la GUI abre directamente la
carpeta específica de la última corrida.

## Flujo de calibración

Se necesitan 4 patrones de referencia (Sección III del paper):

| Patrón | Rol |
|---|---|
| Cortocircuito | Y → ∞, simplifica el modelo matemático |
| Aire (circuito abierto) | εr ≈ 1.0006 |
| Patrón 3 (por defecto: agua) | Líquido con modelo teórico conocido |
| Patrón 4 (por defecto: alcohol isopropílico) | 4to patrón, solo necesario para el método completo (permite calcular Gn) |

El **método completo es opcional**: si solo interesa el método
simplificado (por ejemplo, para materiales de bajas pérdidas), se puede
deshabilitar desde la pestaña de Calibración, y en ese caso no hace
falta el patrón 4 en absoluto — ni el archivo, ni el modelo, ni la
temperatura.

El patrón 3 se usa además como **chequeo de calibración**: al procesarlo
como si fuera un material más, el resultado tiene que coincidir casi
exactamente con su propio modelo teórico. Si no da ~0 % de error, hay un
problema de lectura o calibración antes de analizar cualquier material
real — esto se corre automáticamente al principio de cada análisis.
Cuando el método completo está habilitado, se hace el **mismo chequeo
también para el patrón 4** — una segunda validación independiente: si el
patrón 3 pasa pero el 4 no (o al revés), el problema está
específicamente en la medición o el modelo de ese patrón, algo que
mirando solo el chequeo del otro no se nota.

### Punto de arranque de la marcha en frecuencia (método completo)

El método completo resuelve, para cada frecuencia, un polinomio de 5to
orden y elige la raíz físicamente correcta "marchando" en frecuencia: en
el punto de partida usa como semilla la predicción del método
simplificado, y a partir de ahí usa la solución del punto vecino ya
resuelto. Hay tres opciones para elegir ese punto de partida:

- **Automático (default)**: arranca donde |Gn(f)| es realmente mínimo en
  todo el barrido -- ahí el método simplificado da la mejor aproximación
  posible, sea cual sea la frecuencia donde eso ocurra -- y marcha hacia
  ambos lados desde ahí.
- **Clásico**: arranca siempre en la frecuencia más alta del barrido.
  Válido solo si |Gn(f)| decrece en forma monótona con la frecuencia (C0
  y G0 aproximadamente constantes en toda la sonda); en patrones reales
  eso no siempre se cumple, así que esta opción se deja disponible para
  comparar contra la automática, no como default.
- **Comparar ambas**: corre las dos estrategias y muestra las dos curvas
  juntas (en los chequeos de patrón 3/4 y en cada material con método
  completo), para ver de un vistazo si la elección cambia algo con tus
  datos reales, sin tener que correr el análisis dos veces a mano.

### Diagnóstico: Gn(f)

`Gn` se calcula únicamente a partir de los 4 patrones de calibración (no
depende de ningún material medido). Se grafica en módulo y fase; al estar
relacionada con G0/(jωC0) — una propiedad física continua de la sonda —
la curva debería verse suave. Un salto brusco o un pico aislado suele
delatar un problema con la medición de alguno de los 4 patrones,
típicamente el patrón 4 (el único que interviene en este cálculo). El
gráfico marca con una línea vertical punteada en qué frecuencia arranca
la marcha con cada estrategia configurada, para confirmar de un vistazo
que cae en una zona con |Gn| chico.

## Convención de signos

Todo el proyecto usa la convención εr = εr′ − j·εr″ (con εr″ > 0 para
materiales con pérdidas), consistente con el paper y con la bibliografía
de referencia (Liebe-Hufford-Manabe 1991 para agua; NPL Report MAT 23
para los demás líquidos patrón).

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| `Patrones.py` | Modelos teóricos de permitividad de los líquidos patrón (agua, alcoholes, DMSO, etilenglicol — NPL MAT 23) |
| `funciones.py` | Álgebra S11 → εr (métodos simplificado y completo), agnóstica de qué líquidos se usan como patrón 3/4 |
| `Touchstone.py` | Lectura de `.s1p` y gráficos de S11 (módulo/fase, Smith) |
| `analisis_permitividad.py` | Pipeline completo: calibración, cálculo, gráficos, CSV, informe PDF, carpetas por fecha/hora |
| `reporte_pdf.py` | Armado del informe PDF a partir de los resultados del pipeline |
| `gui_funciones.py` | Lógica no visual de la GUI (config, migración de esquema viejo, hilo de fondo, validaciones) |
| `gui_permitividad.py` | Ventana Tkinter (5 pestañas, gráficos interactivos vía `VisorFigura`) |
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
