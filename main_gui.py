"""
main_gui.py
-----------
Punto de entrada de la GUI de analisis de permitividad. Correlo con:

    python main_gui.py

Requisitos
----------
- Python 3 con tkinter. Viene incluido en la mayoria de las
  instalaciones; en algunas distros de Linux hay que instalarlo aparte:
      sudo apt install python3-tk
- Pillow YA NO hace falta para la GUI (las vistas previas de graficos
  ahora se embeben en vivo con matplotlib/Tkinter directamente, con zoom,
  pan y lectura de coordenadas -- ver gui_permitividad.VisorFigura). Solo
  lo sigue usando reporte_pdf.py, de forma opcional, para ajustar el
  tamaño de las imagenes en el PDF (con un valor de resguardo si no esta).
- Los archivos Touchstone.py, funciones.py, Patrones.py y reporte_pdf.py
  del proyecto tienen que estar en la misma carpeta que este script.

Archivos de este proyecto
--------------------------
- Patrones.py              : modelos teoricos de permitividad (NPL MAT 23).
- funciones.py              : algebra S11 -> permitividad (metodos
                              simplificado y completo).
- Touchstone.py             : lectura/graficos de archivos .s1p.
- analisis_permitividad.py : pipeline de analisis (calibracion, calculo,
                              graficos, informe PDF). Se puede seguir
                              corriendo solo, por consola, sin la GUI.
- reporte_pdf.py            : armado del informe PDF.
- gui_funciones.py          : logica no visual de la GUI (config, hilo
                              de fondo, validaciones).
- gui_permitividad.py       : la ventana en si (Tkinter), con 5 pestañas:
                              Calibracion, Materiales, Vista previa S11,
                              Salida y ejecucion, y Comparar mediciones.
- main_gui.py               : este archivo.
"""
from gui_permitividad import AppPermitividad


def main():
    app = AppPermitividad()
    app.mainloop()


if __name__ == "__main__":
    main()
