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
- Pillow es OPCIONAL (pip install pillow): solo se usa para mostrar la
  vista previa de las figuras generadas dentro de la GUI. Sin Pillow la
  GUI funciona igual, nada mas que sin esa vista previa (se puede abrir
  la carpeta de salida con el boton correspondiente para ver los .png).
- Los archivos Touchstone.py, funciones.py y reporte_pdf.py del
  proyecto original (no modificados por esta GUI) tienen que estar en
  la misma carpeta que este script.

Archivos de este proyecto
--------------------------
- Patrones.py              : modelos teoricos de permitividad (NPL MAT 23).
- analisis_permitividad.py : pipeline de analisis (calibracion, calculo,
                              graficos, informe PDF). Se puede seguir
                              corriendo solo, por consola, sin la GUI.
- gui_funciones.py          : logica no visual de la GUI (config, hilo
                              de fondo, validaciones).
- gui_permitividad.py       : la ventana en si (Tkinter).
- main_gui.py               : este archivo.
"""
from gui_permitividad import AppPermitividad


def main():
    app = AppPermitividad()
    app.mainloop()


if __name__ == "__main__":
    main()
