"""
main_combinatoria.py
--------------------
Punto de entrada de la GUI para el análisis combinatorio y selección
óptima de patrones de calibración de permitividad dieléctrica.

Ejecútalo con:
    python main_combinatoria.py
"""
import sys
import os

# Asegurar que la raíz del proyecto esté en sys.path
_DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))
if _DIR_RAIZ not in sys.path:
    sys.path.insert(0, _DIR_RAIZ)

from combinatoria.gui_combinatoria import AppCombinatoria


def main():
    app = AppCombinatoria()
    app.mainloop()


if __name__ == "__main__":
    main()

