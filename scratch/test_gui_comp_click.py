import os
import sys
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, os.path.abspath('.'))

from combinatoria.gui_combinatoria import AppCombinatoria
from combinatoria.analisis_combinatoria import ejecutar_analisis_combinatoria

app = AppCombinatoria()

# Prepare mock config and run analysis
cfg = {
    'carpeta_datos': 'mediciones_sonda_3mm_09oct',
    'archivo_corto': 'short_sonda_3mm_9oct.s1p',
    'archivo_aire': 'open_sonda_3mm_9oct.s1p',
    'f_min_ghz': 0.5,
    'f_max_ghz': 6.0,
    'f_err_int1_ghz': 2.0,
    'f_err_int2_ghz': 4.0,
    'escala_log_frecuencia': True,
    'estrategia_gn': 'minimo_gn',
    'metodos_calculo': {
        'simplificado': False,
        'completo_sin_acond': True,
        'completo_con_acond': True,
        'mc_sin_acond': False,
        'mc_con_acond': False,
    },
    'mc_k_liquidos': 'todos',
    'acondicionar_gn': {
        'tipo': 'transicion_polinomica_cero',
        'modo': 'automatico',
        'umbral_pct': 30.0,
        'f1_ghz': 0.0,
        'f2_ghz': 0.5,
    },
    'candidatos': [
        {'nombre': "Agua", 'alias': "Agu", 'archivo': "agua21grados_sonda_3mm_9oct.s1p", 'modelo': "agua", 'temperatura': 21.0, 'habilitado': True},
        {'nombre': "Alcohol etilico", 'alias': "Eti", 'archivo': "etilico_sonda_3mm_9oct.s1p", 'modelo': "alcohol_etilico", 'temperatura': 25.0, 'habilitado': True},
        {'nombre': "Alcohol isopropilico", 'alias': "Iso", 'archivo': "isopropilico_sonda_3mm_9oct.s1p", 'modelo': "alcohol_isopropilico", 'temperatura': 25.0, 'habilitado': True},
    ],
}

print("Ejecutando analisis...")
res = ejecutar_analisis_combinatoria(cfg)
app._ultima_config = cfg
app._al_terminar_analisis(res)

print("Estado inicial:")
print("var_combo_seleccionada:", repr(app.var_combo_seleccionada.get()))
print("_combo_id_activo:", repr(getattr(app, '_combo_id_activo', None)))
print("btn_pdf_combo text:", repr(app.btn_pdf_combo['text']))
print("btn_pdf_combo state:", repr(app.btn_pdf_combo['state']))

print("\nSimulando marcar 'Comparar con B:'...")
app.var_comparar_activo.set(True)
app._al_toggle_comparar()

print("Despues de toggle comparar:")
print("var_combo_seleccionada:", repr(app.var_combo_seleccionada.get()))
print("_combo_id_activo:", repr(getattr(app, '_combo_id_activo', None)))
print("var_combo_b_seleccionada:", repr(app.var_combo_b_seleccionada.get()))
print("_combo_id_b_activo:", repr(getattr(app, '_combo_id_b_activo', None)))
print("btn_pdf_combo text:", repr(app.btn_pdf_combo['text']))
print("btn_pdf_combo state:", repr(app.btn_pdf_combo['state']))

# Now test calling _generar_informe_combinacion_pdf with mocks for filedialog & messagebox
with patch('tkinter.filedialog.asksaveasfilename', return_value='scratch/gui_test_comp.pdf'), \
     patch('tkinter.messagebox.askyesno', return_value=False), \
     patch('tkinter.messagebox.showerror') as mock_err, \
     patch('tkinter.messagebox.showwarning') as mock_warn:
    app._generar_informe_combinacion_pdf()
    if mock_err.called:
        print("ERROR POPUP:", mock_err.call_args)
    if mock_warn.called:
        print("WARN POPUP:", mock_warn.call_args)

print("Fin del test.")
app.destroy()
