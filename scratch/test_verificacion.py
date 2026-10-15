import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

# 1. Test alias generation
from combinatoria.gui_combinatoria import generar_alias_defecto

assert generar_alias_defecto('Agua') == 'Agu', f"Got {generar_alias_defecto('Agua')}"
assert generar_alias_defecto('Alcohol etilico') == 'Eti', f"Got {generar_alias_defecto('Alcohol etilico')}"
assert generar_alias_defecto('Alcohol isopropilico') == 'Iso', f"Got {generar_alias_defecto('Alcohol isopropilico')}"
assert generar_alias_defecto('Ciclohexano') == 'Cic', f"Got {generar_alias_defecto('Ciclohexano')}"
assert generar_alias_defecto('Acetona') == 'Ace', f"Got {generar_alias_defecto('Acetona')}"
assert generar_alias_defecto('Solucion salina normal') == 'Sal', f"Got {generar_alias_defecto('Solucion salina normal')}"
print("Test 1 (generar_alias_defecto): PASSED")

# 2. Test graficar_comparacion with limitar_eje_y
from analisis_permitividad import graficar_comparacion

frecs = np.linspace(1e9, 5e9, 50)
er_teo = np.full(50, 2.0 + 0.1j)
# Curve with one outlier point exceeding [2-1, 2+1] = [1, 3]
er_med = np.full(50, 2.1 + 0.12j)
er_med[10] = 50.0 + 10j

fig_lim = graficar_comparacion(
    frecs, {'Medido': er_med}, er_teo,
    titulo="Test con limite", limitar_eje_y=True
)
ax_re, ax_im = fig_lim.axes
assert ax_re.get_ylim() == (1.0, 3.0), f"Expected (1.0, 3.0), got {ax_re.get_ylim()}"
print("Test 2a (graficar_comparacion con limitar_eje_y=True): PASSED")

fig_no_lim = graficar_comparacion(
    frecs, {'Medido': er_med}, er_teo,
    titulo="Test sin limite", limitar_eje_y=False
)
ax_re_no, ax_im_no = fig_no_lim.axes
assert ax_re_no.get_ylim()[1] >= 50.0, f"Expected ylim >= 50, got {ax_re_no.get_ylim()}"
print("Test 2b (graficar_comparacion con limitar_eje_y=False): PASSED")

# 3. Test comparative PDF report generation
from combinatoria.reporte_combinatoria_pdf import generar_reporte_combinacion_pdf, generar_reporte_combinatoria_pdf

config = {
    'archivo_corto': 'corto.s1p',
    'archivo_aire': 'aire.s1p',
    'f_min_ghz': 1.0,
    'f_max_ghz': 5.0,
    'candidatos': [
        {'nombre': 'Agua', 'alias': 'Agu', 'archivo': 'agua.s1p', 'modelo': 'agua', 'temperatura': 25.0, 'habilitado': True},
        {'nombre': 'Alcohol isopropilico', 'alias': 'Iso', 'archivo': 'isopropilico.s1p', 'modelo': 'alcohol_isopropilico', 'temperatura': 25.0, 'habilitado': True},
        {'nombre': 'Ciclohexano', 'alias': 'Cic', 'archivo': 'ciclohexano.s1p', 'modelo': 'ciclohexano', 'temperatura': 25.0, 'habilitado': True},
    ]
}

muestra_data_A = {
    'muestra': 'Ciclohexano',
    'rol': 'Muestra',
    'er_medido': er_med,
    'er_teorico': er_teo,
    'etiqueta_teorico': 'Ciclohexano Debye',
    'frecs_r': frecs,
    'puntaje_int1': 0.05,
    'puntaje_int2': 0.08,
    'puntaje_int3': 0.12,
    'puntaje_global': 0.083,
    'intervalos': [
        {'intervalo_str': '1.00 - 2.00 GHz', 'err_re_medio': 0.05, 'err_re_max': 0.1, 'err_im_medio': 0.02, 'err_im_max': 0.05},
        {'intervalo_str': '2.00 - 4.00 GHz', 'err_re_medio': 0.08, 'err_re_max': 0.15, 'err_im_medio': 0.03, 'err_im_max': 0.06},
        {'intervalo_str': '4.00 - 5.00 GHz', 'err_re_medio': 0.12, 'err_re_max': 0.2, 'err_im_medio': 0.04, 'err_im_max': 0.08},
    ]
}

muestra_data_B = {
    'muestra': 'Ciclohexano',
    'rol': 'Muestra',
    'er_medido': np.full(50, 2.05 + 0.11j),
    'er_teorico': er_teo,
    'etiqueta_teorico': 'Ciclohexano Debye',
    'frecs_r': frecs,
    'puntaje_int1': 0.02,
    'puntaje_int2': 0.03,
    'puntaje_int3': 0.04,
    'puntaje_global': 0.030,
    'intervalos': [
        {'intervalo_str': '1.00 - 2.00 GHz', 'err_re_medio': 0.02, 'err_re_max': 0.04, 'err_im_medio': 0.01, 'err_im_max': 0.02},
        {'intervalo_str': '2.00 - 4.00 GHz', 'err_re_medio': 0.03, 'err_re_max': 0.05, 'err_im_medio': 0.01, 'err_im_max': 0.03},
        {'intervalo_str': '4.00 - 5.00 GHz', 'err_re_medio': 0.04, 'err_re_max': 0.07, 'err_im_medio': 0.02, 'err_im_max': 0.04},
    ]
}

resultados = {
    'frecs_r': frecs,
    'log_x': True,
    'intervalos': [(1e9, 2e9), (2e9, 4e9), (4e9, 5e9)],
    'intervalos_str': ['1.00 - 2.00 GHz', '2.00 - 4.00 GHz', '4.00 - 5.00 GHz'],
    'resumen_ranking_global': [
        {'combo_id': 'combo_A', 'score_global': 0.083},
        {'combo_id': 'combo_B', 'score_global': 0.030},
    ],
    'resultados_por_combinacion': {
        'combo_A': {
            'metodo': 'Completo (original)',
            'patrones_str': 'Agua + Alcohol isopropilico',
            'p3': 'Agua',
            'p4': 'Alcohol isopropilico',
            'es_mc': False,
            'puntaje_promedio_total': 0.083,
            'puntaje_promedio_patrones': 0.04,
            'puntaje_promedio_restantes': 0.083,
            'muestras': {'Ciclohexano': muestra_data_A},
            'Gn_original': np.full(50, 0.01 + 0.02j),
        },
        'combo_B': {
            'metodo': 'Completo (acondicionado)',
            'patrones_str': 'Alcohol isopropilico + Agua',
            'p3': 'Alcohol isopropilico',
            'p4': 'Agua',
            'es_mc': False,
            'puntaje_promedio_total': 0.030,
            'puntaje_promedio_patrones': 0.02,
            'puntaje_promedio_restantes': 0.030,
            'muestras': {'Ciclohexano': muestra_data_B},
            'Gn_original': np.full(50, 0.01 + 0.02j),
        }
    },
    'resultados_por_muestra': {
        'Ciclohexano': [
            {
                'p3': 'Alcohol isopropilico', 'p4': 'Agua', 'metodo': 'Completo',
                'es_patron': False, 'rol': 'Muestra',
                'puntaje_int1': 0.02, 'puntaje_int2': 0.03, 'puntaje_int3': 0.04, 'puntaje_global': 0.03,
                'er_medido': muestra_data_B['er_medido'], 'er_teorico': er_teo,
                'intervalos': muestra_data_B['intervalos'],
            },
            {
                'p3': 'Agua', 'p4': 'Alcohol isopropilico', 'metodo': 'Completo',
                'es_patron': False, 'rol': 'Muestra',
                'puntaje_int1': 0.05, 'puntaje_int2': 0.08, 'puntaje_int3': 0.12, 'puntaje_global': 0.083,
                'er_medido': muestra_data_A['er_medido'], 'er_teorico': er_teo,
                'intervalos': muestra_data_A['intervalos'],
            }
        ]
    }
}

os.makedirs('scratch', exist_ok=True)
pdf_single = 'scratch/test_single_combo.pdf'
generar_reporte_combinacion_pdf(pdf_single, config, resultados, 'combo_A', limitar_eje_y=True)
assert os.path.isfile(pdf_single) and os.path.getsize(pdf_single) > 1000, "Single combo PDF failed"
print("Test 3a (generar_reporte_combinacion_pdf single): PASSED")

pdf_comp = 'scratch/test_comp_combo.pdf'
generar_reporte_combinacion_pdf(pdf_comp, config, resultados, 'combo_A', combo_id_comparar='combo_B', limitar_eje_y=True)
assert os.path.isfile(pdf_comp) and os.path.getsize(pdf_comp) > 1000, "Comparative combo PDF failed"
print("Test 3b (generar_reporte_combinacion_pdf comparative A vs B): PASSED")

pdf_top3 = 'scratch/test_top3.pdf'
generar_reporte_combinatoria_pdf(pdf_top3, config, resultados, limitar_eje_y=True)
assert os.path.isfile(pdf_top3) and os.path.getsize(pdf_top3) > 1000, "Top 3 PDF failed"
print("Test 3c (generar_reporte_combinatoria_pdf top 3): PASSED")

print("ALL TESTS PASSED SUCCESSFULLY!")
