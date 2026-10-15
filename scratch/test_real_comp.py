import os
import sys
import traceback

sys.path.insert(0, os.path.abspath('.'))

from combinatoria.analisis_combinatoria import ejecutar_analisis_combinatoria
from combinatoria.reporte_combinatoria_pdf import generar_reporte_combinacion_pdf

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

print("Ejecutando combinatoria...")
res = ejecutar_analisis_combinatoria(cfg)
print(f"Combinatoria finalizada. Total combos: {len(res['resultados_por_combinacion'])}")

combos = list(res['resultados_por_combinacion'].keys())
print("Combos disponibles:", combos)

c1 = combos[0]
c2 = combos[1] if len(combos) > 1 else combos[0]

print(f"Generando reporte comparativo entre '{c1}' y '{c2}'...")
try:
    ruta_pdf = 'scratch/test_real_comp.pdf'
    generar_reporte_combinacion_pdf(
        ruta_pdf,
        cfg,
        res,
        combo_id=c1,
        combo_id_comparar=c2,
        carpeta_exportar='scratch/export_comp',
        limitar_eje_y=True,
    )
    print("SUCCESS: Reporte comparativo generado correctamente en:", ruta_pdf)
except Exception as e:
    print("ERROR GENERANDO REPORTE COMPARATIVO:")
    traceback.print_exc()

