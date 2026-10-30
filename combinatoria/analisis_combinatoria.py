"""
combinatoria/analisis_combinatoria.py
-------------------------------------
Motor de cálculo combinatorio para la selección óptima de patrones
de calibración de la sonda coaxial de permitividad dieléctrica.

Evalúa todas las combinaciones posibles de patrones (P3, P4 para método
completo; P3 para método simplificado) y mide el error cometido al calibrar
los demás líquidos candidatos restantes frente a sus modelos teóricos.
"""
import os
import sys
import copy
import warnings
from itertools import combinations
import numpy as np

# Asegurar que el directorio raíz esté en sys.path para importar módulos base
_DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR_RAIZ not in sys.path:
    sys.path.insert(0, _DIR_RAIZ)

from Touchstone import leer_s1p, mismas_frecuencias, resamplear
from funciones import (
    get_er_DUTm,
    get_er_DUT_completo,
    calcular_Gn,
    acondicionar_gn,
    error_absoluto,
    calibrar_minimos_cuadrados,
    reajustar_abc_con_gn_fijo,
    get_er_DUT_minimos_cuadrados,
)
from Patrones import (
    PATRONES_TEORICOS,
    etiqueta_patron,
    etiqueta_curva_teorica,
    perdidas_comparables,
    modela_perdidas,
)
from analisis_permitividad import (
    cargar_calibracion,
    resolver_modelo_teorico,
    construir_intervalos_error,
    calcular_error_intervalos,
    ETIQUETA_ESTRATEGIA_GN,
)


def _cargar_s1p_con_cache(ruta, cache):
    """Carga un archivo .s1p usando una caché en memoria para no releerlo del disco."""
    ruta_abs = os.path.abspath(ruta)
    if ruta_abs not in cache:
        cache[ruta_abs] = leer_s1p(ruta_abs)
    return cache[ruta_abs]


def ejecutar_analisis_combinatoria(config, callback_progreso=None, callback_log=None, fn_cancelado=None):
    """
    Ejecuta el barrido combinatorio completo de patrones siguiendo exactamente
    el mismo flujo de procesamiento que `analisis_permitividad.ejecutar_analisis`.
    """
    def log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    def progreso(paso, total, msg):
        if callback_progreso:
            callback_progreso(paso, total, msg)

    def esta_cancelado():
        return fn_cancelado() if fn_cancelado else False

    carpeta_datos = config.get('carpeta_datos', '')
    archivo_corto = config.get('archivo_corto', '')
    archivo_aire = config.get('archivo_aire', '')
    ruta_corto = os.path.join(carpeta_datos, archivo_corto)
    ruta_aire = os.path.join(carpeta_datos, archivo_aire)

    if not os.path.isfile(ruta_corto):
        raise FileNotFoundError(f"No se encontró el archivo de cortocircuito: '{ruta_corto}'")
    if not os.path.isfile(ruta_aire):
        raise FileNotFoundError(f"No se encontró el archivo de circuito abierto (aire): '{ruta_aire}'")

    f_min_ghz = float(config.get('f_min_ghz', 0.5))
    f_max_ghz = float(config.get('f_max_ghz', 6.0))
    f_int1_ghz = config.get('f_err_int1_ghz', 2.0)
    f_int2_ghz = config.get('f_err_int2_ghz', 4.0)
    log_x = bool(config.get('escala_log_frecuencia', True))

    intervalos = construir_intervalos_error(f_min_ghz, f_max_ghz, f_int1_ghz, f_int2_ghz)
    intervalos_str = [f"{fa/1e9:.2f} - {fb/1e9:.2f} GHz" for fa, fb in intervalos]

    metodos_cfg = config.get('metodos_calculo', {})
    evaluar_simpl = bool(metodos_cfg.get('simplificado', False))
    evaluar_comp_sin = bool(metodos_cfg.get('completo_sin_acond', True))
    evaluar_comp_con = bool(metodos_cfg.get('completo_con_acond', False))
    evaluar_mc_sin = bool(metodos_cfg.get('mc_sin_acond', False))
    evaluar_mc_con = bool(metodos_cfg.get('mc_con_acond', False))

    if not (evaluar_simpl or evaluar_comp_sin or evaluar_comp_con or evaluar_mc_sin or evaluar_mc_con):
        raise ValueError("Debe seleccionar al menos un método de cálculo para evaluar.")

    estrategia_gn = config.get('estrategia_gn', 'minimo_gn')
    f_semilla_ghz = float(config.get('f_semilla_ghz', 3.0) or 3.0)
    f_semilla_hz = f_semilla_ghz * 1e9
    cfg_acond = config.get('acondicionar_gn', {})
    mc_k_cfg = config.get('mc_k_liquidos', 'todos')

    # Filtrar candidatos habilitados y verificar existencia de archivos
    candidatos_todos = config.get('candidatos', [])
    candidatos_activos = []
    for cand in candidatos_todos:
        if not cand.get('habilitado', True):
            continue
        ruta_c = os.path.join(carpeta_datos, cand.get('archivo', ''))
        if not os.path.isfile(ruta_c):
            log(f"  [aviso] No se encontró el archivo '{cand.get('archivo')}' para '{cand.get('nombre')}'. Se omite.")
            continue
        if not cand.get('modelo'):
            log(f"  [aviso] El candidato '{cand.get('nombre')}' no tiene modelo teórico asignado. Se omite.")
            continue
        candidatos_activos.append(cand)

    n_cand = len(candidatos_activos)
    if n_cand < 3:
        raise ValueError(
            f"Se necesitan al menos 3 candidatos habilitados con archivo y modelo teórico válidos "
            f"para realizar la combinatoria (se encontraron {n_cand})."
        )
    if (evaluar_mc_sin or evaluar_mc_con) and n_cand < 4:
        if not (evaluar_simpl or evaluar_comp_sin or evaluar_comp_con):
            raise ValueError(
                f"Para evaluar cuadrados mínimos redundantes en la combinatoria se necesitan al menos "
                f"4 candidatos activos (mínimo 3 líquidos como patrones + 1 líquido como muestra; "
                f"se encontraron {n_cand})."
            )
        log("  [aviso] Cuadrados mínimos requiere al menos 4 candidatos activos (k>=3 líquidos + 1 muestra). Se omiten tareas MC.")
        evaluar_mc_sin = False
        evaluar_mc_con = False

    log(f"Iniciando análisis combinatorio con {n_cand} candidatos válidos.")
    log(f"Intervalos de evaluación: {', '.join(intervalos_str)}")

    # 1. Cargar insumos en caché
    cache_s1p = {}
    datos_corto_raw = _cargar_s1p_con_cache(ruta_corto, cache_s1p)
    datos_aire_raw = _cargar_s1p_con_cache(ruta_aire, cache_s1p)
    datos_cand_raw = {}
    for cand in candidatos_activos:
        ruta_c = os.path.join(carpeta_datos, cand['archivo'])
        datos_cand_raw[cand['nombre']] = _cargar_s1p_con_cache(ruta_c, cache_s1p)

    # 2. Generar lista de tareas de combinatoria
    tareas = []
    # A) Método completo: todas las permutaciones de 2 candidatos (P3, P4)
    if evaluar_comp_sin or evaluar_comp_con:
        for p3 in candidatos_activos:
            for p4 in candidatos_activos:
                if p3['nombre'] == p4['nombre']:
                    continue
                restantes = [c for c in candidatos_activos if c['nombre'] not in (p3['nombre'], p4['nombre'])]
                if evaluar_comp_sin:
                    tareas.append({
                        'tipo_metodo': 'completo_sin_acond',
                        'p3': p3,
                        'p4': p4,
                        'patrones_liq': [p3, p4],
                        'muestras': restantes,
                        'acondicionar': False,
                    })
                if evaluar_comp_con:
                    tareas.append({
                        'tipo_metodo': 'completo_con_acond',
                        'p3': p3,
                        'p4': p4,
                        'patrones_liq': [p3, p4],
                        'muestras': restantes,
                        'acondicionar': True,
                    })

    # B) Método simplificado: cada candidato como P3
    if evaluar_simpl:
        for p3 in candidatos_activos:
            restantes = [c for c in candidatos_activos if c['nombre'] != p3['nombre']]
            tareas.append({
                'tipo_metodo': 'simplificado',
                'p3': p3,
                'p4': None,
                'patrones_liq': [p3],
                'muestras': restantes,
                'acondicionar': False,
            })

    # C) Cuadrados mínimos redundantes: combinaciones de k líquidos (k >= 3)
    if evaluar_mc_sin or evaluar_mc_con:
        k_disponibles = list(range(3, n_cand + 1))
        if isinstance(mc_k_cfg, (list, tuple)):
            lista_k = [int(k) for k in mc_k_cfg if int(k) in k_disponibles]
        else:
            mc_k_str = str(mc_k_cfg).strip().lower()
            if mc_k_str in ('todos', 'all', ''):
                lista_k = k_disponibles
            elif mc_k_str in ('todos_m', 'todos_los_liquidos', 'max', 'm'):
                lista_k = [n_cand]
            elif mc_k_str in ('leave_one_out', 'n-1', 'm-1'):
                lista_k = [max(3, n_cand - 1)]
            else:
                try:
                    # Extraer primer entero si viene formato tipo "3 líquidos (5 patrones)"
                    k_val = int(mc_k_str.split()[0])
                    if k_val in k_disponibles:
                        lista_k = [k_val]
                    elif k_val > n_cand:
                        lista_k = [n_cand]
                    else:
                        lista_k = [3]
                except (ValueError, IndexError):
                    lista_k = k_disponibles

        if not lista_k:
            lista_k = k_disponibles

        log(f"Subconjuntos de líquidos patrón para Cuadrados Mínimos: k = {lista_k}")
        for k_liq in lista_k:
            for comb_liq in combinations(candidatos_activos, k_liq):
                patrones_liq = list(comb_liq)
                # Para MC se evalúan TODOS los candidatos (tanto los usados como patrones
                # de ajuste como los restantes) para contrastar cuánto se minimizó el error.
                if evaluar_mc_sin:
                    tareas.append({
                        'tipo_metodo': 'mc_sin_acond',
                        'p3': patrones_liq[0],
                        'p4': None,
                        'patrones_liq': patrones_liq,
                        'muestras': candidatos_activos,
                        'acondicionar': False,
                    })
                if evaluar_mc_con:
                    tareas.append({
                        'tipo_metodo': 'mc_con_acond',
                        'p3': patrones_liq[0],
                        'p4': None,
                        'patrones_liq': patrones_liq,
                        'muestras': candidatos_activos,
                        'acondicionar': True,
                    })

    total_tareas = len(tareas)
    log(f"Total de combinaciones a evaluar: {total_tareas}")

    resultados_por_combinacion = {}
    resultados_por_muestra = {cand['nombre']: [] for cand in candidatos_activos}
    cache_cal_mc = {}

    interrumpido = False
    frecs_r_global = None

    # 3. Bucle de ejecución (idéntico al pipeline de ejecutar_analisis + procesar_material)
    for idx_tarea, tarea in enumerate(tareas):
        if esta_cancelado():
            log("\n[cancelado] Análisis interrumpido por el usuario.")
            interrumpido = True
            break

        tipo_metodo = tarea['tipo_metodo']
        es_mc = tipo_metodo in ('mc_sin_acond', 'mc_con_acond')
        p3 = tarea['p3']
        p4 = tarea['p4']
        patrones_liq = tarea.get('patrones_liq', [p3] if p4 is None else [p3, p4])
        nombres_pat_liq = [c['nombre'] for c in patrones_liq]
        nombres_pat_set = set(nombres_pat_liq)
        patrones_str = " + ".join(nombres_pat_liq)

        if es_mc:
            n_pat_tot = 2 + len(patrones_liq)
            p3_nom = nombres_pat_liq[0]
            p4_nom = " + ".join(nombres_pat_liq[1:]) if len(nombres_pat_liq) > 1 else "—"
            etiqueta_metodo = (
                f"MC {n_pat_tot} pat. (acondicionado)"
                if tarea['acondicionar']
                else f"MC {n_pat_tot} pat. (original)"
            )
            combo_id = f"MC ({patrones_str}) [{etiqueta_metodo}]"
        else:
            p3_nom = p3['nombre']
            p4_nom = p4['nombre'] if p4 else "—"
            etiqueta_metodo = {
                'simplificado': "Simplificado",
                'completo_sin_acond': "Completo (original)",
                'completo_con_acond': "Completo (acondicionado)",
            }[tipo_metodo]
            combo_id = f"P3: {p3_nom} | P4: {p4_nom} [{etiqueta_metodo}]"

        progreso(idx_tarea + 1, total_tareas, f"Evaluando: {combo_id}")

        # Igual que en `cargar_calibracion` de analisis_permitividad.py:
        # la grilla de referencia de la calibración es la de 'patron3' (primer patrón líquido)
        d_p3 = datos_cand_raw[p3_nom]
        frecs = np.asarray(d_p3['Frec'], dtype=float)
        s11_p3 = d_p3['Complex']

        d_corto = datos_corto_raw
        if not mismas_frecuencias(d_p3, d_corto):
            d_corto = resamplear(frecs, d_corto)
        s11_corto = d_corto['Complex']

        d_aire = datos_aire_raw
        if not mismas_frecuencias(d_p3, d_aire):
            d_aire = resamplear(frecs, d_aire)
        s11_aire = d_aire['Complex']

        T_p3 = float(p3.get('temperatura', 25.0) or 25.0)
        Er_p3 = PATRONES_TEORICOS[p3['modelo']](frecs, T=T_p3)

        mascara_f = (frecs >= f_min_ghz * 1e9) & (frecs <= f_max_ghz * 1e9)
        frecs_r = frecs[mascara_f]
        if frecs_r_global is None:
            frecs_r_global = frecs_r

        Gn_orig = None
        Gn_eval = None
        info_acond = None
        marcas_inicio_gn = {}
        cal_mc_eval = None

        if es_mc:
            clave_mc = tuple(nombres_pat_liq)
            if clave_mc not in cache_cal_mc:
                patrones_mc = [
                    {'nombre': 'Cortocircuito', 'S11': s11_corto, 'er': None},
                    {'nombre': 'Aire', 'S11': s11_aire, 'er': np.ones_like(frecs, dtype=complex), 'es_aire': True},
                ]
                for p_liq in patrones_liq:
                    d_pl = datos_cand_raw[p_liq['nombre']]
                    if not mismas_frecuencias(d_p3, d_pl):
                        d_pl = resamplear(frecs, d_pl)
                    T_pl = float(p_liq.get('temperatura', 25.0) or 25.0)
                    er_pl = PATRONES_TEORICOS[p_liq['modelo']](frecs, T=T_pl)
                    patrones_mc.append({
                        'nombre': p_liq['nombre'],
                        'S11': d_pl['Complex'],
                        'er': er_pl,
                    })
                cal_mc_base = calibrar_minimos_cuadrados(frecs, patrones_mc)
                cache_cal_mc[clave_mc] = (patrones_mc, cal_mc_base)
            else:
                patrones_mc, cal_mc_base = cache_cal_mc[clave_mc]

            Gn_orig = cal_mc_base['Gn']
            Gn_eval = Gn_orig
            cal_mc_eval = cal_mc_base

            if tarea['acondicionar']:
                modo_ac = cfg_acond.get('modo', 'automatico')
                tipo_ac = cfg_acond.get('tipo', 'transicion_polinomica_cero')
                f1_ac = float(cfg_acond.get('f1_ghz', 0.0) or 0.0) * 1e9
                f2_ac = float(cfg_acond.get('f2_ghz', 0.5) or 0.5) * 1e9
                umbral_ac = float(cfg_acond.get('umbral_pct', 30.0) or 30.0)

                Gn_eval, info_acond = acondicionar_gn(
                    frecs, Gn_orig, modo=modo_ac, tipo=tipo_ac,
                    f1=f1_ac, f2=f2_ac, umbral_pct=umbral_ac
                )
                cal_mc_eval = reajustar_abc_con_gn_fijo(
                    frecs, patrones_mc, Gn_eval, calibracion_base=cal_mc_base
                )

            if estrategia_gn in ('minimo_gn', 'comparar_ambas', 'comparar_tres'):
                idx_auto = int(np.argmin(np.abs(Gn_eval)))
                marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['minimo_gn']] = frecs[idx_auto]
            if estrategia_gn in ('frecuencia_maxima', 'comparar_ambas', 'comparar_tres'):
                idx_clasico = int(np.argmax(frecs))
                marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['frecuencia_maxima']] = frecs[idx_clasico]
            if estrategia_gn in ('frecuencia_fija', 'comparar_tres'):
                idx_fija = int(np.argmin(np.abs(frecs - f_semilla_hz)))
                marcas_inicio_gn[f"Frecuencia fija ({frecs[idx_fija] / 1e9:.3f} GHz)"] = frecs[idx_fija]

        elif p4 is not None:
            d_p4 = datos_cand_raw[p4_nom]
            if not mismas_frecuencias(d_p3, d_p4):
                d_p4 = resamplear(frecs, d_p4)
            s11_p4 = d_p4['Complex']

            T_p4 = float(p4.get('temperatura', 25.0) or 25.0)
            Er_p4 = PATRONES_TEORICOS[p4['modelo']](frecs, T=T_p4)

            # IMPORTANTE: Mismo orden exacto de argumentos que en analisis_permitividad.py:
            # calcular_Gn(frecs, S11_patron3, S11_aire, S11_patron4, S11_corto, Er_patron3, Er_patron4)
            Gn_orig = calcular_Gn(frecs, s11_p3, s11_aire, s11_p4, s11_corto, Er_p3, Er_p4)
            Gn_eval = Gn_orig

            if tarea['acondicionar']:
                modo_ac = cfg_acond.get('modo', 'automatico')
                tipo_ac = cfg_acond.get('tipo', 'transicion_polinomica_cero')
                f1_ac = float(cfg_acond.get('f1_ghz', 0.0) or 0.0) * 1e9
                f2_ac = float(cfg_acond.get('f2_ghz', 0.5) or 0.5) * 1e9
                umbral_ac = float(cfg_acond.get('umbral_pct', 30.0) or 30.0)

                Gn_eval, info_acond = acondicionar_gn(
                    frecs, Gn_orig, modo=modo_ac, tipo=tipo_ac,
                    f1=f1_ac, f2=f2_ac, umbral_pct=umbral_ac
                )

            if estrategia_gn in ('minimo_gn', 'comparar_ambas', 'comparar_tres'):
                idx_auto = int(np.argmin(np.abs(Gn_eval)))
                marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['minimo_gn']] = frecs[idx_auto]
            if estrategia_gn in ('frecuencia_maxima', 'comparar_ambas', 'comparar_tres'):
                idx_clasico = int(np.argmax(frecs))
                marcas_inicio_gn[ETIQUETA_ESTRATEGIA_GN['frecuencia_maxima']] = frecs[idx_clasico]
            if estrategia_gn in ('frecuencia_fija', 'comparar_tres'):
                idx_fija = int(np.argmin(np.abs(frecs - f_semilla_hz)))
                marcas_inicio_gn[f"Frecuencia fija ({frecs[idx_fija] / 1e9:.3f} GHz)"] = frecs[idx_fija]

        # Evaluar líquidos con esta calibración (igual que procesar_material)
        muestras_resultados = {}
        puntajes_muestras = []

        for m in tarea['muestras']:
            m_nom = m['nombre']
            es_patron = (m_nom in nombres_pat_set) if es_mc else False
            rol = "Patrón" if es_patron else "Muestra"

            d_m = datos_cand_raw[m_nom]
            frec_dut = np.asarray(d_m['Frec'], dtype=float)
            coincide = (len(frec_dut) == len(frecs)) and np.allclose(frec_dut, frecs, atol=1.0)
            if not coincide:
                d_m = resamplear(frecs, d_m)
            s11_m = d_m['Complex']

            modelo_m = m.get('modelo')
            T_m = float(m.get('temperatura', 25.0) or 25.0)
            teorico_fn = resolver_modelo_teorico(modelo_m, T_m)
            er_teo_m = teorico_fn(frecs)

            if tipo_metodo == 'simplificado':
                er_med = get_er_DUTm(frecs, s11_m, s11_p3, s11_aire, s11_corto, Er_p3)
            elif es_mc:
                semilla = 'minimo_gn' if estrategia_gn in ('comparar_ambas', 'comparar_tres') else estrategia_gn
                er_med = get_er_DUT_minimos_cuadrados(
                    frecs, s11_m, cal_mc_eval, estrategia_semilla=semilla,
                    f_semilla_hz=f_semilla_hz
                )
            else:
                semilla = 'minimo_gn' if estrategia_gn in ('comparar_ambas', 'comparar_tres') else estrategia_gn
                er_med = get_er_DUT_completo(
                    frecs, s11_m, s11_p3, s11_aire, s11_p4, s11_corto,
                    Er_p3, Er_p4, Gn=Gn_eval, estrategia_semilla=semilla,
                    f_semilla_hz=f_semilla_hz
                )

            # Recortar a rango de frecuencias de análisis
            er_med_r = er_med[mascara_f]
            er_teo_r = er_teo_m[mascara_f]

            comparar_im = perdidas_comparables(modelo_m)
            stats_ints = calcular_error_intervalos(
                frecs_r, er_med_r, er_teo_r, intervalos, comparar_imag=comparar_im
            )

            # Calcular puntajes por intervalo y global
            puntajes_intervalos = []
            for st in stats_ints:
                err_re = st['err_re_medio']
                err_im = st['err_im_medio']
                if np.isnan(err_im):
                    score = err_re
                elif np.isnan(err_re):
                    score = err_im
                else:
                    score = (err_re + err_im) / 2.0
                st['puntaje'] = score
                if not np.isnan(score):
                    puntajes_intervalos.append(score)

            score_global_muestra = float(np.mean(puntajes_intervalos)) if puntajes_intervalos else float('nan')
            if not np.isnan(score_global_muestra):
                puntajes_muestras.append(score_global_muestra)

            etiq_teo = etiqueta_curva_teorica(modelo_m)
            datos_muestra_combo = {
                'muestra': m_nom,
                'modelo': modelo_m,
                'etiqueta_teorico': etiq_teo,
                'frecs_r': frecs_r,
                'er_medido': er_med_r,
                'er_teorico': er_teo_r,
                'intervalos': stats_ints,
                'puntaje_int1': stats_ints[0]['puntaje'] if len(stats_ints) > 0 else float('nan'),
                'puntaje_int2': stats_ints[1]['puntaje'] if len(stats_ints) > 1 else float('nan'),
                'puntaje_int3': stats_ints[2]['puntaje'] if len(stats_ints) > 2 else float('nan'),
                'puntaje_global': score_global_muestra,
                'es_patron': es_patron,
                'rol': rol,
            }
            muestras_resultados[m_nom] = datos_muestra_combo

            # Agregar a la vista agrupada por muestra
            resultados_por_muestra[m_nom].append({
                'combo_id': combo_id,
                'metodo': etiqueta_metodo,
                'es_mc': es_mc,
                'es_patron': es_patron,
                'rol': rol,
                'p3': p3_nom,
                'p4': p4_nom,
                'patrones_liquidos': nombres_pat_liq,
                'patrones_str': patrones_str,
                'modelo': modelo_m,
                'etiqueta_teorico': etiq_teo,
                'frecs_r': frecs_r,
                'puntaje_int1': datos_muestra_combo['puntaje_int1'],
                'puntaje_int2': datos_muestra_combo['puntaje_int2'],
                'puntaje_int3': datos_muestra_combo['puntaje_int3'],
                'puntaje_global': score_global_muestra,
                'er_medido': er_med_r,
                'er_teorico': er_teo_r,
                'intervalos': stats_ints,
            })

        score_promedio_total = float(np.mean(puntajes_muestras)) if puntajes_muestras else float('nan')
        puntajes_patrones = [
            m_dat['puntaje_global'] for m_dat in muestras_resultados.values()
            if m_dat.get('es_patron') and not np.isnan(m_dat['puntaje_global'])
        ]
        puntajes_restantes = [
            m_dat['puntaje_global'] for m_dat in muestras_resultados.values()
            if not m_dat.get('es_patron') and not np.isnan(m_dat['puntaje_global'])
        ]
        score_promedio_patrones = float(np.mean(puntajes_patrones)) if puntajes_patrones else float('nan')
        score_promedio_restantes = float(np.mean(puntajes_restantes)) if puntajes_restantes else float('nan')

        resultados_por_combinacion[combo_id] = {
            'combo_id': combo_id,
            'metodo': etiqueta_metodo,
            'es_mc': es_mc,
            'p3': p3_nom,
            'p4': p4_nom,
            'patrones_liquidos': nombres_pat_liq,
            'patrones_str': patrones_str,
            'frecs_r': frecs_r,
            'Gn_original': Gn_orig[mascara_f] if Gn_orig is not None else None,
            'Gn_acondicionado': (Gn_eval[mascara_f] if tarea['acondicionar'] and Gn_eval is not None else None),
            'Gn_evaluado': Gn_eval[mascara_f] if Gn_eval is not None else None,
            'info_acond': info_acond,
            'marcas_inicio_gn': marcas_inicio_gn,
            'muestras': muestras_resultados,
            'puntaje_promedio_total': score_promedio_total,
            'puntaje_promedio_patrones': score_promedio_patrones,
            'puntaje_promedio_restantes': score_promedio_restantes,
        }

    # Armar ranking global de combinaciones (ordenado de menor a mayor error promedio)
    resumen_ranking_global = sorted(
        resultados_por_combinacion.values(),
        key=lambda x: (np.isnan(x['puntaje_promedio_total']), x['puntaje_promedio_total'])
    )

    progreso(total_tareas, total_tareas, "Listo" if not interrumpido else "Cancelado")
    log("\nAnálisis combinatorio completado con éxito." if not interrumpido else "\nAnálisis combinatorio finalizado parcialmente.")

    return {
        'frecs_r': frecs_r_global if frecs_r_global is not None else np.array([]),
        'log_x': log_x,
        'intervalos': intervalos,
        'intervalos_str': intervalos_str,
        'resultados_por_combinacion': resultados_por_combinacion,
        'resultados_por_muestra': resultados_por_muestra,
        'resumen_ranking_global': resumen_ranking_global,
        'cancelado': interrumpido,
    }


def obtener_firma_patrones(item):
    """
    Devuelve (metodo, rol, frozenset(patrones)) para identificar el conjunto
    no ordenado de patrones utilizados en la calibración.
    """
    metodo = item.get('metodo', '')
    rol = item.get('rol', 'Muestra')
    if item.get('es_mc'):
        pats = item.get('patrones_liquidos') or []
        return (metodo, rol, frozenset(pats))
    p3 = item.get('p3', '')
    p4 = item.get('p4')
    if p4 and p4 != '—':
        return (metodo, rol, frozenset([p3, p4]))
    return (metodo, rol, frozenset([p3]))


def son_resultados_equivalentes(it1, it2, clave_sort='puntaje_global', tol=1e-4):
    """
    Determina si dos combinaciones dieron resultados efectivamente idénticos
    (dentro de una tolerancia numérica estricta).
    """
    s1 = it1.get(clave_sort, float('nan'))
    s2 = it2.get(clave_sort, float('nan'))
    if np.isnan(s1) and np.isnan(s2):
        pass
    elif np.isnan(s1) or np.isnan(s2):
        return False
    elif abs(s1 - s2) > tol:
        return False

    sg1 = it1.get('puntaje_global', float('nan'))
    sg2 = it2.get('puntaje_global', float('nan'))
    if not (np.isnan(sg1) and np.isnan(sg2)):
        if abs(sg1 - sg2) > tol:
            return False

    er1 = it1.get('er_medido')
    er2 = it2.get('er_medido')
    if er1 is not None and er2 is not None and hasattr(er1, 'shape') and hasattr(er2, 'shape'):
        if er1.shape == er2.shape:
            if not np.allclose(er1, er2, atol=1e-3, rtol=1e-3):
                return False

    return True


def filtrar_combinaciones_redundantes(combos_ordenados, clave_sort='puntaje_global',
                                      top_n=None, lista_completa=None):
    """
    Filtra combinaciones redundantes donde la permutación del orden de los patrones
    (p. ej. P3 y P4) arroja resultados idénticos.
    Conserva la primera aparición en el ranking y marca `orden_indistinto = True`.
    Las combinaciones recíprocas redundantes son omitidas.

    Si `top_n` es un entero, retorna a lo sumo las mejores `top_n` combinaciones únicas.
    """
    if lista_completa is None:
        lista_completa = combos_ordenados

    # Pre-identificar qué combinaciones tienen una recíproca equivalente en la lista
    combos_con_reciproca = set()
    for i, a in enumerate(lista_completa):
        firma_a = obtener_firma_patrones(a)
        if len(firma_a[2]) < 2:
            continue
        p3_a = a.get('p3')
        p4_a = a.get('p4')
        for b in lista_completa[i + 1:]:
            if obtener_firma_patrones(b) == firma_a:
                p3_b = b.get('p3')
                p4_b = b.get('p4')
                if p3_a == p4_b and p4_a == p3_b:
                    if son_resultados_equivalentes(a, b, clave_sort):
                        combos_con_reciproca.add(id(a))
                        combos_con_reciproca.add(id(b))

    resultado = []
    firmas_aceptadas = []

    for it in combos_ordenados:
        firma = obtener_firma_patrones(it)
        es_redundante = False

        if len(firma[2]) >= 2:
            for f_prev, item_prev in firmas_aceptadas:
                if f_prev == firma and son_resultados_equivalentes(it, item_prev, clave_sort):
                    es_redundante = True
                    item_prev['orden_indistinto'] = True
                    break

        if es_redundante:
            continue

        it_dict = dict(it)
        if id(it) in combos_con_reciproca:
            it_dict['orden_indistinto'] = True

        resultado.append(it_dict)
        firmas_aceptadas.append((firma, it_dict))

        if top_n is not None and len(resultado) >= top_n:
            break

    return resultado

