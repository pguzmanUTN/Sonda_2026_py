"""
combinatoria/reporte_combinatoria_pdf.py
----------------------------------------
Genera un informe en PDF con los resultados del análisis combinatorio de
patrones de calibración:
  - Portada con parámetros de configuración, intervalos de frecuencia,
    criterio de selección elegido y candidatos evaluados.
  - Tabla resumen del ranking global de combinaciones según el intervalo
    seleccionado.
  - Una sección por cada líquido analizado con:
      1. Gráfico superpuesto de las 3 mejores combinaciones (Top 3) vs.
         el modelo teórico de referencia (usando `graficar_comparacion` de
         `analisis_permitividad.py`).
      2. Tabla resumen de puntajes de las 3 mejores combinaciones.
      3. Tabla detallada de errores absolutos (medio y máximo de er' y er'')
         en los 3 intervalos de frecuencia para cada una de las 3 mejores
         combinaciones.
"""
import os
import csv
import tempfile
from datetime import datetime
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, KeepTogether, HRFlowable
)

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

from analisis_permitividad import graficar_comparacion, graficar_Gn, exportar_csv, _slug
from Patrones import etiqueta_patron
from combinatoria.analisis_combinatoria import filtrar_combinaciones_redundantes


_ESTILOS = getSampleStyleSheet()
_ESTILOS.add(ParagraphStyle(name='TituloInformeComb', parent=_ESTILOS['Title'], fontSize=19))
_ESTILOS.add(ParagraphStyle(name='SubtituloComb', parent=_ESTILOS['Normal'],
                             fontSize=11, textColor=colors.grey, alignment=TA_CENTER))
_ESTILOS.add(ParagraphStyle(name='SeccionMuestra', parent=_ESTILOS['Heading2'], fontSize=13))
_ESTILOS.add(ParagraphStyle(name='SubseccionComb', parent=_ESTILOS['Heading3'], fontSize=10.5))
_ESTILOS.add(ParagraphStyle(name='NotaComb', parent=_ESTILOS['Normal'],
                             fontSize=9, textColor=colors.grey))
_ESTILOS.add(ParagraphStyle(name='CeldaTablaComb', parent=_ESTILOS['Normal'],
                             fontSize=8.2, leading=10.2, alignment=TA_CENTER))
_ESTILOS.add(ParagraphStyle(name='CeldaTablaDestacada', parent=_ESTILOS['CeldaTablaComb'],
                             fontName='Helvetica-Bold'))
_ESTILOS.add(ParagraphStyle(name='CeldaTablaEncabezadoComb', parent=_ESTILOS['CeldaTablaComb'],
                             fontName='Helvetica-Bold', textColor=colors.white))
_ESTILOS.add(ParagraphStyle(name='CeldaMetadataComb', parent=_ESTILOS['Normal'],
                             fontSize=9.2, leading=11.5))
_ESTILOS.add(ParagraphStyle(name='CeldaMetadataLabelComb', parent=_ESTILOS['CeldaMetadataComb'],
                             fontName='Helvetica-Bold'))

_COLOR_ENCABEZADO = colors.HexColor('#2c3e50')
_COLOR_SUBENCABEZADO = colors.HexColor('#34495e')
_COLOR_FILA_PAR = colors.HexColor('#f2f2f2')
_COLOR_DESTACADO = colors.HexColor('#e8f4f8')
_ANCHO_PAGINA_UTIL = 17 * cm


def _fmt_err(valor):
    """Formatea un error absoluto (4 decimales), o 'n/a' si es NaN/None."""
    if valor is None:
        return "n/a"
    try:
        if valor != valor:  # NaN check
            return "n/a"
        return f"{valor:.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _pie_de_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2.0, 1.3 * cm,
        f"Informe de combinatoria de patrones de calibración  -  página {doc.page}"
    )
    canvas.restoreState()


def _separador():
    return HRFlowable(
        width="100%", thickness=0.6, color=colors.HexColor('#c9ced3'),
        spaceBefore=0.25 * cm, spaceAfter=0.5 * cm
    )


def _imagen_ajustada(ruta_png, ancho_max=_ANCHO_PAGINA_UTIL):
    if _PILImage is not None:
        with _PILImage.open(ruta_png) as im:
            w, h = im.size
    else:
        w, h = 1200, 900
    escala = ancho_max / w
    return Image(ruta_png, width=ancho_max, height=h * escala)


def _estilo_tabla_base(colorear_filas=True):
    estilo = [
        ('BACKGROUND', (0, 0), (-1, 0), _COLOR_ENCABEZADO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    if colorear_filas:
        estilo.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, _COLOR_FILA_PAR]))
    return TableStyle(estilo)


def _resolver_clave_criterio(criterio_str, intervalos_str):
    """
    Traduce la opción elegida por el usuario ('Promedio Global', 'Intervalo 1',
    'Intervalo 2', 'Intervalo 3' o clave directa) en:
      (clave_dict, etiqueta_legible, indice_columna_destacada)
    """
    c = (criterio_str or "").strip()
    int1_lbl = intervalos_str[0] if len(intervalos_str) > 0 else "Intervalo 1"
    int2_lbl = intervalos_str[1] if len(intervalos_str) > 1 else "Intervalo 2"
    int3_lbl = intervalos_str[2] if len(intervalos_str) > 2 else "Intervalo 3"

    if c in ("Intervalo 1", "puntaje_int1") or c.startswith("Intervalo 1"):
        return "puntaje_int1", f"Intervalo 1 ({int1_lbl})", 5
    if c in ("Intervalo 2", "puntaje_int2") or c.startswith("Intervalo 2"):
        return "puntaje_int2", f"Intervalo 2 ({int2_lbl})", 6
    if c in ("Intervalo 3", "puntaje_int3") or c.startswith("Intervalo 3"):
        return "puntaje_int3", f"Intervalo 3 ({int3_lbl})", 7
    return "puntaje_global", "Promedio Global (los 3 intervalos)", 8


def _tabla_metadata_combinatoria(config, resultados, etiqueta_criterio, omitir_redundantes=True, excluir_patrones=True):
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    intervalos_str = resultados.get('intervalos_str', [])

    met_cfg = config.get('metodos_calculo', {})
    metodos_activos = []
    if met_cfg.get('completo_sin_acond'):
        metodos_activos.append("Completo (Gn original)")
    if met_cfg.get('completo_con_acond'):
        metodos_activos.append("Completo (Gn acondicionado)")
    if met_cfg.get('mc_sin_acond'):
        metodos_activos.append("Cuadrados mínimos (Gn original)")
    if met_cfg.get('mc_con_acond'):
        metodos_activos.append("Cuadrados mínimos (Gn acondicionado)")
    if met_cfg.get('simplificado'):
        metodos_activos.append("Simplificado (3 patrones)")

    estrategia = config.get('estrategia_gn', 'minimo_gn')
    f_sem = config.get('f_semilla_ghz')
    etiquetas_estrategia = {
        'minimo_gn': "Mínimo de |Gn| en el barrido (automático)",
        'frecuencia_maxima': "Frecuencia más alta (clásico)",
        'frecuencia_fija': f"Frecuencia fija ({float(f_sem):.3f} GHz)" if f_sem else "Frecuencia fija elegida",
        'comparar_tres': f"Las 3 estrategias comparadas (auto, clásico y fija {float(f_sem):.3f} GHz)" if f_sem else "Las 3 estrategias comparadas",
    }

    filas = [
        ["Fecha de generación", fecha],
        ["Criterio de selección (Top 3)", etiqueta_criterio],
        ["Banda de frecuencias analizada", f"{config.get('f_min_ghz', 0.5)} - {config.get('f_max_ghz', 6.0)} GHz"],
        ["Intervalos de evaluación", "  |  ".join(
            [f"Int {i+1}: [{s}]" for i, s in enumerate(intervalos_str)]
        )],
        ["Métodos evaluados", ", ".join(metodos_activos) or "—"],
        ["Arranque de marcha Gn", etiquetas_estrategia.get(estrategia, estrategia)],
    ]

    if omitir_redundantes:
        filas.append(["Filtrado de redundancias", "Sí (se omiten permutaciones recíprocas con idéntico resultado; marcadas con *)"])

    if excluir_patrones:
        filas.append(["Evaluación en reporte", "Sólo muestras independientes (excluye combinaciones donde el líquido actuó como patrón)"])
    else:
        filas.append(["Evaluación en reporte", "Todas las mediciones (incluye ajuste como patrón de calibración)"])

    if met_cfg.get('mc_sin_acond') or met_cfg.get('mc_con_acond'):
        mc_k_cfg = config.get('mc_k_liquidos', 'todos')
        filas.append(["Subconjuntos MC (líquidos k)", str(mc_k_cfg)])

    if met_cfg.get('completo_con_acond') or met_cfg.get('mc_con_acond'):
        ac = config.get('acondicionar_gn', {})
        tipo_ac = ac.get('tipo', 'transicion_polinomica_cero')
        modo_ac = ac.get('modo', 'automatico')
        if modo_ac == 'automatico':
            detalle_ac = f"Modo automático (umbral = {ac.get('umbral_pct', 30.0):.1f}%)"
        else:
            detalle_ac = f"Modo manual (f1 = {ac.get('f1_ghz', 0.0):.3f} GHz, f2 = {ac.get('f2_ghz', 0.5):.3f} GHz)"
        filas.append(["Acondicionamiento de Gn", f"{tipo_ac} — {detalle_ac}"])

    corto_base = os.path.basename(config.get('archivo_corto', ''))
    aire_base = os.path.basename(config.get('archivo_aire', ''))
    filas.append(["Patrón Cortocircuito / Abierto", f"Short: {corto_base}   |   Open: {aire_base}"])

    filas_wrap = [
        [Paragraph(str(k), _ESTILOS['CeldaMetadataLabelComb']),
         Paragraph(str(v), _ESTILOS['CeldaMetadataComb'])]
        for k, v in filas
    ]

    t = Table(filas_wrap, hAlign='LEFT', colWidths=[5.8 * cm, 11.2 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), _COLOR_FILA_PAR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def _tabla_candidatos_config(config):
    """Tabla con los líquidos candidatos habilitados que participaron de la combinatoria."""
    encabezado = ["Líquido candidato", "Archivo .s1p", "Modelo teórico", "Temp. (°C)"]
    datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encabezado]]

    for c in config.get('candidatos', []):
        if not c.get('habilitado', True):
            continue
        mod_key = c.get('modelo', '')
        mod_et = etiqueta_patron(mod_key) or mod_key
        datos.append([
            Paragraph(str(c.get('nombre', '')), _ESTILOS['CeldaTablaComb']),
            Paragraph(os.path.basename(str(c.get('archivo', ''))), _ESTILOS['CeldaTablaComb']),
            Paragraph(str(mod_et), _ESTILOS['CeldaTablaComb']),
            Paragraph(f"{float(c.get('temperatura', 25.0)):.1f}", _ESTILOS['CeldaTablaComb']),
        ])

    col_widths = [3.8 * cm, 5.2 * cm, 5.8 * cm, 2.2 * cm]
    t = Table(datos, hAlign='LEFT', colWidths=col_widths)
    t.setStyle(_estilo_tabla_base())
    return t


def _tabla_top3_resumen(top_items, intervalos_str, col_destacada_idx):
    """
    Tabla compacta que muestra los puntajes (error promedio entre er' y er'')
    de las 3 mejores combinaciones en cada intervalo y el promedio global,
    resaltando la columna usada como criterio de ordenamiento.
    """
    int1_s = intervalos_str[0] if len(intervalos_str) > 0 else "Int 1"
    int2_s = intervalos_str[1] if len(intervalos_str) > 1 else "Int 2"
    int3_s = intervalos_str[2] if len(intervalos_str) > 3 - 1 else "Int 3"

    encabezado = [
        "Puesto",
        "Rol",
        "Patrón 3 (P3)",
        "Patrón 4 / Adic.",
        "Método",
        f"Score Int 1\n({int1_s})",
        f"Score Int 2\n({int2_s})",
        f"Score Int 3\n({int3_s})",
        "Score\nGlobal",
    ]
    datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encabezado]]

    for idx, it in enumerate(top_items, 1):
        rol_txt = it.get('rol', 'Muestra')
        ast = "*" if it.get('orden_indistinto') else ""
        fila = [
            Paragraph(f"#{idx}{ast}", _ESTILOS['CeldaTablaDestacada']),
            Paragraph(str(rol_txt), _ESTILOS['CeldaTablaComb']),
            Paragraph(str(it.get('p3', '—')), _ESTILOS['CeldaTablaComb']),
            Paragraph(str(it.get('p4') or '—'), _ESTILOS['CeldaTablaComb']),
            Paragraph(str(it.get('metodo', '')), _ESTILOS['CeldaTablaComb']),
            Paragraph(_fmt_err(it.get('puntaje_int1')),
                      _ESTILOS['CeldaTablaDestacada'] if col_destacada_idx == 5 else _ESTILOS['CeldaTablaComb']),
            Paragraph(_fmt_err(it.get('puntaje_int2')),
                      _ESTILOS['CeldaTablaDestacada'] if col_destacada_idx == 6 else _ESTILOS['CeldaTablaComb']),
            Paragraph(_fmt_err(it.get('puntaje_int3')),
                      _ESTILOS['CeldaTablaDestacada'] if col_destacada_idx == 7 else _ESTILOS['CeldaTablaComb']),
            Paragraph(_fmt_err(it.get('puntaje_global')),
                      _ESTILOS['CeldaTablaDestacada'] if col_destacada_idx == 8 else _ESTILOS['CeldaTablaComb']),
        ]
        datos.append(fila)

    col_widths = [1.1 * cm, 1.6 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.4 * cm]
    t = Table(datos, hAlign='LEFT', colWidths=col_widths)
    estilo = _estilo_tabla_base(colorear_filas=True)
    # Resaltar la columna del criterio seleccionado
    if 5 <= col_destacada_idx <= 8 and len(datos) > 1:
        estilo.add('BACKGROUND', (col_destacada_idx, 1), (col_destacada_idx, len(datos) - 1), _COLOR_DESTACADO)
    t.setStyle(estilo)
    return t


def _tabla_errores_detallada_top3(top_items):
    """
    Tabla con el mismo formato que `_tabla_errores` de `reporte_pdf.py`,
    desglosando Error medio er', Error máx er', Error medio er'' y Error máx er''
    en los 3 intervalos para cada una de las 3 mejores combinaciones.
    """
    encabezado = [
        "Combinación (Top 3)",
        "Intervalo",
        "Error medio\ner'",
        "Error máx\ner'",
        "Error medio\ner''",
        "Error máx\ner''",
    ]
    datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encabezado]]

    estilo_extra = []
    fila_idx = 1
    colores_bloque = [colors.white, _COLOR_FILA_PAR]

    for idx_rank, it in enumerate(top_items, 1):
        rol_it = it.get('rol', 'Muestra')
        rol_tag = f" <font color='#666666'>[{rol_it}]</font>" if rol_it else ""
        ast = "*" if it.get('orden_indistinto') else ""
        if it.get('es_mc'):
            pat_s = it.get('patrones_str') or f"{it.get('p3', '')} + {it.get('p4', '')}"
            etiqueta_combo = f"<b>#{idx_rank}{ast}</b>{rol_tag} — MC: {pat_s}<br/><i>({it['metodo']})</i>"
        else:
            etiqueta_combo = f"<b>#{idx_rank}{ast}</b>{rol_tag} — P3: {it['p3']} | P4: {it['p4'] or '—'}<br/><i>({it['metodo']})</i>"
        items_int = it.get('intervalos', [])
        num_ints = len(items_int)
        if num_ints == 0:
            continue

        fila_inicio = fila_idx
        color_fondo = colores_bloque[(idx_rank - 1) % 2]

        for i, e in enumerate(items_int):
            interv_str = e.get('intervalo_str', '')
            texto_combo = etiqueta_combo if i == 0 else ""
            datos.append([
                Paragraph(texto_combo, _ESTILOS['CeldaTablaComb']),
                Paragraph(interv_str, _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(e.get('err_re_medio')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(e.get('err_re_max')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(e.get('err_im_medio')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(e.get('err_im_max')), _ESTILOS['CeldaTablaComb']),
            ])
            fila_idx += 1

        estilo_extra.append(('BACKGROUND', (0, fila_inicio), (-1, fila_idx - 1), color_fondo))
        if num_ints > 1:
            estilo_extra.append(('SPAN', (0, fila_inicio), (0, fila_inicio + num_ints - 1)))
            estilo_extra.append(('VALIGN', (0, fila_inicio), (0, fila_inicio + num_ints - 1), 'MIDDLE'))
        estilo_extra.append(('LINEBELOW', (0, fila_idx - 1), (-1, fila_idx - 1), 0.7, colors.HexColor('#9aa0a6')))

    ancho_combo = 5.2 * cm
    ancho_intervalo = 3.0 * cm
    ancho_col_err = (_ANCHO_PAGINA_UTIL - ancho_combo - ancho_intervalo) / 4.0
    col_widths = [ancho_combo, ancho_intervalo] + [ancho_col_err] * 4

    estilo_base = [
        ('BACKGROUND', (0, 0), (-1, 0), _COLOR_SUBENCABEZADO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]

    t = Table(datos, hAlign='LEFT', colWidths=col_widths)
    t.setStyle(TableStyle(estilo_base + estilo_extra))
    return t


def generar_reporte_combinatoria_pdf(ruta_pdf, config, resultados, criterio_orden="Promedio Global",
                                     top_n=3, omitir_redundantes=True, excluir_patrones=True,
                                     carpeta_exportar=None, limitar_eje_y=True):
    """
    Construye el informe PDF con las `top_n` (por defecto 3) mejores combinaciones
    para cada líquido evaluado, ordenadas de menor a mayor error promedio según
    `criterio_orden` ('Promedio Global', 'Intervalo 1', 'Intervalo 2' o 'Intervalo 3').

    Si `omitir_redundantes=True`, cuando dos combinaciones utilicen el mismo par/conjunto
    de patrones permutados con idéntico resultado (p. ej. P3 Agua y P4 Isopropanol vs
    P3 Isopropanol y P4 Agua), se incluye únicamente la primera de ellas en el ranking
    y se coloca un asterisco (*) aclarando que el orden resultó indistinto.

    Si `excluir_patrones=True`, se omiten del ranking de un líquido aquellas combinaciones
    donde dicho líquido formó parte del set de patrones de calibración (evaluación in-sample),
    considerando exclusivamente su desempeño como muestra independiente de validación.

    Si `carpeta_exportar` no es None, exporta en esa carpeta las figuras .png de cada líquido
    (er_top3_<muestra>.png) y las tablas de permitividad compleja en formato .csv (tabla_top3_<muestra>.csv),
    además del ranking resumen en CSV (ranking_top3_<criterio>.csv).
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_pdf)), exist_ok=True)
    if carpeta_exportar:
        os.makedirs(os.path.abspath(carpeta_exportar), exist_ok=True)

    intervalos = resultados.get('intervalos', [])
    intervalos_str = resultados.get('intervalos_str', [])
    log_x = resultados.get('log_x', True)

    clave_sort, etiqueta_criterio, col_destacada_idx = _resolver_clave_criterio(
        criterio_orden, intervalos_str
    )

    doc = SimpleDocTemplate(
        ruta_pdf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Informe de Selección Óptima de Patrones (Combinatoria)",
    )
    historia = []

    # ------------------------------------------------------------------
    # 1. Portada / Encabezado
    # ------------------------------------------------------------------
    historia.append(Spacer(1, 0.2 * cm))
    historia.append(Paragraph(
        "Informe de Selección Óptima de Patrones (Combinatoria)",
        _ESTILOS['TituloInformeComb']
    ))
    historia.append(Paragraph(
        f"Top {top_n} mejores combinaciones por líquido — Ordenado por: {etiqueta_criterio}",
        _ESTILOS['SubtituloComb']
    ))
    historia.append(Spacer(1, 0.5 * cm))
    historia.append(_tabla_metadata_combinatoria(config, resultados, etiqueta_criterio,
                                                omitir_redundantes=omitir_redundantes,
                                                excluir_patrones=excluir_patrones))
    historia.append(Spacer(1, 0.4 * cm))
    historia.append(Paragraph("Líquidos candidatos evaluados", _ESTILOS['SubseccionComb']))
    historia.append(Spacer(1, 0.15 * cm))
    historia.append(_tabla_candidatos_config(config))
    historia.append(Spacer(1, 0.35 * cm))
    historia.append(Paragraph(
        "El puntaje (Score) en cada intervalo se define como el promedio entre el error absoluto "
        "medio de la parte real (er') y el error absoluto medio de las pérdidas (er''): "
        "Score = (e_abs(er') + e_abs(er'')) / 2.",
        _ESTILOS['NotaComb']
    ))
    historia.append(_separador())

    # ------------------------------------------------------------------
    # 2. Una sección por cada líquido evaluado (Top 3 combinaciones)
    # ------------------------------------------------------------------
    resultados_por_muestra = resultados.get('resultados_por_muestra', {})
    nombres_muestras = list(resultados_por_muestra.keys())
    todos_los_top_items = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx_m, muestra_nom in enumerate(nombres_muestras):
            lista_combos = resultados_por_muestra.get(muestra_nom, [])
            if not lista_combos:
                continue

            if excluir_patrones:
                lista_combos_eval = [x for x in lista_combos if not x.get('es_patron')]
                if not lista_combos_eval:
                    lista_combos_eval = lista_combos
            else:
                lista_combos_eval = lista_combos

            # Ordenar de menor a mayor error en el intervalo seleccionado
            combos_ordenados = sorted(
                lista_combos_eval,
                key=lambda x: (np.isnan(x.get(clave_sort, float('nan'))), x.get(clave_sort, float('inf')))
            )
            if omitir_redundantes:
                top_items = filtrar_combinaciones_redundantes(
                    combos_ordenados, clave_sort=clave_sort, top_n=top_n, lista_completa=lista_combos_eval
                )
            else:
                top_items = combos_ordenados[:top_n]

            if not top_items:
                continue

            todos_los_top_items[muestra_nom] = top_items
            slug_m = _slug(muestra_nom)

            primer_item = top_items[0]
            frecs_r = primer_item.get('frecs_r', resultados.get('frecs_r'))
            er_teo = primer_item.get('er_teorico')
            etiq_teo = primer_item.get('etiqueta_teorico', 'Teórico')

            # Construir diccionario de las mejores curvas medidas para superponer
            er_medido_top = {}
            for rank_i, it in enumerate(top_items, 1):
                val_score = it.get(clave_sort, float('nan'))
                score_txt = f"{val_score:.3f}" if not np.isnan(val_score) else "n/a"
                rol_str = f" [{it.get('rol', 'Muestra')}]"
                ast = "*" if it.get('orden_indistinto') else ""
                if it.get('es_mc'):
                    pat_s = it.get('patrones_str') or f"{it.get('p3', '')} + {it.get('p4', '')}"
                    etiqueta_curva = f"#{rank_i}{ast}: MC({pat_s}){rol_str} [{it['metodo']}] (err={score_txt})"
                else:
                    p4_txt = f", P4={it['p4']}" if it.get('p4') and it['p4'] != "—" else ""
                    etiqueta_curva = f"#{rank_i}{ast}: P3={it['p3']}{p4_txt}{rol_str} [{it['metodo']}] (err={score_txt})"
                er_medido_top[etiqueta_curva] = it['er_medido']

            titulo_graf = f"{muestra_nom} — Top {len(top_items)} mejores combinaciones ({etiqueta_criterio})"
            fig = graficar_comparacion(
                frecs_r,
                er_medido_top,
                er_teo,
                titulo=titulo_graf,
                archivo_salida=None,
                log_x=log_x,
                etiqueta_teorico=etiq_teo,
                limitar_eje_y=limitar_eje_y,
            )

            # Marcar divisiones de intervalos y sombrear suavemente el intervalo criterio si aplica
            for _, fb in intervalos[:-1]:
                for ax in fig.axes:
                    ax.axvline(fb / 1e9, color='#888888', linestyle=':', lw=1.1)

            if clave_sort in ('puntaje_int1', 'puntaje_int2', 'puntaje_int3'):
                idx_int_sel = {'puntaje_int1': 0, 'puntaje_int2': 1, 'puntaje_int3': 2}[clave_sort]
                if idx_int_sel < len(intervalos):
                    fa_sel, fb_sel = intervalos[idx_int_sel]
                    for ax in fig.axes:
                        ax.axvspan(fa_sel / 1e9, fb_sel / 1e9, color='#fff59d', alpha=0.22, zorder=0)

            if carpeta_exportar:
                ruta_png = os.path.join(carpeta_exportar, f"er_top3_{slug_m}.png")
                fig.savefig(ruta_png, dpi=150)

                # Exportar CSV de permitividad compleja (teórico + top_n medidos)
                columnas_csv = {}
                if er_teo is not None:
                    columnas_csv['teorico'] = er_teo
                for rank_i, it in enumerate(top_items, 1):
                    ast = "_indistinto" if it.get('orden_indistinto') else ""
                    if it.get('es_mc'):
                        pat_tag = _slug(it.get('patrones_str', 'mc'))
                        etiq_col = f"top{rank_i}_mc_{pat_tag}{ast}"
                    else:
                        p3_tag = _slug(it.get('p3', 'p3'))
                        p4_tag = _slug(it.get('p4', 'none')) if it.get('p4') and it['p4'] != "—" else "none"
                        etiq_col = f"top{rank_i}_p3_{p3_tag}_p4_{p4_tag}{ast}"
                    columnas_csv[etiq_col] = it['er_medido']

                ruta_csv = os.path.join(carpeta_exportar, f"tabla_top3_{slug_m}.csv")
                exportar_csv(ruta_csv, frecs_r, columnas_csv)
            else:
                ruta_png = os.path.join(tmpdir, f"top3_muestra_{idx_m}.png")
                fig.savefig(ruta_png, dpi=150)

            # Armar bloque PDF para esta muestra
            bloque_graf = [
                Paragraph(f"Muestra evaluada: {muestra_nom}", _ESTILOS['SeccionMuestra']),
                Spacer(1, 0.15 * cm),
                _imagen_ajustada(ruta_png, ancho_max=15.5 * cm),
                Spacer(1, 0.25 * cm),
            ]
            historia.append(KeepTogether(bloque_graf))

            hay_indistinto = any(it.get('orden_indistinto') for it in top_items)
            bloque_tablas = [
                Paragraph(
                    f"Resumen de puntajes — Top {len(top_items)} combinaciones para {muestra_nom} "
                    f"(ordenado por: {etiqueta_criterio})",
                    _ESTILOS['NotaComb']
                ),
                Spacer(1, 0.12 * cm),
                _tabla_top3_resumen(top_items, intervalos_str, col_destacada_idx),
            ]
            if hay_indistinto:
                bloque_tablas.append(Spacer(1, 0.10 * cm))
                bloque_tablas.append(Paragraph(
                    "<b>* Nota:</b> El orden de los patrones (P3 y P4) resultó indistinto al arrojar idénticos resultados; "
                    "se contabiliza solo la primera combinación y se omite la recíproca redundante.",
                    _ESTILOS['NotaComb']
                ))

            bloque_tablas.extend([
                Spacer(1, 0.25 * cm),
                Paragraph(
                    f"Desglose de error absoluto por intervalo — Top {len(top_items)} ({muestra_nom})",
                    _ESTILOS['NotaComb']
                ),
                Spacer(1, 0.12 * cm),
                _tabla_errores_detallada_top3(top_items),
            ])
            if hay_indistinto:
                bloque_tablas.append(Spacer(1, 0.10 * cm))
                bloque_tablas.append(Paragraph(
                    "<b>*</b> Combinación con orden de patrones permutables de idéntico resultado.",
                    _ESTILOS['NotaComb']
                ))

            historia.append(KeepTogether(bloque_tablas))

            if idx_m < len(nombres_muestras) - 1:
                historia.append(_separador())

        if carpeta_exportar and todos_los_top_items:
            ruta_resumen_csv = os.path.join(carpeta_exportar, f"ranking_top3_{_slug(etiqueta_criterio)}.csv")
            with open(ruta_resumen_csv, "w", newline="", encoding="utf-8") as f_res:
                w_res = csv.writer(f_res)
                int1_s = intervalos_str[0] if len(intervalos_str) > 0 else "Int 1"
                int2_s = intervalos_str[1] if len(intervalos_str) > 1 else "Int 2"
                int3_s = intervalos_str[2] if len(intervalos_str) > 2 else "Int 3"
                w_res.writerow([
                    "Muestra", "Puesto", "Rol", "Patron_3", "Patron_4_o_Adicionales",
                    "Metodo", f"Score_Int1_{int1_s}", f"Score_Int2_{int2_s}",
                    f"Score_Int3_{int3_s}", "Score_Global", "Orden_Indistinto"
                ])
                for m_nom, m_top in todos_los_top_items.items():
                    for r_idx, it_top in enumerate(m_top, 1):
                        w_res.writerow([
                            m_nom, r_idx, it_top.get('rol', 'Muestra'),
                            it_top.get('p3', ''), it_top.get('p4', '') or '—',
                            it_top.get('metodo', ''),
                            f"{it_top.get('puntaje_int1', float('nan')):.6f}",
                            f"{it_top.get('puntaje_int2', float('nan')):.6f}",
                            f"{it_top.get('puntaje_int3', float('nan')):.6f}",
                            f"{it_top.get('puntaje_global', float('nan')):.6f}",
                            "Si" if it_top.get('orden_indistinto') else "No"
                        ])

        doc.build(historia, onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)

    return ruta_pdf


def _generar_reporte_combinacion_comparativa_pdf(ruta_pdf, config, resultados, combo_id_a, combo_id_b,
                                                 carpeta_exportar=None, limitar_eje_y=True):
    """
    Construye un informe PDF comparativo entre dos combinaciones de calibración (Combo A vs Combo B):
      - Metadatos comparativos de ambas calibraciones y sus puestos en el ranking global.
      - Gráficos de diagnóstico de conductancia Gn(f) (si aplica a alguna o ambas).
      - Tabla clasificatoria resumen comparando el desempeño en cada muestra/líquido.
      - Gráficos comparativos de permitividad con curvas superpuestas:
        Medido Combo A vs Medido Combo B vs Modelo Teórico de Referencia.
      - Tablas detalladas de errores absolutos en los 3 intervalos para ambas combinaciones.
      - Exportación opcional de figuras .png comparativas y tablas .csv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_pdf)), exist_ok=True)
    if carpeta_exportar:
        os.makedirs(os.path.abspath(carpeta_exportar), exist_ok=True)

    combo_a = resultados.get('resultados_por_combinacion', {}).get(combo_id_a)
    combo_b = resultados.get('resultados_por_combinacion', {}).get(combo_id_b)
    if not combo_a:
        raise ValueError(f"No se encontró la combinación A '{combo_id_a}' en los resultados.")
    if not combo_b:
        raise ValueError(f"No se encontró la combinación B '{combo_id_b}' en los resultados.")

    intervalos = resultados.get('intervalos', [])
    intervalos_str = resultados.get('intervalos_str', [])
    log_x = resultados.get('log_x', True)

    doc = SimpleDocTemplate(
        ruta_pdf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Informe Comparativo - {combo_id_a} vs {combo_id_b}",
    )
    historia = []

    # 1. Portada / Título
    historia.append(Spacer(1, 0.2 * cm))
    historia.append(Paragraph(
        "Informe Comparativo de Calibraciones y Mediciones",
        _ESTILOS['TituloInformeComb']
    ))
    historia.append(Paragraph(
        f"Combinación A: <b>{combo_id_a}</b> &nbsp;&nbsp;vs.&nbsp;&nbsp; Combinación B: <b>{combo_id_b}</b>",
        _ESTILOS['SubtituloComb']
    ))
    historia.append(Spacer(1, 0.45 * cm))

    # Ranking y metadatos comparativos
    ranking_global = resultados.get('resumen_ranking_global', [])
    total_combos = len(ranking_global)
    puesto_a = next((i for i, c in enumerate(ranking_global, 1) if c.get('combo_id') == combo_id_a), None)
    puesto_a_str = f"#{puesto_a} de {total_combos}" if puesto_a else "—"
    puesto_b = next((i for i, c in enumerate(ranking_global, 1) if c.get('combo_id') == combo_id_b), None)
    puesto_b_str = f"#{puesto_b} de {total_combos}" if puesto_b else "—"

    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    corto_base = os.path.basename(config.get('archivo_corto', ''))
    aire_base = os.path.basename(config.get('archivo_aire', ''))

    score_tot_a = combo_a.get('puntaje_promedio_total')
    score_pat_a = combo_a.get('puntaje_promedio_patrones')
    score_rest_a = combo_a.get('puntaje_promedio_restantes')

    score_tot_b = combo_b.get('puntaje_promedio_total')
    score_pat_b = combo_b.get('puntaje_promedio_patrones')
    score_rest_b = combo_b.get('puntaje_promedio_restantes')

    info_a = combo_a.get('info_acond')
    info_b = combo_b.get('info_acond')
    acond_a_str = f"{info_a.get('tipo')} ({info_a.get('modo', '')})" if (info_a and info_a.get('tipo')) else "Original (sin acond.)"
    acond_b_str = f"{info_b.get('tipo')} ({info_b.get('modo', '')})" if (info_b and info_b.get('tipo')) else "Original (sin acond.)"

    filas_meta = [
        ["Fecha de generación", fecha, fecha],
        ["Combinación", combo_id_a, combo_id_b],
        ["Puesto en ranking global", puesto_a_str, puesto_b_str],
        ["Método de cálculo", str(combo_a.get('metodo', '—')), str(combo_b.get('metodo', '—'))],
        ["Patrón Corto / Abierto", f"Short: {corto_base} | Open: {aire_base}", f"Short: {corto_base} | Open: {aire_base}"],
        ["Patrones líquidos", str(combo_a.get('patrones_str', '—')), str(combo_b.get('patrones_str', '—'))],
        ["Banda analizada", f"{config.get('f_min_ghz', 0.5)} - {config.get('f_max_ghz', 6.0)} GHz", f"{config.get('f_min_ghz', 0.5)} - {config.get('f_max_ghz', 6.0)} GHz"],
        ["Score Promedio Global", _fmt_err(score_tot_a), _fmt_err(score_tot_b)],
        ["Score Patrones (ajuste)", _fmt_err(score_pat_a), _fmt_err(score_pat_b)],
        ["Score Muestras (validación)", _fmt_err(score_rest_a), _fmt_err(score_rest_b)],
        ["Acondicionamiento Gn", acond_a_str, acond_b_str],
    ]

    filas_wrap = [
        [
            Paragraph(f"<b>{k}</b>", _ESTILOS['CeldaMetadataLabelComb']),
            Paragraph(str(v_a), _ESTILOS['CeldaMetadataComb']),
            Paragraph(str(v_b), _ESTILOS['CeldaMetadataComb'])
        ]
        for k, v_a, v_b in filas_meta
    ]
    encabezado_meta = [
        Paragraph("<b>Parámetro</b>", _ESTILOS['CeldaTablaEncabezadoComb']),
        Paragraph(f"<b>Combinación A</b><br/><font size=7.5>{combo_id_a}</font>", _ESTILOS['CeldaTablaEncabezadoComb']),
        Paragraph(f"<b>Combinación B</b><br/><font size=7.5>{combo_id_b}</font>", _ESTILOS['CeldaTablaEncabezadoComb']),
    ]
    t_meta = Table([encabezado_meta] + filas_wrap, hAlign='LEFT', colWidths=[4.4 * cm, 6.3 * cm, 6.3 * cm])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), _COLOR_ENCABEZADO),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, _COLOR_FILA_PAR]),
    ]))
    historia.append(t_meta)
    historia.append(Spacer(1, 0.4 * cm))

    slug_a = _slug(combo_id_a)[:30].rstrip('_')
    slug_b = _slug(combo_id_b)[:30].rstrip('_')

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Diagnóstico de Gn(f) si aplica
        frecs_r = combo_a.get('frecs_r', resultados.get('frecs_r'))
        gn_a = combo_a.get('Gn_original')
        gn_b = combo_b.get('Gn_original')

        if (gn_a is not None and len(frecs_r) == len(gn_a)) or (gn_b is not None and len(frecs_r) == len(gn_b)):
            historia.append(Paragraph("Diagnóstico de Conductancia Normalizada Gn(f)", _ESTILOS['SubseccionComb']))
            historia.append(Spacer(1, 0.15 * cm))

            if gn_a is not None:
                fig_gn_a = graficar_Gn(
                    frecs_r, gn_a, archivo_salida=None, log_x=log_x,
                    marcas_inicio=combo_a.get('marcas_inicio_gn'),
                    Gn_acondicionado=combo_a.get('Gn_acondicionado'),
                    info_acond=info_a
                )
                fig_gn_a.axes[0].set_title(f"Gn(f) — Combinación A: {combo_id_a}")
                r_gn_a = os.path.join(carpeta_exportar if carpeta_exportar else tmpdir, f"Gn_{slug_a}.png")
                fig_gn_a.savefig(r_gn_a, dpi=150)
                historia.append(KeepTogether([
                    _imagen_ajustada(r_gn_a, ancho_max=15.5 * cm),
                    Spacer(1, 0.25 * cm)
                ]))

            if gn_b is not None:
                fig_gn_b = graficar_Gn(
                    frecs_r, gn_b, archivo_salida=None, log_x=log_x,
                    marcas_inicio=combo_b.get('marcas_inicio_gn'),
                    Gn_acondicionado=combo_b.get('Gn_acondicionado'),
                    info_acond=info_b
                )
                fig_gn_b.axes[0].set_title(f"Gn(f) — Combinación B: {combo_id_b}")
                r_gn_b = os.path.join(carpeta_exportar if carpeta_exportar else tmpdir, f"Gn_{slug_b}.png")
                fig_gn_b.savefig(r_gn_b, dpi=150)
                historia.append(KeepTogether([
                    _imagen_ajustada(r_gn_b, ancho_max=15.5 * cm),
                    Spacer(1, 0.25 * cm)
                ]))

        # 3. Tabla resumen comparativa de todos los líquidos evaluados
        muestras_dict_a = combo_a.get('muestras', {})
        muestras_dict_b = combo_b.get('muestras', {})
        muestras_union = list(dict.fromkeys(list(muestras_dict_a.keys()) + list(muestras_dict_b.keys())))

        historia.append(Paragraph("Resumen Comparativo de Líquidos Evaluados", _ESTILOS['SubseccionComb']))
        historia.append(Spacer(1, 0.15 * cm))

        encab_m = [
            "Líquido / Muestra",
            "Rol Combo A",
            "Score Global\nCombo A",
            "Rol Combo B",
            "Score Global\nCombo B",
            "Comparativa\n(Mejor score)",
        ]
        datos_tabla_m = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encab_m]]

        for m_nom in muestras_union:
            m_a = muestras_dict_a.get(m_nom, {})
            m_b = muestras_dict_b.get(m_nom, {})
            rol_a = m_a.get('rol', '—') if m_a else 'No eval.'
            rol_b = m_b.get('rol', '—') if m_b else 'No eval.'
            sc_a = m_a.get('puntaje_global') if m_a else None
            sc_b = m_b.get('puntaje_global') if m_b else None

            diff_str = "—"
            if sc_a is not None and sc_b is not None and not np.isnan(sc_a) and not np.isnan(sc_b):
                d = sc_a - sc_b
                if d < -1e-4:
                    diff_str = f"<b>Combo A</b> ({d:+.4f})"
                elif d > 1e-4:
                    diff_str = f"<b>Combo B</b> ({d:+.4f})"
                else:
                    diff_str = "Empate"

            datos_tabla_m.append([
                Paragraph(str(m_nom), _ESTILOS['CeldaTablaComb']),
                Paragraph(str(rol_a), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(sc_a), _ESTILOS['CeldaTablaDestacada'] if (sc_a is not None and sc_b is not None and sc_a <= sc_b) else _ESTILOS['CeldaTablaComb']),
                Paragraph(str(rol_b), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(sc_b), _ESTILOS['CeldaTablaDestacada'] if (sc_a is not None and sc_b is not None and sc_b < sc_a) else _ESTILOS['CeldaTablaComb']),
                Paragraph(diff_str, _ESTILOS['CeldaTablaComb']),
            ])

        t_res_m = Table(datos_tabla_m, hAlign='LEFT', colWidths=[3.8 * cm, 2.5 * cm, 2.7 * cm, 2.5 * cm, 2.7 * cm, 2.8 * cm])
        t_res_m.setStyle(_estilo_tabla_base(colorear_filas=True))
        historia.append(KeepTogether([t_res_m, Spacer(1, 0.4 * cm)]))
        historia.append(_separador())

        if carpeta_exportar and muestras_union:
            ruta_resumen_csv = os.path.join(carpeta_exportar, f"resumen_comparativo_{slug_a}_{slug_b}.csv")
            with open(ruta_resumen_csv, "w", newline="", encoding="utf-8") as f_res:
                w_res = csv.writer(f_res)
                w_res.writerow([
                    "Liquido", "Rol_A", "Score_Global_A", "Rol_B", "Score_Global_B", "Diferencia_A_menos_B"
                ])
                for m_nom in muestras_union:
                    m_a = muestras_dict_a.get(m_nom, {})
                    m_b = muestras_dict_b.get(m_nom, {})
                    sc_a = m_a.get('puntaje_global', float('nan')) if m_a else float('nan')
                    sc_b = m_b.get('puntaje_global', float('nan')) if m_b else float('nan')
                    diff_val = (sc_a - sc_b) if (not np.isnan(sc_a) and not np.isnan(sc_b)) else float('nan')
                    w_res.writerow([
                        m_nom,
                        m_a.get('rol', '—') if m_a else 'No eval.',
                        f"{sc_a:.6f}",
                        m_b.get('rol', '—') if m_b else 'No eval.',
                        f"{sc_b:.6f}",
                        f"{diff_val:.6f}",
                    ])

        # 4. Sección detallada comparativa por líquido
        for idx_m, m_nom in enumerate(muestras_union):
            m_a = muestras_dict_a.get(m_nom)
            m_b = muestras_dict_b.get(m_nom)
            er_med_a = m_a.get('er_medido') if m_a else None
            er_med_b = m_b.get('er_medido') if m_b else None
            er_teo = m_a.get('er_teorico') if m_a else (m_b.get('er_teorico') if m_b else None)
            etiq_teo = m_a.get('etiqueta_teorico') if m_a else (m_b.get('etiqueta_teorico') if m_b else 'Teórico')
            frecs_m = m_a.get('frecs_r', frecs_r) if m_a else m_b.get('frecs_r', frecs_r)
            slug_m = _slug(m_nom)

            rol_a_str = f" [A: {m_a.get('rol')}]" if (m_a and m_a.get('rol')) else ""
            rol_b_str = f" [B: {m_b.get('rol')}]" if (m_b and m_b.get('rol')) else ""

            curvas = {}
            if er_med_a is not None:
                curvas[f"A ({combo_a.get('metodo', 'Medido')}): {combo_id_a}"] = er_med_a
            if er_med_b is not None:
                curvas[f"B ({combo_b.get('metodo', 'Medido')}): {combo_id_b}"] = er_med_b

            titulo_graf = f"{m_nom}{rol_a_str}{rol_b_str} — Comparación A vs B vs {etiq_teo}"
            fig_m = graficar_comparacion(
                frecs_m,
                curvas,
                er_teo,
                titulo=titulo_graf,
                archivo_salida=None,
                log_x=log_x,
                etiqueta_teorico=etiq_teo,
                limitar_eje_y=limitar_eje_y,
            )

            for _, fb in intervalos[:-1]:
                for ax in fig_m.axes:
                    ax.axvline(fb / 1e9, color='#888888', linestyle=':', lw=1.1)

            if carpeta_exportar:
                ruta_png_m = os.path.join(carpeta_exportar, f"er_comp_{slug_a}_{slug_b}_{slug_m}.png")
                fig_m.savefig(ruta_png_m, dpi=150)

                # Exportar CSV de permitividad medido A, medido B vs teórico
                cols_m = {}
                if er_teo is not None:
                    cols_m['teorico'] = er_teo
                if er_med_a is not None:
                    cols_m[f"medido_A_{slug_a}"] = er_med_a
                if er_med_b is not None:
                    cols_m[f"medido_B_{slug_b}"] = er_med_b
                ruta_csv_m = os.path.join(carpeta_exportar, f"tabla_comp_{slug_a}_{slug_b}_{slug_m}.csv")
                exportar_csv(ruta_csv_m, frecs_m, cols_m)
            else:
                ruta_png_m = os.path.join(tmpdir, f"comp_muestra_{idx_m}.png")
                fig_m.savefig(ruta_png_m, dpi=150)

            # Tabla comparativa de errores por intervalo
            encab_err = ["Calibración", "Intervalo", "Error medio er'", "Error máx er'", "Error medio er''", "Error máx er''"]
            datos_err = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encab_err]]

            if m_a:
                for st in m_a.get('intervalos', []):
                    datos_err.append([
                        Paragraph(f"<b>Combo A</b><br/><font size=7 color='#1b4f72'>{combo_id_a}</font>", _ESTILOS['CeldaTablaComb']),
                        Paragraph(str(st.get('intervalo_str', '')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_re_medio')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_re_max')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_im_medio')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_im_max')), _ESTILOS['CeldaTablaComb']),
                    ])
            if m_b:
                for st in m_b.get('intervalos', []):
                    datos_err.append([
                        Paragraph(f"<b>Combo B</b><br/><font size=7 color='#6c3483'>{combo_id_b}</font>", _ESTILOS['CeldaTablaComb']),
                        Paragraph(str(st.get('intervalo_str', '')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_re_medio')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_re_max')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_im_medio')), _ESTILOS['CeldaTablaComb']),
                        Paragraph(_fmt_err(st.get('err_im_max')), _ESTILOS['CeldaTablaComb']),
                    ])

            t_err = Table(datos_err, hAlign='LEFT', colWidths=[3.7 * cm, 2.7 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
            t_err.setStyle(_estilo_tabla_base(colorear_filas=True))

            bloque_m = [
                Paragraph(f"Comparativa para líquido: {m_nom}", _ESTILOS['SeccionMuestra']),
                Spacer(1, 0.15 * cm),
                _imagen_ajustada(ruta_png_m, ancho_max=15.5 * cm),
                Spacer(1, 0.2 * cm),
                Paragraph(f"Desglose comparativo de error absoluto por intervalo — {m_nom}", _ESTILOS['NotaComb']),
                Spacer(1, 0.1 * cm),
                t_err,
                Spacer(1, 0.3 * cm),
            ]
            historia.append(KeepTogether(bloque_m))
            if idx_m < len(muestras_union) - 1:
                historia.append(_separador())

        doc.build(historia, onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)

    return ruta_pdf


def generar_reporte_combinacion_pdf(ruta_pdf, config, resultados, combo_id,
                                    combo_id_comparar=None,
                                    carpeta_exportar=None,
                                    limitar_eje_y=True):
    """
    Construye un informe PDF detallado para una combinación de calibración específica,
    o un informe comparativo entre dos combinaciones si se especifica `combo_id_comparar`.
      - Metadatos de la calibración y puesto en el ranking global.
      - Gráfico de diagnóstico de conductancia Gn(f) (si aplica).
      - Tabla clasificatoria de todas las muestras/líquidos medidos bajo esta calibración.
      - Gráficos comparativos de permitividad (Medido vs Teórico) y tabla de errores
        en los 3 intervalos para cada líquido evaluado.

    Si `carpeta_exportar` no es None, exporta en esa carpeta la figura de Gn (si aplica),
    las figuras de cada líquido y las tablas de permitividad en .csv.
    """
    if combo_id_comparar and combo_id_comparar in resultados.get('resultados_por_combinacion', {}):
        return _generar_reporte_combinacion_comparativa_pdf(
            ruta_pdf, config, resultados, combo_id, combo_id_comparar,
            carpeta_exportar=carpeta_exportar, limitar_eje_y=limitar_eje_y
        )
    os.makedirs(os.path.dirname(os.path.abspath(ruta_pdf)), exist_ok=True)
    if carpeta_exportar:
        os.makedirs(os.path.abspath(carpeta_exportar), exist_ok=True)

    combo_info = resultados.get('resultados_por_combinacion', {}).get(combo_id)
    if not combo_info:
        raise ValueError(f"No se encontró la combinación '{combo_id}' en los resultados.")

    intervalos = resultados.get('intervalos', [])
    intervalos_str = resultados.get('intervalos_str', [])
    log_x = resultados.get('log_x', True)

    doc = SimpleDocTemplate(
        ruta_pdf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Informe de Combinación - {combo_id}",
    )
    historia = []

    # 1. Portada / Título
    historia.append(Spacer(1, 0.2 * cm))
    historia.append(Paragraph(
        "Informe de Calibración y Mediciones por Combinación",
        _ESTILOS['TituloInformeComb']
    ))
    historia.append(Paragraph(
        f"Calibración evaluada: {combo_id}",
        _ESTILOS['SubtituloComb']
    ))
    historia.append(Spacer(1, 0.45 * cm))

    # Ranking y metadatos de esta combinación
    ranking_global = resultados.get('resumen_ranking_global', [])
    total_combos = len(ranking_global)
    puesto = next((i for i, c in enumerate(ranking_global, 1) if c.get('combo_id') == combo_id), None)
    puesto_str = f"#{puesto} de {total_combos}" if puesto else "—"

    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    corto_base = os.path.basename(config.get('archivo_corto', ''))
    aire_base = os.path.basename(config.get('archivo_aire', ''))

    score_tot = combo_info.get('puntaje_promedio_total')
    score_pat = combo_info.get('puntaje_promedio_patrones')
    score_rest = combo_info.get('puntaje_promedio_restantes')

    filas_meta = [
        ["Fecha de generación", fecha],
        ["Combinación evaluada", combo_id],
        ["Puesto en ranking global", puesto_str],
        ["Método de cálculo", str(combo_info.get('metodo', '—'))],
        ["Patrón Cortocircuito / Abierto", f"Short: {corto_base}   |   Open: {aire_base}"],
        ["Patrones líquidos", str(combo_info.get('patrones_str', '—'))],
        ["Banda de frecuencias analizada", f"{config.get('f_min_ghz', 0.5)} - {config.get('f_max_ghz', 6.0)} GHz"],
        ["Intervalos de evaluación", "  |  ".join(
            [f"Int {i+1}: [{s}]" for i, s in enumerate(intervalos_str)]
        )],
        ["Score Promedio Global (todos)", _fmt_err(score_tot)],
        ["Score Promedio Patrones (ajuste in-sample)", _fmt_err(score_pat)],
        ["Score Promedio Muestras (validación)", _fmt_err(score_rest)],
    ]

    info_acond = combo_info.get('info_acond')
    if info_acond and info_acond.get('tipo'):
        filas_meta.append(["Acondicionamiento de Gn", f"{info_acond.get('tipo')} ({info_acond.get('modo', '')})"])

    filas_wrap = [
        [Paragraph(str(k), _ESTILOS['CeldaMetadataLabelComb']),
         Paragraph(str(v), _ESTILOS['CeldaMetadataComb'])]
        for k, v in filas_meta
    ]
    t_meta = Table(filas_wrap, hAlign='LEFT', colWidths=[5.8 * cm, 11.2 * cm])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), _COLOR_FILA_PAR),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    historia.append(t_meta)
    historia.append(Spacer(1, 0.4 * cm))

    slug_combo = _slug(combo_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Diagnóstico de Gn(f) si está disponible
        frecs_r = combo_info.get('frecs_r', resultados.get('frecs_r'))
        Gn_orig = combo_info.get('Gn_original')
        if Gn_orig is not None and len(frecs_r) == len(Gn_orig):
            historia.append(Paragraph("Diagnóstico de Conductancia Normalizada Gn(f)", _ESTILOS['SubseccionComb']))
            historia.append(Spacer(1, 0.15 * cm))
            Gn_acond = combo_info.get('Gn_acondicionado')
            marcas = combo_info.get('marcas_inicio_gn')
            fig_gn = graficar_Gn(
                frecs_r, Gn_orig, archivo_salida=None, log_x=log_x,
                marcas_inicio=marcas, Gn_acondicionado=Gn_acond, info_acond=info_acond
            )
            if carpeta_exportar:
                ruta_gn_png = os.path.join(carpeta_exportar, f"Gn_{slug_combo}.png")
            else:
                ruta_gn_png = os.path.join(tmpdir, "diagnostico_gn.png")
            fig_gn.savefig(ruta_gn_png, dpi=150)
            historia.append(KeepTogether([
                _imagen_ajustada(ruta_gn_png, ancho_max=15.5 * cm),
                Spacer(1, 0.3 * cm)
            ]))

        # 3. Tabla resumen de todos los líquidos evaluados bajo esta combinación
        muestras_dict = combo_info.get('muestras', {})
        int1_s = intervalos_str[0] if len(intervalos_str) > 0 else "Int 1"
        int2_s = intervalos_str[1] if len(intervalos_str) > 1 else "Int 2"
        int3_s = intervalos_str[2] if len(intervalos_str) > 2 else "Int 3"

        historia.append(Paragraph("Resumen de Líquidos Evaluados con esta Calibración", _ESTILOS['SubseccionComb']))
        historia.append(Spacer(1, 0.15 * cm))

        encab_m = [
            "Líquido / Muestra",
            "Rol",
            f"Score Int 1\n({int1_s})",
            f"Score Int 2\n({int2_s})",
            f"Score Int 3\n({int3_s})",
            "Score\nGlobal",
        ]
        datos_tabla_m = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encab_m]]

        for m_nom, m_dat in muestras_dict.items():
            rol_txt = m_dat.get('rol', 'Muestra')
            datos_tabla_m.append([
                Paragraph(str(m_nom), _ESTILOS['CeldaTablaComb']),
                Paragraph(str(rol_txt), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(m_dat.get('puntaje_int1')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(m_dat.get('puntaje_int2')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(m_dat.get('puntaje_int3')), _ESTILOS['CeldaTablaComb']),
                Paragraph(_fmt_err(m_dat.get('puntaje_global')), _ESTILOS['CeldaTablaDestacada']),
            ])

        t_res_m = Table(datos_tabla_m, hAlign='LEFT', colWidths=[4.2 * cm, 2.8 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
        t_res_m.setStyle(_estilo_tabla_base(colorear_filas=True))
        historia.append(KeepTogether([t_res_m, Spacer(1, 0.4 * cm)]))
        historia.append(_separador())

        if carpeta_exportar and muestras_dict:
            ruta_resumen_csv = os.path.join(carpeta_exportar, f"resumen_muestras_{slug_combo}.csv")
            with open(ruta_resumen_csv, "w", newline="", encoding="utf-8") as f_res:
                w_res = csv.writer(f_res)
                w_res.writerow([
                    "Liquido", "Rol", f"Score_Int1_{int1_s}", f"Score_Int2_{int2_s}",
                    f"Score_Int3_{int3_s}", "Score_Global"
                ])
                for m_nom, m_dat in muestras_dict.items():
                    w_res.writerow([
                        m_nom, m_dat.get('rol', 'Muestra'),
                        f"{m_dat.get('puntaje_int1', float('nan')):.6f}",
                        f"{m_dat.get('puntaje_int2', float('nan')):.6f}",
                        f"{m_dat.get('puntaje_int3', float('nan')):.6f}",
                        f"{m_dat.get('puntaje_global', float('nan')):.6f}",
                    ])

        # 4. Sección detallada por líquido
        nombres_m = list(muestras_dict.keys())
        for idx_m, m_nom in enumerate(nombres_m):
            m_dat = muestras_dict[m_nom]
            er_med = m_dat.get('er_medido')
            er_teo = m_dat.get('er_teorico')
            etiq_teo = m_dat.get('etiqueta_teorico', 'Teórico')
            rol_txt = m_dat.get('rol', 'Muestra')
            slug_m = _slug(m_nom)

            titulo_graf = f"{m_nom} [{rol_txt}] — Medido ({combo_info.get('metodo')}) vs {etiq_teo}"
            fig_m = graficar_comparacion(
                m_dat.get('frecs_r', frecs_r),
                {f"Medido ({combo_info.get('metodo')})": er_med},
                er_teo,
                titulo=titulo_graf,
                archivo_salida=None,
                log_x=log_x,
                etiqueta_teorico=etiq_teo,
                limitar_eje_y=limitar_eje_y,
            )

            for _, fb in intervalos[:-1]:
                for ax in fig_m.axes:
                    ax.axvline(fb / 1e9, color='#888888', linestyle=':', lw=1.1)

            if carpeta_exportar:
                ruta_png_m = os.path.join(carpeta_exportar, f"er_{slug_combo}_{slug_m}.png")
                fig_m.savefig(ruta_png_m, dpi=150)

                # Exportar CSV de permitividad medido vs teórico
                frecs_m = m_dat.get('frecs_r', frecs_r)
                cols_m = {}
                if er_teo is not None:
                    cols_m['teorico'] = er_teo
                cols_m[f"medido_{_slug(combo_info.get('metodo', 'medido'))}"] = er_med
                ruta_csv_m = os.path.join(carpeta_exportar, f"tabla_{slug_combo}_{slug_m}.csv")
                exportar_csv(ruta_csv_m, frecs_m, cols_m)
            else:
                ruta_png_m = os.path.join(tmpdir, f"muestra_combo_{idx_m}.png")
                fig_m.savefig(ruta_png_m, dpi=150)

            encab_err = ["Intervalo", "Error medio er'", "Error máx er'", "Error medio er''", "Error máx er''"]
            datos_err = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezadoComb']) for h in encab_err]]
            for st in m_dat.get('intervalos', []):
                datos_err.append([
                    Paragraph(str(st.get('intervalo_str', '')), _ESTILOS['CeldaTablaComb']),
                    Paragraph(_fmt_err(st.get('err_re_medio')), _ESTILOS['CeldaTablaComb']),
                    Paragraph(_fmt_err(st.get('err_re_max')), _ESTILOS['CeldaTablaComb']),
                    Paragraph(_fmt_err(st.get('err_im_medio')), _ESTILOS['CeldaTablaComb']),
                    Paragraph(_fmt_err(st.get('err_im_max')), _ESTILOS['CeldaTablaComb']),
                ])
            t_err = Table(datos_err, hAlign='LEFT', colWidths=[4.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])
            t_err.setStyle(_estilo_tabla_base(colorear_filas=True))

            bloque_m = [
                Paragraph(f"Medición de líquido: {m_nom} <font color='#666666'>[{rol_txt}]</font>", _ESTILOS['SeccionMuestra']),
                Spacer(1, 0.15 * cm),
                _imagen_ajustada(ruta_png_m, ancho_max=15.5 * cm),
                Spacer(1, 0.2 * cm),
                Paragraph(f"Desglose de error absoluto por intervalo — {m_nom}", _ESTILOS['NotaComb']),
                Spacer(1, 0.1 * cm),
                t_err,
                Spacer(1, 0.3 * cm),
            ]
            historia.append(KeepTogether(bloque_m))
            if idx_m < len(nombres_m) - 1:
                historia.append(_separador())

        doc.build(historia, onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)

    return ruta_pdf


