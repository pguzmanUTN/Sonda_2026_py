"""
reporte_pdf.py
---------------
Arma un PDF de informe con los graficos y tablas que produce
`analisis_permitividad.py`: portada con los datos de la medicion, chequeo
de calibracion y un bloque por cada material analizado (grafico + tabla de
error si hay modelo teorico, o solo el grafico si no lo hay).

Usa `reportlab`, que es una libreria pura de Python (no hace falta instalar
LaTeX/pandoc en la computadora donde se corra el analisis).

Requiere: pip install reportlab pillow --break-system-packages

Este archivo no depende de `funciones.py` ni de `Patrones.py`: solo arma el
PDF a partir de los datos que le pasa `analisis_permitividad.py` (graficos
ya guardados como .png, y diccionarios con las tablas/errores). Esto separa
"calcular" de "armar el informe", asi que se puede reutilizar para otro
tipo de reporte sin tocar el resto del proyecto.
"""
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, KeepTogether,
                                 HRFlowable)

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None


_ESTILOS = getSampleStyleSheet()
_ESTILOS.add(ParagraphStyle(name='TituloInforme', parent=_ESTILOS['Title'], fontSize=20))
_ESTILOS.add(ParagraphStyle(name='Subtitulo', parent=_ESTILOS['Normal'],
                             fontSize=11, textColor=colors.grey, alignment=TA_CENTER))
_ESTILOS.add(ParagraphStyle(name='NombreMaterial', parent=_ESTILOS['Heading2']))
_ESTILOS.add(ParagraphStyle(name='Nota', parent=_ESTILOS['Normal'],
                             fontSize=9, textColor=colors.grey))
_ESTILOS.add(ParagraphStyle(name='AvisoCancelado', parent=_ESTILOS['Normal'],
                             fontSize=10.5, fontName='Helvetica-Bold',
                             textColor=colors.HexColor('#b34700'), alignment=TA_CENTER))
# Estilos para celdas de tabla: a diferencia de un string plano (lo que
# se usaba antes), un Paragraph SI hace salto de linea automatico si el
# texto no entra en el ancho de columna disponible. Sin esto, una
# etiqueta larga (un modelo teorico con nombre largo, una ruta de
# archivo, "Medido (completo, 4 patrones, Automatico)", etc.) no se
# recorta ni ajusta: se dibuja igual, se sale de su celda y queda
# superpuesta con el texto de al lado -- se ve roto aunque el dato en si
# este bien.
_ESTILOS.add(ParagraphStyle(name='CeldaTabla', parent=_ESTILOS['Normal'],
                             fontSize=8.5, leading=10.5, alignment=TA_CENTER))
_ESTILOS.add(ParagraphStyle(name='CeldaTablaEncabezado', parent=_ESTILOS['CeldaTabla'],
                             fontName='Helvetica-Bold', textColor=colors.white))
_ESTILOS.add(ParagraphStyle(name='CeldaMetadata', parent=_ESTILOS['Normal'],
                             fontSize=9.5, leading=12))
_ESTILOS.add(ParagraphStyle(name='CeldaMetadataLabel', parent=_ESTILOS['CeldaMetadata'],
                             fontName='Helvetica-Bold'))

_COLOR_ENCABEZADO = colors.HexColor('#2c3e50')
_COLOR_FILA_PAR = colors.HexColor('#f2f2f2')
_ANCHO_PAGINA_UTIL = 17 * cm  # A4 menos margenes de 2cm a cada lado

# Paleta de las cajas de advertencia (ver `_caja_advertencia`), indexada
# por el 'nivel' que declara cada modelo teorico en
# Patrones.INFO_PATRONES: 'info' (neutro), 'aviso' (ambar) y 'critico'
# (rojo). Cada entrada es (color_del_borde_y_titulo, color_de_fondo).
_COLORES_ADVERTENCIA = {
    'info': (colors.HexColor('#2c5d8f'), colors.HexColor('#eaf2fa')),
    'aviso': (colors.HexColor('#8a6100'), colors.HexColor('#fdf4e0')),
    'critico': (colors.HexColor('#a32020'), colors.HexColor('#fdeaea')),
}

_ESTILOS.add(ParagraphStyle(name='TituloAdvertencia', parent=_ESTILOS['Normal'],
                             fontSize=10, leading=13, fontName='Helvetica-Bold'))
_ESTILOS.add(ParagraphStyle(name='TextoAdvertencia', parent=_ESTILOS['Normal'],
                             fontSize=8.8, leading=11.5))


def _fmt_error(valor):
    """Formatea un error absoluto (adimensional), o 'n/a' si es NaN.
    """
    if valor is None:
        return "n/a"
    try:
        if valor != valor:  # NaN != NaN, sin necesidad de importar math
            return "n/a"
        return f"{valor:.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _caja_advertencia(titulo, texto, nivel='aviso', ancho=_ANCHO_PAGINA_UTIL):
    """
    Caja destacada (con borde de color y fondo suave) para las
    advertencias de los modelos teoricos -- por ejemplo, avisar de que
    fuentes salio un modelo armado a mano (la acetona), o que un liquido
    no polar no tiene ni puede tener relajacion (ciclohexano, silicona).

    `texto` puede traer saltos de linea y vinetas: cada parrafo se
    convierte en su propio Paragraph, para que el salto de linea
    automatico funcione y las vinetas queden una por renglon.
    """
    color_borde, color_fondo = _COLORES_ADVERTENCIA.get(
        nivel, _COLORES_ADVERTENCIA['aviso'])

    # Nada de simbolos tipo "⚠" aca: las fuentes estandar del PDF
    # (Helvetica con WinAnsiEncoding) no tienen ese glifo y ReportLab lo
    # dibuja como un cuadrado negro. Se usa una palabra, que ademas se
    # entiende igual si el PDF se imprime en blanco y negro.
    prefijo = {'critico': "ATENCIÓN", 'aviso': "AVISO", 'info': "NOTA"}.get(
        nivel, "AVISO")
    contenido = [Paragraph(f"{prefijo} — {titulo}", _ESTILOS['TituloAdvertencia'])]
    for parrafo in str(texto).split("\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        contenido.append(Spacer(1, 0.12 * cm))
        contenido.append(Paragraph(parrafo, _ESTILOS['TextoAdvertencia']))

    t = Table([[contenido]], colWidths=[ancho], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), color_fondo),
        ('BOX', (0, 0), (-1, -1), 0.9, color_borde),
        ('LINEBEFORE', (0, 0), (0, -1), 3.2, color_borde),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def _pie_de_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2.0, 1.3 * cm,
                              f"Informe de permitividad dielectrica  -  pagina {doc.page}")
    canvas.restoreState()


def _imagen_ajustada(ruta_png, ancho_max=_ANCHO_PAGINA_UTIL):
    """Flowable Image escalado para que entre en `ancho_max`, manteniendo
    la relacion de aspecto original del PNG."""
    if _PILImage is not None:
        with _PILImage.open(ruta_png) as im:
            w, h = im.size
    else:
        w, h = 1200, 900  # fallback razonable si no esta Pillow
    escala = ancho_max / w
    return Image(ruta_png, width=ancho_max, height=h * escala)


def _estilo_tabla(colorear_filas=True):
    estilo = [
        ('BACKGROUND', (0, 0), (-1, 0), _COLOR_ENCABEZADO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    if colorear_filas:
        estilo.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, _COLOR_FILA_PAR]))
    return TableStyle(estilo)


def _tabla_errores(errores, titulo_col0="Metodo"):
    """
    errores: dict {etiqueta: list of dicts o dict}
    donde cada entrada en la lista de intervalos contiene:
      'intervalo_str', 'err_re_medio', 'err_re_max', 'err_im_medio', 'err_im_max'
    """
    encabezado = [
        titulo_col0,
        "Intervalo",
        "Error medio\ner'",
        "Error max\ner'",
        "Error medio\ner''",
        "Error max\ner''"
    ]
    datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezado']) for h in encabezado]]

    estilo_extra = []
    fila_idx = 1
    colores_bloque = [colors.white, _COLOR_FILA_PAR]
    idx_metodo = 0

    for etiqueta, items in errores.items():
        if isinstance(items, dict):
            items = [{'intervalo_str': "Banda completa", **items}]

        num_ints = len(items)
        fila_inicio_metodo = fila_idx
        color_fondo = colores_bloque[idx_metodo % 2]
        idx_metodo += 1

        for i, e in enumerate(items):
            interv_str = e.get('intervalo_str', '')
            texto_metodo = str(etiqueta) if i == 0 else ""
            datos.append([
                Paragraph(texto_metodo, _ESTILOS['CeldaTabla']),
                Paragraph(interv_str, _ESTILOS['CeldaTabla']),
                Paragraph(_fmt_error(e.get('err_re_medio')), _ESTILOS['CeldaTabla']),
                Paragraph(_fmt_error(e.get('err_re_max')), _ESTILOS['CeldaTabla']),
                Paragraph(_fmt_error(e.get('err_im_medio')), _ESTILOS['CeldaTabla']),
                Paragraph(_fmt_error(e.get('err_im_max')), _ESTILOS['CeldaTabla']),
            ])
            fila_idx += 1

        # Fondo del metodo agrupado
        estilo_extra.append(('BACKGROUND', (0, fila_inicio_metodo), (-1, fila_idx - 1), color_fondo))

        if num_ints > 1:
            estilo_extra.append(('SPAN', (0, fila_inicio_metodo), (0, fila_inicio_metodo + num_ints - 1)))
            estilo_extra.append(('VALIGN', (0, fila_inicio_metodo), (0, fila_inicio_metodo + num_ints - 1), 'MIDDLE'))

        # Linea divisoria inferior para separar metodos
        estilo_extra.append(('LINEBELOW', (0, fila_idx - 1), (-1, fila_idx - 1), 0.7, colors.HexColor('#9aa0a6')))

    ancho_metodo = 4.8 * cm
    ancho_intervalo = 3.2 * cm
    ancho_col_err = (_ANCHO_PAGINA_UTIL - ancho_metodo - ancho_intervalo) / 4  # ~2.3 cm cada una
    col_widths = [ancho_metodo, ancho_intervalo] + [ancho_col_err] * 4

    estilo_base = [
        ('BACKGROUND', (0, 0), (-1, 0), _COLOR_ENCABEZADO),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]

    t = Table(datos, hAlign='LEFT', colWidths=col_widths)
    t.setStyle(TableStyle(estilo_base + estilo_extra))
    return t


def _tabla_resumen(frecs_ghz, columnas):
    """
    frecs_ghz : lista/array de frecuencias (GHz) a mostrar como filas.
    columnas  : dict {titulo: ndarray complejo} evaluado en esas frecuencias
                (mismo largo que frecs_ghz).
    """
    encabezado = ["f (GHz)"] + list(columnas.keys())
    datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezado']) for h in encabezado]]
    for i, f in enumerate(frecs_ghz):
        fila = [Paragraph(f"{f:.2f}", _ESTILOS['CeldaTabla'])]
        for col in columnas.values():
            v = col[i]
            fila.append(Paragraph(f"{v.real:.2f} - {abs(v.imag):.2f}j", _ESTILOS['CeldaTabla']))
        datos.append(fila)

    ancho_primera = 2.2 * cm
    n_col = max(len(columnas), 1)
    ancho_resto = (_ANCHO_PAGINA_UTIL - ancho_primera) / n_col
    t = Table(datos, hAlign='LEFT',
              colWidths=[ancho_primera] + [ancho_resto] * n_col)
    t.setStyle(_estilo_tabla())
    return t


def _separador():
    """
    Linea fina horizontal para separar una seccion (chequeo de calibracion
    o un material) de la siguiente, sin forzar un salto de pagina completo.

    Antes cada seccion terminaba siempre con PageBreak(), que arranca la
    seccion siguiente si o si en una pagina en blanco nueva. Si la seccion
    anterior no llegaba a llenar la pagina (p.ej. el chequeo de
    calibracion, que es corto), eso dejaba un pedazo grande de pagina
    vacio antes del salto. Con este separador, cada seccion sigue
    inmediatamente despues de la anterior si entra en lo que queda de
    pagina, y solo pasa a la pagina siguiente cuando de verdad no entra
    (ver KeepTogether en `generar_reporte_pdf`), que es lo que realmente
    evita el "hueco" en blanco.
    """
    return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor('#c9ced3'),
                       spaceBefore=0.25 * cm, spaceAfter=0.6 * cm)


def _fila_temperatura(etiqueta, valor):
    """Fila (etiqueta, texto) para una temperatura de metadata.get(...):
    si `valor` es None (clave ausente, p.ej. un informe generado con una
    version anterior de analisis_permitividad.py), muestra "no
    especificada" en vez de romper con un KeyError/TypeError."""
    texto = f"{valor:.1f} \u00b0C" if valor is not None else "no especificada"
    return [etiqueta, texto]


def _formatear_info_acond_pdf(info_acond):
    """Genera el parrafo explicativo de las frecuencias de transicion / corte
    para ubicar debajo del grafico de Gn(f) en el informe PDF."""
    if not info_acond:
        return ""
    tipo = info_acond.get('tipo', '')
    modo = info_acond.get('modo', '')
    disparo = info_acond.get('disparo_detectado')

    nombres_tipo = {
        'cero_clamping': "anulaci\u00f3n a cero (cero clamping)",
        'transicion_polinomica_cero': "transici\u00f3n cuadr\u00e1tica (f\u00b2) hacia cero",
        'tapering': "tapering con ventana de coseno alzado",
    }
    tipo_str = nombres_tipo.get(tipo, tipo)

    if modo == 'automatico':
        if disparo is False:
            aviso = info_acond.get('aviso') or "No se detect\u00f3 variaci\u00f3n mayor al umbral. Curva intacta."
            return f"<b>Acondicionamiento de Gn (autom\u00e1tico):</b> {aviso}"

        if tipo == 'cero_clamping':
            fc = info_acond.get('f_corte_aplicada', 0.0) or 0.0
            fc_str = f"{fc/1e9:.4f} GHz ({fc/1e6:.2f} MHz)" if fc < 1e9 else f"{fc/1e9:.4f} GHz"
            return (
                f"<b>Acondicionamiento de Gn:</b> Se aplic\u00f3 {tipo_str} en modo autom\u00e1tico. "
                f"Frecuencia de corte calculada autom\u00e1ticamente: "
                f"<b>f<sub>corte</sub> = {fc_str}</b> (valores de Gn anulados a cero por debajo de esta frecuencia)."
            )
        else:
            f1 = info_acond.get('f1_aplicada', 0.0) or 0.0
            f2 = info_acond.get('f2_aplicada', 0.0) or 0.0
            f1_str = f"{f1/1e9:.4f} GHz ({f1/1e6:.2f} MHz)" if (0 < f1 < 1e9) else f"{f1/1e9:.4f} GHz"
            f2_str = f"{f2/1e9:.4f} GHz ({f2/1e6:.2f} MHz)" if f2 < 1e9 else f"{f2/1e9:.4f} GHz"
            return (
                f"<b>Acondicionamiento de Gn:</b> Se aplic\u00f3 {tipo_str} en modo autom\u00e1tico. "
                f"Frecuencias de transici\u00f3n calculadas autom\u00e1ticamente: "
                f"<b>f<sub>1</sub> = {f1_str}</b> (Gn = 0) y "
                f"<b>f<sub>2</sub> = {f2_str}</b> (empalme continuo con la curva original calculada)."
            )
    else:
        # Modo manual
        if tipo == 'cero_clamping':
            fc = info_acond.get('f_corte_aplicada', 0.0) or 0.0
            fc_str = f"{fc/1e9:.4f} GHz ({fc/1e6:.2f} MHz)" if fc < 1e9 else f"{fc/1e9:.4f} GHz"
            return (
                f"<b>Acondicionamiento de Gn:</b> Se aplic\u00f3 {tipo_str} en modo manual con "
                f"frecuencia de corte <b>f<sub>corte</sub> = {fc_str}</b>."
            )
        else:
            f1 = info_acond.get('f1_aplicada', 0.0) or 0.0
            f2 = info_acond.get('f2_aplicada', 0.0) or 0.0
            f1_str = f"{f1/1e9:.4f} GHz"
            f2_str = f"{f2/1e9:.4f} GHz ({f2/1e6:.2f} MHz)" if f2 < 1e9 else f"{f2/1e9:.4f} GHz"
            return (
                f"<b>Acondicionamiento de Gn:</b> Se aplic\u00f3 {tipo_str} en modo manual con "
                f"frecuencias de transici\u00f3n <b>f<sub>1</sub> = {f1_str}</b> y "
                f"<b>f<sub>2</sub> = {f2_str}</b>."
            )


def _tabla_metadata(metadata):
    fecha = metadata.get('fecha') or datetime.now().strftime("%d/%m/%Y %H:%M")
    usar_completo = metadata.get('usar_metodo_completo', True)
    etiqueta_p3 = metadata.get('patron3_etiqueta') or metadata.get('patron3_modelo') or "Patron 3"
    filas = [
        ["Fecha de generacion", fecha],
        _fila_temperatura(f"Patron 3 ({etiqueta_p3}) - T (CAL)",
                           metadata.get('patron3_temperatura')),
    ]
    filtro_sal = metadata.get('filtro_salida') or {}
    if filtro_sal.get('habilitado'):
        tipo_fs = filtro_sal.get('tipo', 'savitzky_golay')
        map_nomb = {
            'pasabajos': f"Pasabajos (orden={filtro_sal.get('pb_orden', 2)}, fc={filtro_sal.get('pb_fcorte', 0.1):.2f})",
            'savitzky_golay': f"Savitzky-Golay (ventana={filtro_sal.get('sg_window', 15)}, orden={filtro_sal.get('sg_polyorder', 3)})",
            'mediana_movil': f"Mediana m\u00f3vil (ventana={filtro_sal.get('med_window', 15)})",
            'splines': f"Splines (s={filtro_sal.get('spline_s', 0.10):.3f})",
        }
        nom_det = map_nomb.get(tipo_fs, tipo_fs)
        if filtro_sal.get('modo_rango') == 'rango':
            nom_det += f" [{filtro_sal.get('f1_ghz', 0.5):.2f} - {filtro_sal.get('f2_ghz', 6.0):.2f} GHz]"
        filas.append(["Filtro en salida (er', er'')", nom_det])

    if usar_completo:
        etiqueta_p4 = metadata.get('patron4_etiqueta') or metadata.get('patron4_modelo') or "Patron 4"
        filas.append(_fila_temperatura(f"Patron 4 ({etiqueta_p4}) - T (CAL)",
                                        metadata.get('patron4_temperatura')))
        estrategia = metadata.get('estrategia_gn') or "minimo_gn"
        f_sem = metadata.get('f_semilla_ghz')
        etiquetas_estrategia = {
            'minimo_gn': "m\u00ednimo de |Gn| en el barrido (autom\u00e1tico)",
            'frecuencia_maxima': "frecuencia m\u00e1s alta (cl\u00e1sico)",
            'frecuencia_fija': f"frecuencia fija ({float(f_sem):.3f} GHz)" if f_sem else "frecuencia fija elegida",
            'comparar_ambas': "ambas estrategias, comparadas (automático vs. clásico)",
            'comparar_tres': f"las 3 estrategias comparadas (auto, clásico y fija {float(f_sem):.3f} GHz)" if f_sem else "las 3 estrategias comparadas",
        }
        etiqueta_estrategia = etiquetas_estrategia.get(estrategia, estrategia)
        filas.append(["Arranque de Gn (m\u00e9todo completo)", etiqueta_estrategia])

        cfg_acond = metadata.get('acondicionar_gn') or {}
        if cfg_acond.get('habilitado'):
            modo_ac = cfg_acond.get('modo', 'automatico')
            tipo_ac = cfg_acond.get('tipo', 'transicion_polinomica_cero')
            detalle_ac = f"{tipo_ac}, {modo_ac}"
            if tipo_ac in ('pasabajos', 'transicion_pasabajos'):
                detalle_ac += f", ord={cfg_acond.get('pb_orden', 2)}, fc={cfg_acond.get('pb_fcorte', 0.10)}"
            filas.append(["Acondicionamiento de Gn", f"Habilitado ({detalle_ac})"])
    else:
        filas.append(["M\u00e9todo completo", "No usado (solo m\u00e9todo simplificado)"])
        cfg_acond = metadata.get('acondicionar_gn') or {}
        if cfg_acond.get('habilitado') and metadata.get('minimos_cuadrados'):
            modo_ac = cfg_acond.get('modo', 'automatico')
            tipo_ac = cfg_acond.get('tipo', 'transicion_polinomica_cero')
            detalle_ac = f"{tipo_ac}, {modo_ac}"
            if tipo_ac in ('pasabajos', 'transicion_pasabajos'):
                detalle_ac += f", ord={cfg_acond.get('pb_orden', 2)}, fc={cfg_acond.get('pb_fcorte', 0.10)}"
            filas.append(["Acondicionamiento de Gn", f"Habilitado ({detalle_ac})"])

    mc = metadata.get('minimos_cuadrados')
    if mc:
        nombres_mc = ", ".join(mc.get('nombres', []))
        filas.append([
            "Calibraci\u00f3n por cuadrados m\u00ednimos",
            f"{mc.get('n_patrones')} patrones ({nombres_mc})",
        ])

    cfg_filtro_pat = metadata.get('filtrar_patrones_sg') or metadata.get('filtrar_patrones') or {}
    if cfg_filtro_pat.get('habilitado'):
        tipo_f_pat = cfg_filtro_pat.get('tipo', cfg_filtro_pat.get('tipo_filtro', 'savitzky_golay'))
        en_pat = ", ".join(cfg_filtro_pat.get('aplicar_en', []))
        if tipo_f_pat == 'pasabajos':
            ord_pat = cfg_filtro_pat.get('pb_orden', 2)
            fc_pat = cfg_filtro_pat.get('pb_fcorte', 0.10)
            filas.append(["Filtro en patrones", f"Pasabajos Butterworth (orden={ord_pat}, fc={fc_pat:.2f}) en: {en_pat}"])
        else:
            w_pat = cfg_filtro_pat.get('sg_window', 15)
            p_pat = cfg_filtro_pat.get('sg_polyorder', 3)
            filas.append(["Filtro en patrones", f"Savitzky-Golay (N={w_pat}, p={p_pat}) en: {en_pat}"])

    filas.append(["Banda analizada", f"{metadata['f_min_ghz']} - {metadata['f_max_ghz']} GHz"])
    if metadata.get('intervalos_error'):
        ints_str = ", ".join([f"[{fa/1e9:.2f} - {fb/1e9:.2f}] GHz" for fa, fb in metadata['intervalos_error']])
        filas.append(["Intervalos de error", ints_str])
    for clave, archivo in metadata.get('archivos_calibracion', {}).items():
        nombre_archivo = os.path.basename(archivo) if archivo else archivo
        filas.append([f"Patron de calibracion: {clave}", nombre_archivo])

    filas_wrap = [
        [Paragraph(str(etiqueta), _ESTILOS['CeldaMetadataLabel']),
         Paragraph(str(valor), _ESTILOS['CeldaMetadata'])]
        for etiqueta, valor in filas
    ]

    t = Table(filas_wrap, hAlign='LEFT', colWidths=[7 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef2f5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def _seccion_minimos_cuadrados(chequeo_mc):
    """
    Arma los elementos de ReportLab (lista para envolver o agregar al story) con el
    diagnostico de la calibracion redundante por cuadrados minimos:
    explicacion breve, grafico de residuo por patron, tabla resumen de residuo
    de S11 y tabla de validacion cruzada con error absoluto desglosado en los
    3 intervalos de frecuencia.
    """
    n_pat = chequeo_mc.get('n_patrones', 0)
    dof = chequeo_mc.get('grados_libertad', 0)
    residuo_rms = chequeo_mc.get('residuo_rms', {})
    validacion = chequeo_mc.get('validacion_cruzada') or {}
    errores_vc = chequeo_mc.get('errores_validacion_cruzada') or {}

    elementos = [
        Paragraph(f"Calibraci\u00f3n redundante por cuadrados m\u00ednimos ({n_pat} patrones)",
                  _ESTILOS['NombreMaterial']),
        Paragraph(
            f"Ajuste conjunto de los 4 par\u00e1metros del modelo de la sonda "
            f"(3 de la red bilineal + Gn de radiaci\u00f3n) sobre los {n_pat} patrones "
            f"de calibraci\u00f3n a la vez ({dof} grados de libertad reales por "
            f"frecuencia). Cuando hay 5 o m\u00e1s patrones el sistema queda "
            f"sobredeterminado: el error de medici\u00f3n se promedia entre todos y el "
            f"residuo de cada patr\u00f3n mide qu\u00e9 tan consistente es con los dem\u00e1s.",
            _ESTILOS['Normal']),
        Spacer(1, 0.25 * cm),
    ]
    fig_res = chequeo_mc.get('figura_residuos')
    if fig_res and os.path.isfile(fig_res):
        elementos.append(_imagen_ajustada(fig_res))
        elementos.append(Spacer(1, 0.2 * cm))

    info_acond_mc = chequeo_mc.get('info_acond_mc')
    if chequeo_mc.get('acondicionado') and info_acond_mc and info_acond_mc.get('tipo'):
        texto_acond_mc = _formatear_info_acond_pdf(info_acond_mc)
        if texto_acond_mc:
            elementos.append(Paragraph(
                texto_acond_mc + " <i>(Posteriormente se reajustaron los par\u00e1metros a, b, c "
                "fijando el Gn acondicionado).</i>",
                _ESTILOS['Normal']))
            elementos.append(Spacer(1, 0.2 * cm))

    if residuo_rms:
        tiene_vc = bool(validacion) and dof > 0
        if tiene_vc:
            encabezado = [
                "Patr\u00f3n",
                "Residuo rms (S11)",
                "Residuo de los dem\u00e1s sin \u00e9l",
            ]
            anchos = [6.6 * cm, 5.2 * cm, 5.2 * cm]
        else:
            encabezado = ["Patr\u00f3n", "Residuo rms (S11)"]
            anchos = [9.0 * cm, 8.0 * cm]

        datos = [[Paragraph(h, _ESTILOS['CeldaTablaEncabezado']) for h in encabezado]]
        for nom, rms in residuo_rms.items():
            fila = [
                Paragraph(str(nom), _ESTILOS['CeldaTabla']),
                Paragraph(f"{rms:.4e}", _ESTILOS['CeldaTabla']),
            ]
            if tiene_vc:
                v = validacion.get(nom)
                if v is not None:
                    fila.append(Paragraph(f"{v['residuo_resto']:.4e}", _ESTILOS['CeldaTabla']))
                else:
                    fila.append(Paragraph("fijo (corto/aire)", _ESTILOS['CeldaTabla']))
            datos.append(fila)

        t = Table(datos, hAlign='LEFT', colWidths=anchos)
        t.setStyle(_estilo_tabla())
        elementos.append(Paragraph("Consistencia en S11 entre patrones:", _ESTILOS['Normal']))
        elementos.append(Spacer(1, 0.12 * cm))
        elementos.append(t)
        if tiene_vc:
            elementos.append(Spacer(1, 0.15 * cm))
            elementos.append(Paragraph(
                "Residuo de los dem\u00e1s sin \u00e9l: residuo de S11 al recalibrar excluyendo ese "
                "l\u00edquido. Si al sacar un patr\u00f3n el residuo de los dem\u00e1s baja mucho, ese "
                "patr\u00f3n es el que estaba introduciendo tensi\u00f3n en el ajuste.",
                _ESTILOS['Nota']))
        elif dof == 0:
            elementos.append(Spacer(1, 0.15 * cm))
            elementos.append(Paragraph(
                "Nota: con exactamente 4 patrones el ajuste es exacto (0 grados de "
                "libertad) y equivale al m\u00e9todo completo de 4 patrones. Agregue un 5to "
                "patr\u00f3n en la configuraci\u00f3n para aprovechar la redundancia y la "
                "validaci\u00f3n cruzada.",
                _ESTILOS['Nota']))

    return elementos


def generar_reporte_pdf(ruta_salida, metadata, chequeo_calibracion, materiales):
    """
    Arma el informe PDF final.
    """
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    doc = SimpleDocTemplate(ruta_salida, pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm,
                             title="Informe de permitividad dielectrica")
    story = []

    # ---- Portada ----
    story.append(Spacer(1, 2.5 * cm))
    story.append(Paragraph("Informe de medici\u00f3n de permitividad diel\u00e9ctrica",
                            _ESTILOS['TituloInforme']))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Sonda coaxial open-ended \u2014 Medidas Electr\u00f3nicas 2",
                            _ESTILOS['Subtitulo']))
    story.append(Spacer(1, 1.5 * cm))
    story.append(_tabla_metadata(metadata))
    if metadata.get('cancelado'):
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "AVISO \u2014 Informe PARCIAL: el an\u00e1lisis se cancel\u00f3 antes de "
            "procesar todos los materiales configurados.",
            _ESTILOS['AvisoCancelado']))

    for adv in metadata.get('advertencias_calibracion', []) or []:
        rol = adv.get('rol') or "Patr\u00f3n de calibraci\u00f3n"
        etiqueta_adv = adv.get('etiqueta', '')
        story.append(Spacer(1, 0.45 * cm))
        story.append(_caja_advertencia(
            f"{rol}: {etiqueta_adv}",
            adv.get('texto', ''),
            nivel=adv.get('nivel', 'aviso')))
    story.append(PageBreak())

    # ---- Chequeo de calibracion ----
    if chequeo_calibracion is not None:
        etiqueta_p3 = chequeo_calibracion.get('etiqueta_patron3') or "patron 3"
        etiqueta_p4 = chequeo_calibracion.get('etiqueta_patron4') or "patron 4"
        bloque = [
            Paragraph(f"Chequeo de calibraci\u00f3n ({etiqueta_p3})", _ESTILOS['NombreMaterial']),
            Paragraph(
                f"{etiqueta_p3} es uno de los patrones de calibraci\u00f3n: al procesarlo "
                "como si fuera un material m\u00e1s, el resultado tiene que coincidir "
                "pr\u00e1cticamente con su propio modelo te\u00f3rico. Sirve para confirmar "
                "que la lectura de archivos y la calibraci\u00f3n funcionaron bien antes "
                "de analizar el resto de los materiales.",
                _ESTILOS['Normal']),
            Spacer(1, 0.3 * cm),
        ]
        if chequeo_calibracion.get('figura') and os.path.isfile(chequeo_calibracion['figura']):
            bloque.append(_imagen_ajustada(chequeo_calibracion['figura']))
        if chequeo_calibracion.get('errores'):
            bloque.append(Spacer(1, 0.25 * cm))
            bloque.append(Paragraph("Error absoluto respecto del modelo te\u00f3rico (por intervalos de frecuencia):",
                                    _ESTILOS['Normal']))
            bloque.append(Spacer(1, 0.15 * cm))
            bloque.append(_tabla_errores(chequeo_calibracion['errores']))
        story.append(KeepTogether(bloque))

        chequeo_p4 = chequeo_calibracion.get('patron4')
        if chequeo_p4 is not None:
            bloque_p4 = [
                Paragraph(f"Chequeo de calibraci\u00f3n ({etiqueta_p4})", _ESTILOS['NombreMaterial']),
                Paragraph(
                    f"Mismo chequeo que el de arriba, pero para {etiqueta_p4} (el 4to "
                    "patron): al procesarlo como si fuera un material m\u00e1s, el "
                    "resultado tiene que coincidir pr\u00e1cticamente con su propio "
                    "modelo te\u00f3rico. Da una validaci\u00f3n independiente de la de "
                    f"{etiqueta_p3} -- un problema espec\u00edfico de este patr\u00f3n "
                    "puede no notarse mirando solo el chequeo anterior.",
                    _ESTILOS['Normal']),
                Spacer(1, 0.3 * cm),
            ]
            if chequeo_p4.get('figura') and os.path.isfile(chequeo_p4['figura']):
                bloque_p4.append(_imagen_ajustada(chequeo_p4['figura']))
            if chequeo_p4.get('errores'):
                bloque_p4.append(Spacer(1, 0.25 * cm))
                bloque_p4.append(Paragraph("Error absoluto respecto del modelo te\u00f3rico (por intervalos de frecuencia):",
                                           _ESTILOS['Normal']))
                bloque_p4.append(Spacer(1, 0.15 * cm))
                bloque_p4.append(_tabla_errores(chequeo_p4['errores']))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether(bloque_p4))

        if chequeo_calibracion.get('figura_Gn') and os.path.isfile(chequeo_calibracion['figura_Gn']):
            elementos_gn = [
                Paragraph("Diagn\u00f3stico: conductancia normalizada Gn(f)",
                          _ESTILOS['NombreMaterial']),
                Paragraph(
                    "Gn se calcula a partir de los patrones de calibraci\u00f3n (no "
                    "depende del material bajo ensayo), por lo que deber\u00eda variar "
                    "en forma suave con la frecuencia. Un salto brusco o un pico "
                    "aislado suele indicar un problema con la medici\u00f3n de alguno de "
                    f"los patrones, t\u00edpicamente {etiqueta_p4} (el \u00fanico "
                    "que interviene en este c\u00e1lculo en el m\u00e9todo de 4 patrones).",
                    _ESTILOS['Normal']),
                Spacer(1, 0.2 * cm),
                _imagen_ajustada(chequeo_calibracion['figura_Gn']),
            ]
            info_acond = chequeo_calibracion.get('info_acond_Gn')
            if info_acond and info_acond.get('tipo'):
                texto_acond = _formatear_info_acond_pdf(info_acond)
                if texto_acond:
                    elementos_gn.append(Spacer(1, 0.2 * cm))
                    elementos_gn.append(Paragraph(texto_acond, _ESTILOS['Normal']))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether(elementos_gn))

        chequeo_mc = chequeo_calibracion.get('minimos_cuadrados')
        if chequeo_mc is not None:
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether(_seccion_minimos_cuadrados(chequeo_mc)))
            errores_vc = chequeo_mc.get('errores_validacion_cruzada')
            if errores_vc and chequeo_mc.get('grados_libertad', 0) > 0:
                story.append(Spacer(1, 0.25 * cm))
                bloque_vc_err = [
                    Paragraph(
                        "Error al medirlo sin \u00e9l (validaci\u00f3n cruzada, error absoluto por intervalos de frecuencia):",
                        _ESTILOS['Normal']),
                    Spacer(1, 0.12 * cm),
                    _tabla_errores(errores_vc, titulo_col0="Patr\u00f3n medido (sin \u00e9l)"),
                    Spacer(1, 0.12 * cm),
                    Paragraph(
                        "Error al medirlo sin \u00e9l: se recalibra excluyendo ese patr\u00f3n l\u00edquido "
                        "(respetando el acondicionamiento de Gn si est\u00e1 activo) y se lo mide "
                        "como si fuera una muestra desconocida, desglosando su error absoluto "
                        "en los 3 intervalos de frecuencia.",
                        _ESTILOS['Nota']),
                ]
                story.append(KeepTogether(bloque_vc_err))

        if materiales:
            story.append(_separador())

    # ---- Un bloque por material ----
    # Cada sub-seccion (titulo+grafico principal, S11 modulo/fase, Smith,
    # tabla de errores, tabla resumen) va en su propio KeepTogether: si no
    # entra en lo que queda de la pagina actual, ReportLab la mueve entera
    # a la pagina siguiente. Antes esas piezas iban sueltas (un Paragraph
    # con el titulo, aparte la imagen/tabla), asi que era comun que el
    # titulo quedara solo al pie de una pagina y la imagen recien en la
    # proxima -- o, peor, que la tabla se partiera a la mitad (el
    # encabezado de la tabla en una pagina y las filas de datos en la
    # siguiente). Ademas ya no se fuerza PageBreak() entre secciones: se
    # usa `_separador()` (una linea fina), asi que si dos materiales
    # entran completos en la misma pagina no se desperdicia espacio en
    # blanco, y si no entran, el salto de pagina lo decide el contenido
    # real (KeepTogether), no una orden ciega.
    n_materiales = len(materiales)
    for idx, mat in enumerate(materiales):
        encabezado = [
            Paragraph(mat['nombre'], _ESTILOS['NombreMaterial']),
            Paragraph(f"Archivo: {mat['archivo']}", _ESTILOS['Nota']),
        ]
        modelo_etiqueta = mat.get('modelo_etiqueta')
        if modelo_etiqueta:
            temp = mat.get('temperatura')
            texto_temp = f" (T = {temp:.1f} \u00b0C)" if temp is not None else ""
            resumen = mat.get('modelo_resumen')
            texto_resumen = f" \u2014 {resumen}" if resumen else ""
            encabezado.append(
                Paragraph(f"Modelo te\u00f3rico: {modelo_etiqueta}{texto_temp}"
                          f"{texto_resumen}",
                          _ESTILOS['Nota']))
        if mat.get('filtrar_s11') or mat.get('filtrar_sg'):
            t_f = mat.get('tipo_filtro', 'savitzky_golay')
            if t_f == 'pasabajos':
                encabezado.append(Paragraph(
                    f"Filtro aplicado sobre S11: Pasabajos Butterworth (orden={mat.get('pb_orden', 2)}, fc={mat.get('pb_fcorte', 0.10):.2f})",
                    _ESTILOS['Nota']))
            else:
                encabezado.append(Paragraph(
                    f"Filtro aplicado sobre S11: Savitzky-Golay (ventana={mat.get('sg_window', 15)}, orden={mat.get('sg_polyorder', 3)})",
                    _ESTILOS['Nota']))
        encabezado.append(Spacer(1, 0.2 * cm))
        if mat.get('figura') and os.path.isfile(mat['figura']):
            encabezado.append(_imagen_ajustada(mat['figura']))
        story.append(KeepTogether(encabezado))
        story.append(Spacer(1, 0.3 * cm))

        # Caja de advertencia del modelo teorico de ESTE material (p.ej.
        # acetona/ciclohexano/silicona, que no tienen ecuacion de Debye y
        # se modelan como permitividad constante). Va inmediatamente
        # despues del grafico, antes de las tablas de error, para que se
        # lea ANTES de sacar conclusiones de esos numeros.
        if mat.get('advertencia_modelo'):
            story.append(KeepTogether([
                _caja_advertencia(
                    f"Modelo te\u00f3rico sin ecuaci\u00f3n de relajaci\u00f3n: "
                    f"{modelo_etiqueta or mat.get('modelo', '')}",
                    mat['advertencia_modelo'],
                    nivel=mat.get('nivel_advertencia', 'aviso')),
            ]))
            story.append(Spacer(1, 0.3 * cm))

        if mat.get('figura_s11') and os.path.isfile(mat['figura_s11']):
            story.append(KeepTogether([
                Paragraph("S11 medido (m\u00f3dulo y fase):", _ESTILOS['Normal']),
                Spacer(1, 0.15 * cm),
                _imagen_ajustada(mat['figura_s11']),
            ]))
            story.append(Spacer(1, 0.3 * cm))

        if mat.get('figura_smith') and os.path.isfile(mat['figura_smith']):
            story.append(KeepTogether([
                Paragraph("S11 medido (diagrama de Smith):", _ESTILOS['Normal']),
                Spacer(1, 0.15 * cm),
                _imagen_ajustada(mat['figura_smith'], ancho_max=9 * cm),
            ]))
            story.append(Spacer(1, 0.3 * cm))

        if mat.get('tiene_teorico') and mat.get('errores'):
            bloque_err = [
                Paragraph("Error absoluto respecto del modelo te\u00f3rico (por intervalos de frecuencia):",
                          _ESTILOS['Normal']),
                Spacer(1, 0.15 * cm),
                _tabla_errores(mat['errores']),
            ]
            # Si el modelo de referencia no modela perdidas, la columna de
            # er'' sale toda en "n/a": conviene aclarar por que, para que
            # no se lea como un fallo de calculo.
            if mat.get('modelo_perdidas_comparables') is False:
                # Dos motivos distintos para el mismo "n/a": la referencia
                # no modela perdidas, o las modela pero son minusculas.
                if mat.get('modelo_modela_perdidas') is False:
                    motivo = ("el modelo te\u00f3rico de referencia no modela "
                              "p\u00e9rdidas (er'' = 0)")
                else:
                    motivo = ("las p\u00e9rdidas de referencia son de \u00f3rdenes de "
                              "magnitud menores que la resoluci\u00f3n de la sonda coaxial")
                bloque_err.append(Spacer(1, 0.15 * cm))
                bloque_err.append(Paragraph(
                    f"Las columnas de er'' figuran como <b>n/a</b> porque {motivo}. "
                    "Ver la advertencia de arriba.",
                    _ESTILOS['Nota']))
            story.append(KeepTogether(bloque_err))
            story.append(Spacer(1, 0.3 * cm))

        if mat.get('tabla_columnas'):
            story.append(KeepTogether([
                Paragraph("Tabla resumen en frecuencias de inter\u00e9s:",
                          _ESTILOS['Normal']),
                Spacer(1, 0.15 * cm),
                _tabla_resumen(mat['tabla_frecs_ghz'], mat['tabla_columnas']),
            ]))

        if mat.get('nota'):
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(mat['nota'], _ESTILOS['Nota']))

        if idx < n_materiales - 1:
            story.append(_separador())

    doc.build(story, onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)
    print(f"  Informe PDF guardado: {ruta_salida}")
    return ruta_salida