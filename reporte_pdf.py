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

_COLOR_ENCABEZADO = colors.HexColor('#2c3e50')
_COLOR_FILA_PAR = colors.HexColor('#f2f2f2')
_ANCHO_PAGINA_UTIL = 17 * cm  # A4 menos margenes de 2cm a cada lado


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


def _tabla_errores(errores):
    """
    errores: dict {etiqueta: {'err_re_medio','err_re_max','err_im_medio','err_im_max'}}
    (valores en %, ya calculados con funciones.error_relativo_porcentual).
    """
    datos = [["Metodo", "Error medio\ner'", "Error max\ner'", "Error medio\ner''", "Error max\ner''"]]
    for etiqueta, e in errores.items():
        datos.append([
            etiqueta,
            f"{e['err_re_medio']:.2f}%", f"{e['err_re_max']:.2f}%",
            f"{e['err_im_medio']:.2f}%", f"{e['err_im_max']:.2f}%",
        ])
    ancho_primera = 6 * cm
    ancho_resto = (_ANCHO_PAGINA_UTIL - ancho_primera) / 4
    t = Table(datos, hAlign='LEFT',
              colWidths=[ancho_primera] + [ancho_resto] * 4)
    t.setStyle(_estilo_tabla())
    return t


def _tabla_resumen(frecs_ghz, columnas):
    """
    frecs_ghz : lista/array de frecuencias (GHz) a mostrar como filas.
    columnas  : dict {titulo: ndarray complejo} evaluado en esas frecuencias
                (mismo largo que frecs_ghz).
    """
    encabezado = ["f (GHz)"] + list(columnas.keys())
    datos = [encabezado]
    for i, f in enumerate(frecs_ghz):
        fila = [f"{f:.2f}"]
        for col in columnas.values():
            v = col[i]
            fila.append(f"{v.real:.2f} - {abs(v.imag):.2f}j")
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


def _tabla_metadata(metadata):
    fecha = metadata.get('fecha') or datetime.now().strftime("%d/%m/%Y %H:%M")
    filas = [
        ["Fecha de generacion", fecha],
        _fila_temperatura("Temperatura del agua (CAL)",
                           metadata.get('temperatura_agua')),
        _fila_temperatura("Temperatura del alcohol isopropilico (CAL)",
                           metadata.get('temperatura_isoprop')),
        ["Banda analizada", f"{metadata['f_min_ghz']} - {metadata['f_max_ghz']} GHz"],
    ]
    for clave, archivo in metadata.get('archivos_calibracion', {}).items():
        filas.append([f"Patron de calibracion: {clave}", archivo])

    t = Table(filas, hAlign='LEFT', colWidths=[7 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef2f5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return t


def generar_reporte_pdf(ruta_salida, metadata, chequeo_calibracion, materiales):
    """
    Arma el informe PDF final.

    Parametros
    ----------
    ruta_salida : str
        Path del PDF a generar (por ejemplo "./salidas/informe.pdf").
    metadata : dict
        'temperatura_agua', 'temperatura_isoprop' (temperaturas de los
        dos patrones de calibracion que dependen de T; cualquiera de las
        dos puede faltar -- p.ej. en un informe armado con datos mas
        viejos -- y se muestra "no especificada" en vez de romper),
        'f_min_ghz', 'f_max_ghz', 'archivos_calibracion' (dict
        {clave: nombre_archivo}), 'fecha' (opcional, string),
        'cancelado' (opcional, bool: si True, se muestra un aviso en la
        portada indicando que el analisis se corto antes de terminar
        todos los materiales -- ver `analisis_permitividad.ejecutar_
        analisis` y su parametro `cancelado`).
    chequeo_calibracion : dict o None
        'figura' (path al .png) y 'errores' (dict como en `_tabla_errores`,
        con una sola entrada por metodo), o None si no se hizo ese chequeo.
    materiales : list[dict]
        Uno por material analizado, con las claves:
        'nombre', 'archivo', 'figura' (path .png),
        'tiene_teorico' (bool), 'errores' (dict o None),
        'tabla_frecs_ghz' (list[float] o None),
        'tabla_columnas' (dict {titulo: ndarray} o None),
        'nota' (str opcional), y opcionalmente 'modelo_etiqueta' (nombre
        del modelo teorico usado, o None si no tiene) junto con
        'temperatura' (la temperatura de ESE material durante el
        ensayo, distinta de las de calibracion de arriba; solo tiene
        sentido si 'modelo_etiqueta' no es None). Si faltan estas dos
        ultimas claves (informe armado con una version anterior de
        analisis_permitividad.py) simplemente no se muestra esa linea.
        Tambien opcionales: 'figura_s11' (path .png del S11 medido en
        modulo/fase) y 'figura_smith' (path .png del diagrama de Smith
        de ese S11); si faltan o el archivo no existe, esas figuras
        simplemente no aparecen (no rompe nada).

    Retorna
    -------
    La ruta del PDF generado (mismo valor que `ruta_salida`).
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
    story.append(PageBreak())

    # ---- Chequeo de calibracion ----
    # Todo el bloque (titulo + texto + grafico + tabla) va en un unico
    # KeepTogether: como junto mide bastante menos que una pagina, entra
    # entero sin problema, y asi nunca queda el titulo/grafico separado de
    # su tabla por un salto de pagina en el medio.
    if chequeo_calibracion is not None:
        bloque = [
            Paragraph("Chequeo de calibraci\u00f3n (agua)", _ESTILOS['NombreMaterial']),
            Paragraph(
                "El agua es uno de los patrones de calibraci\u00f3n: al procesarla como "
                "si fuera un material m\u00e1s, el resultado tiene que coincidir "
                "pr\u00e1cticamente con su propio modelo te\u00f3rico. Sirve para confirmar "
                "que la lectura de archivos y la calibraci\u00f3n funcionaron bien antes "
                "de analizar el resto de los materiales.",
                _ESTILOS['Normal']),
            Spacer(1, 0.3 * cm),
        ]
        if chequeo_calibracion.get('figura') and os.path.isfile(chequeo_calibracion['figura']):
            bloque.append(_imagen_ajustada(chequeo_calibracion['figura']))
        if chequeo_calibracion.get('errores'):
            bloque.append(Spacer(1, 0.3 * cm))
            bloque.append(_tabla_errores(chequeo_calibracion['errores']))
        story.append(KeepTogether(bloque))

        # Diagnostico aparte: Gn(f) no depende del agua en particular sino
        # de los 4 patrones de calibracion en conjunto, asi que va en su
        # propio KeepTogether (no forzado a entrar junto con el bloque de
        # arriba, que ya puede ser largo).
        if chequeo_calibracion.get('figura_Gn') and os.path.isfile(chequeo_calibracion['figura_Gn']):
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether([
                Paragraph("Diagn\u00f3stico: conductancia normalizada Gn(f)",
                          _ESTILOS['NombreMaterial']),
                Paragraph(
                    "Gn se calcula a partir de los 4 patrones de calibraci\u00f3n (no "
                    "depende del material bajo ensayo), por lo que deber\u00eda variar "
                    "en forma suave con la frecuencia. Un salto brusco o un pico "
                    "aislado suele indicar un problema con la medici\u00f3n de alguno de "
                    "los patrones, t\u00edpicamente el alcohol isoprop\u00edlico (el \u00fanico "
                    "que interviene en este c\u00e1lculo).",
                    _ESTILOS['Normal']),
                Spacer(1, 0.2 * cm),
                _imagen_ajustada(chequeo_calibracion['figura_Gn']),
            ]))

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
            encabezado.append(
                Paragraph(f"Modelo te\u00f3rico: {modelo_etiqueta}{texto_temp}",
                          _ESTILOS['Nota']))
        encabezado.append(Spacer(1, 0.2 * cm))
        if mat.get('figura') and os.path.isfile(mat['figura']):
            encabezado.append(_imagen_ajustada(mat['figura']))
        story.append(KeepTogether(encabezado))
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
            story.append(KeepTogether([
                Paragraph("Error relativo respecto del modelo te\u00f3rico:",
                          _ESTILOS['Normal']),
                Spacer(1, 0.15 * cm),
                _tabla_errores(mat['errores']),
            ]))
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
