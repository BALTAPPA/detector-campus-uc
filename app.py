
import streamlit as st
import numpy as np
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from datetime import date, timedelta


# =====================================
# 1. CONFIGURACION DE LA PAGINA
# =====================================

st.set_page_config(
    page_title="Detector Campus UC",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Detector de disponibilidad - Campus UC")

st.write(
    "Sistema de análisis de ocupación del campus mediante "
    "inteligencia artificial y estimaciones de actividad académica."
)


# =====================================
# 2. FUNCIONES GENERALES
# =====================================

def calcular_interseccion(caja_mesa, caja_persona):

    mx1, my1, mx2, my2 = caja_mesa
    px1, py1, px2, py2 = caja_persona

    ix1 = max(mx1, px1)
    iy1 = max(my1, py1)
    ix2 = min(mx2, px2)
    iy2 = min(my2, py2)

    ancho = max(0, ix2 - ix1)
    alto = max(0, iy2 - iy1)

    return ancho * alto > 0


# =====================================
# 3. CARGAR MODELO YOLO
# =====================================

@st.cache_resource
def cargar_modelo():

    return YOLO("yolov8n.pt")


# =====================================
# 4. CREAR PESTAÑAS
# =====================================

pestana1, pestana2 = st.tabs([
    "📷 Disponibilidad actual",
    "📊 Predicción del campus"
])


# ==========================================================
# PESTAÑA 1: DISPONIBILIDAD ACTUAL
# ==========================================================

with pestana1:

    st.header("Disponibilidad actual")

    st.write(
        "Selecciona fotografías del campus automáticamente "
        "y consulta la disponibilidad de las mesas."
    )

    # =================================
    # 5. BUSCAR FOTOGRAFIAS
    # =================================

    CARPETA = Path(__file__).resolve().parent

    FOTOGRAFIAS = sorted([
        foto for foto in CARPETA.iterdir()
        if foto.is_file()
        and foto.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if FOTOGRAFIAS:

        # =================================
        # 6. SELECCIONAR FOTOGRAFIA
        # =================================

        if (
            "foto_seleccionada" not in st.session_state
            or st.session_state.foto_seleccionada
            not in [str(f) for f in FOTOGRAFIAS]
        ):

            st.session_state.foto_seleccionada = str(
                random.choice(FOTOGRAFIAS)
            )

        # Botón para cambiar fotografía
        if st.button("🔄 Cambiar fotografía"):

            fotos_disponibles = [
                str(foto)
                for foto in FOTOGRAFIAS
                if str(foto) != st.session_state.foto_seleccionada
            ]

            if fotos_disponibles:

                st.session_state.foto_seleccionada = (
                    random.choice(fotos_disponibles)
                )

        # =================================
        # 7. MOSTRAR FOTOGRAFIA PEQUEÑA
        # =================================

        ruta_foto = st.session_state.foto_seleccionada

        imagen = Image.open(ruta_foto).convert("RGB")

        ancho, alto = imagen.size

        st.image(
            imagen,
            caption=Path(ruta_foto).name,
            width=260
        )

        # =================================
        # 8. COORDENADAS INTERNAS DE LAS MESAS
        # =================================

        # Las coordenadas no aparecen en la interfaz.
        # Se utilizan internamente para localizar las mesas
        # dentro de la fotografía.

        mesas = {
            "Mesa 1": [100, 200, 240, 330],
            "Mesa 2": [430, 220, 540, 320],
            "Mesa 3": [570, 250, 680, 350],
            "Mesa 4": [760, 380, 880, 480],
            "Mesa 5": [240, 380, 500, 520]
        }

        # Ajustar las coordenadas al tamaño de la imagen

        for nombre, caja in mesas.items():

            caja[0] = min(caja[0], ancho - 1)
            caja[1] = min(caja[1], alto - 1)
            caja[2] = min(caja[2], ancho)
            caja[3] = min(caja[3], alto)

        # =================================
        # 9. DETECTAR PERSONAS CON YOLO
        # =================================

        # Los resultados se vuelven a calcular
        # cuando se selecciona una fotografía diferente.

        if (
            "foto_procesada" not in st.session_state
            or st.session_state.foto_procesada != ruta_foto
        ):

            with st.spinner("Cargando inteligencia artificial..."):

                modelo = cargar_modelo()

            with st.spinner("Analizando fotografía y detectando personas..."):

                resultados = modelo.predict(
                    source=imagen,
                    classes=[0],
                    conf=0.25,
                    verbose=False
                )[0]

                personas = []

                for box in resultados.boxes:

                    coordenadas = box.xyxy[0].cpu().numpy()

                    personas.append(coordenadas.tolist())

                # =================================
                # 10. ANALIZAR LAS CINCO MESAS
                # =================================

                estados = {}

                for nombre, coordenadas in mesas.items():

                    ocupada = False

                    for persona in personas:

                        if calcular_interseccion(
                            coordenadas,
                            persona
                        ):

                            ocupada = True
                            break

                    estados[nombre] = (
                        "OCUPADA" if ocupada else "DISPONIBLE"
                    )

                # Guardar resultados en la sesión

                st.session_state.estados_mesas = estados
                st.session_state.foto_procesada = ruta_foto
                st.session_state.personas_detectadas = len(personas)

        # Recuperar resultados guardados

        estados = st.session_state.estados_mesas

        # =================================
        # 11. MAPA DE DISPONIBILIDAD
        # =================================

        st.divider()

        st.subheader("Mapa de disponibilidad")

        ocupadas = sum(
            estado == "OCUPADA"
            for estado in estados.values()
        )

        disponibles = 5 - ocupadas

        filas = [
            ["Mesa 1", "Mesa 2", "Mesa 3"],
            ["Mesa 4", "Mesa 5"]
        ]

        for fila in filas:

            columnas_mapa = st.columns(3)

            for i in range(3):

                with columnas_mapa[i]:

                    if i < len(fila):

                        nombre = fila[i]
                        estado = estados[nombre]

                        if estado == "DISPONIBLE":

                            color_fondo = "#bbf7d0"
                            color_texto = "#166534"
                            simbolo = "✓"

                        else:

                            color_fondo = "#fecaca"
                            color_texto = "#991b1b"
                            simbolo = "✕"

                        st.markdown(
                            f"""
                            <div style="
                                background:{color_fondo};
                                color:{color_texto};
                                border-radius:12px;
                                border:2px solid {color_texto};
                                min-height:115px;
                                display:flex;
                                flex-direction:column;
                                align-items:center;
                                justify-content:center;
                                margin:5px;
                                font-weight:bold;
                            ">
                                <span style="font-size:20px;">
                                    {nombre}
                                </span>

                                <span style="font-size:25px;">
                                    {simbolo}
                                </span>

                                <span style="font-size:13px;">
                                    {estado}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            "<div style='height:125px'></div>",
                            unsafe_allow_html=True
                        )

        # =================================
        # 12. ESTADISTICAS GENERALES
        # =================================

        st.divider()

        st.subheader("Estado general")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total de mesas", 5)

        col2.metric("Disponibles", disponibles)

        col3.metric("Ocupadas", ocupadas)

        st.caption(
            f"Personas detectadas en la fotografía: "
            f"{st.session_state.personas_detectadas}"
        )

        st.info(
            "La disponibilidad se estima mediante la "
            "intersección entre las personas detectadas "
            "y las zonas correspondientes a cada mesa."
        )

    else:

        st.error(
            "No se encontraron fotografías. "
            "Agrega imágenes JPG, JPEG o PNG al repositorio "
            "junto a app.py."
        )


# ==========================================================
# PESTAÑA 2: PREDICCION DE OCUPACION DEL CAMPUS
# ==========================================================

with pestana2:

    st.header("Predicción de ocupación del campus")

    st.write(
        "Estimación de la actividad general del Campus San Joaquín "
        "según el calendario académico UC y la programación "
        "semanal de talleres de Arquitectura."
    )

    st.info(
        "Los porcentajes son estimaciones iniciales basadas "
        "en reglas académicas. No corresponden a mediciones "
        "reales de personas presentes en el campus."
    )

    # =================================
    # 13. CALENDARIO ACADEMICO 2026
    # =================================

    INICIO_CLASES = date(2026, 8, 5)
    FIN_CLASES = date(2026, 11, 27)

    INICIO_RECESO = date(2026, 9, 14)
    FIN_RECESO = date(2026, 9, 21)

    INICIO_EVALUACIONES = date(2026, 11, 30)
    FIN_EVALUACIONES = date(2026, 12, 14)

    FIN_PERIODO = date(2026, 12, 16)

    SUSPENSION = date(2026, 12, 7)
    FERIADO = date(2026, 12, 8)

    # =================================
    # 14. FUNCION DE PREDICCION
    # =================================

    def predecir_ocupacion(fecha):

        dia = fecha.weekday()

        # Fin de semana

        if dia >= 5:

            return None, "Fin de semana", (
                "No se estima actividad académica regular."
            )

        # Fechas fuera del segundo periodo

        if fecha < INICIO_CLASES or fecha > FIN_PERIODO:

            return None, "Sin datos del periodo", (
                "La fecha está fuera del segundo periodo académico "
                "considerado en este modelo."
            )

        # Receso académico

        if INICIO_RECESO <= fecha <= FIN_RECESO:

            return 10, "Receso académico", (
                "Disminución importante de la actividad universitaria."
            )

        # Suspensión de actividades

        if fecha == SUSPENSION:

            return 0, "Suspensión de actividades", (
                "Suspensión de actividades académicas y administrativas."
            )

        # Feriado

        if fecha == FERIADO:

            return 0, "Feriado", (
                "Se espera una actividad universitaria muy baja."
            )

        # Periodo de evaluaciones

        if INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:

            valores_evaluacion = {
                0: 65,
                1: 70,
                2: 65,
                3: 70,
                4: 60
            }

            valor = valores_evaluacion[dia]

            return valor, "Periodo de evaluaciones", (
                "Actividad asociada a evaluaciones y cierres de cursos. "
                "La actividad puede variar según cada carrera."
            )

        # Cierre del periodo

        if fecha > FIN_EVALUACIONES:

            return 25, "Cierre del periodo académico", (
                "Actividad académica reducida después del periodo "
                "de evaluaciones."
            )

        # =================================
        # SEMANAS NORMALES DE CLASES
        # =================================

        valores_semanales = {
            0: 55,
            1: 90,
            2: 60,
            3: 90,
            4: 80
        }

        actividades = {
            0: (
                "Actividad media: cursos teóricos, electivos "
                "y trabajo autónomo."
            ),

            1: (
                "Actividad muy alta: talleres de Arquitectura, "
                "correcciones y revisiones de proyectos."
            ),

            2: (
                "Actividad media: clases teóricas, laboratorios "
                "y desarrollo autónomo de proyectos."
            ),

            3: (
                "Actividad muy alta: talleres, correcciones "
                "y trabajo presencial de Arquitectura."
            ),

            4: (
                "Actividad alta: entregas, presentaciones "
                "y revisiones de proyectos."
            )
        }

        return (
            valores_semanales[dia],
            "Periodo normal de clases",
            actividades[dia]
        )

    # =================================
    # 15. SELECCIONAR SEMANA
    # =================================

    st.divider()

    st.subheader("Consultar actividad semanal")

    fecha_seleccionada = st.date_input(
        "Selecciona un día de la semana que quieres consultar",
        value=date.today(),
        min_value=date(2026, 1, 1),
        max_value=date(2027, 12, 31)
    )

    # Encontrar el lunes de esa semana

    lunes = fecha_seleccionada - timedelta(
        days=fecha_seleccionada.weekday()
    )

    dias_semana = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes"
    ]

    st.write(
        f"**Semana consultada:** "
        f"{lunes.strftime('%d/%m/%Y')} al "
        f"{(lunes + timedelta(days=4)).strftime('%d/%m/%Y')}"
    )

    # =================================
    # 16. PREDICCION POR DIA
    # =================================

    predicciones = []

    columnas_dias = st.columns(5)

    for i in range(5):

        fecha = lunes + timedelta(days=i)

        porcentaje, etapa, explicacion = predecir_ocupacion(fecha)

        predicciones.append({
            "dia": dias_semana[i],
            "fecha": fecha,
            "porcentaje": porcentaje,
            "etapa": etapa,
            "explicacion": explicacion
        })

        with columnas_dias[i]:

            st.markdown(f"### {dias_semana[i]}")

            st.caption(fecha.strftime("%d/%m"))

            if porcentaje is None:

                st.metric("Estimación", "N/D")

                st.caption(etapa)

            else:

                st.metric(
                    "Ocupación estimada",
                    f"{porcentaje}%"
                )

                st.progress(porcentaje / 100)

                if porcentaje >= 80:

                    st.error("Actividad muy alta")

                elif porcentaje >= 65:

                    st.warning("Actividad alta")

                elif porcentaje >= 40:

                    st.info("Actividad media")

                else:

                    st.success("Actividad baja")

    # =================================
    # 17. DETALLE DE LA SEMANA
    # =================================

    st.divider()

    st.subheader("Detalle de la semana")

    for prediccion in predicciones:

        with st.expander(
            f"{prediccion['dia']} - "
            f"{prediccion['fecha'].strftime('%d/%m/%Y')}"
        ):

            st.write(
                f"**Periodo:** {prediccion['etapa']}"
            )

            st.write(prediccion["explicacion"])

            if prediccion["porcentaje"] is not None:

                st.write(
                    f"**Índice estimado de actividad: "
                    f"{prediccion['porcentaje']}%**"
                )

    # =================================
    # 18. RESUMEN SEMANAL
    # =================================

    st.divider()

    st.subheader("Resumen semanal")

    dias_validos = [
        p for p in predicciones
        if p["porcentaje"] is not None
    ]

    if dias_validos:

        dia_mayor = max(
            dias_validos,
            key=lambda p: p["porcentaje"]
        )

        dia_menor = min(
            dias_validos,
            key=lambda p: p["porcentaje"]
        )

        promedio = sum(
            p["porcentaje"] for p in dias_validos
        ) / len(dias_validos)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Promedio semanal estimado",
            f"{promedio:.0f}%"
        )

        col2.metric(
            "Día de mayor actividad",
            dia_mayor["dia"],
            f"{dia_mayor['porcentaje']}%"
        )

        col3.metric(
            "Día de menor actividad",
            dia_menor["dia"],
            f"{dia_menor['porcentaje']}%"
        )

    else:

        st.warning(
            "No hay días con estimaciones disponibles "
            "para esta semana."
        )

    # =================================
    # 19. EXPLICACION DEL MODELO
    # =================================

    st.divider()

    with st.expander("¿Cómo funciona esta predicción?"):

        st.write("""
        La estimación considera tres componentes:

        **1. Calendario académico UC**

        - Periodo de clases.
        - Recesos académicos.
        - Feriados y suspensiones.
        - Periodo de evaluaciones finales.

        **2. Actividad general del campus**

        Se considera la actividad de cursos, electivos,
        laboratorios y trabajo autónomo de los estudiantes.

        **3. Actividad de Arquitectura**

        - Lunes y miércoles: cursos teóricos y trabajo autónomo.
        - Martes y jueves: talleres y correcciones.
        - Viernes: entregas y revisiones.

        Los valores son ponderaciones iniciales definidas
        para construir una primera aproximación del modelo.
        No son porcentajes obtenidos de cámaras ni sensores.
        """)
