
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
        "El sistema selecciona una fotografía de la base de datos "
        "del campus y analiza la disponibilidad de las mesas."
    )

    # =================================
    # 5. BUSCAR FOTOGRAFIAS
    # =================================

    CARPETA = Path(__file__).resolve().parent

    FOTOGRAFIAS = [
        foto for foto in CARPETA.iterdir()
        if foto.is_file()
        and foto.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    FOTOGRAFIAS = sorted(FOTOGRAFIAS)

    # =================================
    # 6. SELECCIONAR FOTOGRAFIA
    # =================================

    if "foto_seleccionada" not in st.session_state:

        if FOTOGRAFIAS:
            st.session_state.foto_seleccionada = str(
                random.choice(FOTOGRAFIAS)
            )
        else:
            st.session_state.foto_seleccionada = None

    if FOTOGRAFIAS:

        # Botón para cambiar fotografía
        if st.button("🔄 Cambiar fotografía"):

            fotos_disponibles = [
                str(foto) for foto in FOTOGRAFIAS
                if str(foto) != st.session_state.foto_seleccionada
            ]

            if fotos_disponibles:

                st.session_state.foto_seleccionada = str(
                    random.choice(fotos_disponibles)
                )

        # Mostrar nombre de la fotografía
        st.caption(
            "Fotografía seleccionada: "
            + Path(st.session_state.foto_seleccionada).name
        )

        # Abrir fotografía
        imagen = Image.open(
            st.session_state.foto_seleccionada
        ).convert("RGB")

        ancho, alto = imagen.size

        st.image(
            imagen,
            caption="Fotografía seleccionada del campus",
            use_container_width=True
        )

        # =================================
        # 7. CARGAR MODELO Y DETECTAR PERSONAS
        # =================================

        with st.spinner("Cargando inteligencia artificial..."):

            modelo = cargar_modelo()

        with st.spinner("Detectando personas..."):

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

        st.write(
            f"Personas detectadas: **{len(personas)}**"
        )

        # =================================
        # 8. CONFIGURAR LAS 5 MESAS
        # =================================

        st.divider()
        st.subheader("Configuración de las mesas")

        coordenadas_iniciales = [
            ([100, 240], [200, 330]),
            ([430, 540], [220, 320]),
            ([570, 680], [250, 350]),
            ([760, 880], [380, 480]),
            ([240, 500], [380, 520])
        ]

        mesas = {}

        columnas = st.columns(2)

        for i in range(5):

            nombre = f"Mesa {i + 1}"

            x_inicial = coordenadas_iniciales[i][0]
            y_inicial = coordenadas_iniciales[i][1]

            x_min = min(x_inicial[0], ancho - 1)
            x_max = min(x_inicial[1], ancho)

            y_min = min(y_inicial[0], alto - 1)
            y_max = min(y_inicial[1], alto)

            if x_max <= x_min:
                x_min, x_max = 0, ancho

            if y_max <= y_min:
                y_min, y_max = 0, alto

            with columnas[i % 2]:

                with st.expander(
                    nombre,
                    expanded=(i == 0)
                ):

                    x1, x2 = st.slider(
                        f"Coordenadas X - {nombre}",
                        0,
                        ancho,
                        (x_min, x_max),
                        key=f"x_{i}"
                    )

                    y1, y2 = st.slider(
                        f"Coordenadas Y - {nombre}",
                        0,
                        alto,
                        (y_min, y_max),
                        key=f"y_{i}"
                    )

                    mesas[nombre] = [
                        x1, y1, x2, y2
                    ]

        # =================================
        # 9. ANALIZAR DISPONIBILIDAD
        # =================================

        if st.button(
            "ANALIZAR DISPONIBILIDAD",
            type="primary"
        ):

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

            # =================================
            # 10. ESTADISTICAS
            # =================================

            ocupadas = sum(
                estado == "OCUPADA"
                for estado in estados.values()
            )

            disponibles = 5 - ocupadas

            # =================================
            # 11. MAPA ESQUEMATICO
            # =================================

            st.divider()
            st.subheader("Mapa de disponibilidad")

            st.markdown("""
            <style>
            .mesa {
                border-radius: 12px;
                min-height: 115px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                margin: 8px;
                font-weight: bold;
                box-shadow: 0px 3px 8px rgba(0,0,0,0.10);
            }

            .disponible {
                background-color: #bbf7d0;
                border: 3px solid #16a34a;
                color: #166534;
            }

            .ocupada {
                background-color: #fecaca;
                border: 3px solid #dc2626;
                color: #991b1b;
            }

            .mesa span {
                font-size: 22px;
            }

            .mesa small {
                font-size: 13px;
                font-weight: normal;
                margin-top: 6px;
            }

            .leyenda {
                display: flex;
                gap: 25px;
                justify-content: center;
                margin-top: 20px;
                font-size: 14px;
            }

            .punto {
                display: inline-block;
                width: 13px;
                height: 13px;
                border-radius: 50%;
                margin-right: 6px;
            }
            </style>
            """, unsafe_allow_html=True)

            # Distribución de las mesas
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

                                clase = "disponible"
                                simbolo = "✓"

                            else:

                                clase = "ocupada"
                                simbolo = "✕"

                            st.markdown(
                                f"""
                                <div class="mesa {clase}">
                                    <span>{nombre}</span>
                                    <span>{simbolo}</span>
                                    <small>{estado}</small>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                "<div style='height:131px'></div>",
                                unsafe_allow_html=True
                            )

            # Leyenda
            st.markdown("""
            <div class="leyenda">
                <div>
                    <span class="punto" style="background:#16a34a"></span>
                    Disponible
                </div>
                <div>
                    <span class="punto" style="background:#dc2626"></span>
                    Ocupada
                </div>
            </div>
            """, unsafe_allow_html=True)

            # =================================
            # 12. MOSTRAR ESTADISTICAS
            # =================================

            st.divider()
            st.subheader("Estado general")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total de mesas", 5)
            col2.metric("Disponibles", disponibles)
            col3.metric("Ocupadas", ocupadas)

            st.info(
                "El sistema estima la ocupación mediante "
                "la intersección entre personas detectadas "
                "y las zonas definidas para cada mesa."
            )

    else:

        st.error(
            "No se encontraron fotografías. "
            "Agrega imágenes JPG, JPEG o PNG al repositorio "
            "junto a app.py."
        )


# ==========================================================
# PESTAÑA 2: PREDICCION DEL CAMPUS
# ==========================================================

with pestana2:

    st.header("Predicción de ocupación del campus")

    st.write(
        "Estimación de la actividad general del Campus San Joaquín "
        "según el calendario académico UC y la programación "
        "semanal de talleres de Arquitectura."
    )

    st.info(
        "Los porcentajes son estimaciones iniciales basadas en "
        "reglas académicas. No corresponden a mediciones reales "
        "de personas presentes en el campus."
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

    # Encontrar lunes de la semana
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
