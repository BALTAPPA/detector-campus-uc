
import streamlit as st
import random
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from datetime import date

# =====================================
# 1. CONFIGURACION DE LA PAGINA
# =====================================

st.set_page_config(
    page_title="Detector Campus UC",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Detector de disponibilidad - Campus UC")

# =====================================
# 2. FUNCIONES DEL MODELO
# =====================================

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")


def calcular_interseccion(caja_mesa, caja_persona):
    """
    Comprueba si una persona se superpone con una mesa.
    """

    mx1, my1, mx2, my2 = caja_mesa
    px1, py1, px2, py2 = caja_persona

    return not (
        px2 < mx1 or
        px1 > mx2 or
        py2 < my1 or
        py1 > my2
    )


def analizar_fotografia(ruta_foto, modelo, mesas):
    """
    Detecta personas en la fotografía y determina
    si cada mesa está ocupada o disponible.
    """

    imagen = Image.open(ruta_foto).convert("RGB")
    ancho, alto = imagen.size

    resultados = modelo.predict(
        imagen,
        classes=[0],
        conf=0.25,
        verbose=False
    )

    cajas_personas = []

    for resultado in resultados:
        for caja in resultado.boxes.xyxy.tolist():
            cajas_personas.append(caja)

    estados = {}

    for nombre, coordenadas in mesas.items():

        x1, y1, x2, y2 = coordenadas

        # Ajustar coordenadas a las dimensiones de la foto
        x1 = max(0, min(x1, ancho))
        x2 = max(0, min(x2, ancho))
        y1 = max(0, min(y1, alto))
        y2 = max(0, min(y2, alto))

        caja_mesa = [x1, y1, x2, y2]

        ocupada = False

        for caja_persona in cajas_personas:
            if calcular_interseccion(caja_mesa, caja_persona):
                ocupada = True
                break

        estados[nombre] = "OCUPADA" if ocupada else "DISPONIBLE"

    return estados, len(cajas_personas)


# =====================================
# 3. FOTOGRAFIAS DEL CAMPUS
# =====================================

CARPETA = Path(__file__).parent

FOTOGRAFIAS = [
    archivo for archivo in CARPETA.iterdir()
    if archivo.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

# =====================================
# 4. DISTRIBUCION DE LAS MESAS
# =====================================

mesas = {
    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]
}


# =====================================
# 5. PESTAÑAS
# =====================================

pestana1, pestana2 = st.tabs([
    "📍 Disponibilidad de mesas",
    "📊 Ocupación del campus"
])


# =====================================
# PESTAÑA 1: DISPONIBILIDAD DE MESAS
# =====================================

with pestana1:

    st.subheader("Mapa de disponibilidad")

    if len(FOTOGRAFIAS) == 0:

        st.error(
            "No se encontraron fotografías. "
            "Sube imágenes JPG, JPEG o PNG al repositorio."
        )

    else:

        # ---------------------------------
        # Seleccionar fotografia inicial
        # ---------------------------------

        if "foto_actual" not in st.session_state:
            st.session_state.foto_actual = random.choice(FOTOGRAFIAS)

        # ---------------------------------
        # Boton para cambiar fotografia
        # ---------------------------------

        col1, col2 = st.columns([1, 3])

        with col1:

            if st.button("🔄 Cambiar fotografía"):

                if len(FOTOGRAFIAS) > 1:

                    opciones = [
                        foto for foto in FOTOGRAFIAS
                        if foto != st.session_state.foto_actual
                    ]

                    st.session_state.foto_actual = random.choice(opciones)

                else:
                    st.info("Solo hay una fotografía disponible.")

        # ---------------------------------
        # Mostrar fotografia pequeña
        # ---------------------------------

        with col1:

            st.image(
                str(st.session_state.foto_actual),
                width=260
            )

            st.caption(st.session_state.foto_actual.name)

        # ---------------------------------
        # Analizar fotografia
        # ---------------------------------

        modelo = cargar_modelo()

        if (
            "foto_procesada" not in st.session_state
            or st.session_state.foto_procesada != str(st.session_state.foto_actual)
        ):

            with st.spinner("Analizando fotografía..."):

                estados, cantidad_personas = analizar_fotografia(
                    st.session_state.foto_actual,
                    modelo,
                    mesas
                )

                st.session_state.estados_mesas = estados
                st.session_state.cantidad_personas = cantidad_personas
                st.session_state.foto_procesada = str(
                    st.session_state.foto_actual
                )

        # ---------------------------------
        # Obtener resultados
        # ---------------------------------

        estados = st.session_state.estados_mesas
        cantidad_personas = st.session_state.cantidad_personas

        disponibles = sum(
            1 for estado in estados.values()
            if estado == "DISPONIBLE"
        )

        ocupadas = sum(
            1 for estado in estados.values()
            if estado == "OCUPADA"
        )

        # ---------------------------------
        # Indicadores
        # ---------------------------------

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total de mesas", len(mesas))
        c2.metric("Mesas disponibles", disponibles)
        c3.metric("Mesas ocupadas", ocupadas)

        st.caption(
            f"Personas detectadas en la fotografía: {cantidad_personas}"
        )

        st.divider()

        # ---------------------------------
        # MAPA DE MESAS
        # ---------------------------------

        st.subheader("Mapa de disponibilidad")

        def mostrar_mesa(nombre_mesa):

            estado = estados[nombre_mesa]

            if estado == "DISPONIBLE":
                color = "#238B45"
            else:
                color = "#C0392B"

            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    padding: 25px 10px;
                    border-radius: 12px;
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                    margin-bottom: 15px;
                ">
                    {nombre_mesa}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Primera fila: 3 mesas

        col1, col2, col3 = st.columns(3)

        with col1:
            mostrar_mesa("MESA 1")

        with col2:
            mostrar_mesa("MESA 2")

        with col3:
            mostrar_mesa("MESA 3")

        # Segunda fila: 2 mesas

        col4, col5, col6 = st.columns(3)

        with col4:
            mostrar_mesa("MESA 4")

        with col5:
            mostrar_mesa("MESA 5")


# =====================================
# PESTAÑA 2: OCUPACION DEL CAMPUS
# =====================================

with pestana2:

    st.subheader("Estimación de ocupación del campus")

    st.write(
        "Estimación referencial de actividad universitaria "
        "según el calendario académico y las actividades de Arquitectura."
    )

    # ---------------------------------
    # CALENDARIO ACADEMICO UC 2026
    # ---------------------------------

    INICIO_PERIODO = date(2026, 8, 5)
    INICIO_RECESO = date(2026, 9, 14)
    FIN_RECESO = date(2026, 9, 21)

    FIN_CLASES = date(2026, 11, 27)

    INICIO_EVALUACIONES = date(2026, 11, 30)
    FIN_EVALUACIONES = date(2026, 12, 14)

    FIN_PERIODO = date(2026, 12, 16)

    SUSPENSIONES = [
        date(2026, 12, 7),
        date(2026, 12, 8)
    ]

    # ---------------------------------
    # ESTIMACION SEGUN DIA
    # ---------------------------------

    ocupacion_normal = {
        0: 55,  # Lunes
        1: 90,  # Martes
        2: 60,  # Miercoles
        3: 90,  # Jueves
        4: 80   # Viernes
    }

    ocupacion_evaluaciones = {
        0: 65,
        1: 70,
        2: 65,
        3: 70,
        4: 60
    }

    nombres_dias = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes"
    ]

    # ---------------------------------
    # FUNCION DE PREDICCION
    # ---------------------------------

    def predecir_ocupacion(fecha):

        # Fuera del periodo academico
        if fecha < INICIO_PERIODO or fecha > FIN_PERIODO:
            return None

        # Fines de semana
        if fecha.weekday() >= 5:
            return 0

        # Suspensiones y feriados
        if fecha in SUSPENSIONES:
            return 0

        # Receso academico
        if INICIO_RECESO <= fecha <= FIN_RECESO:
            return 10

        # Periodo de evaluaciones
        if INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:
            return ocupacion_evaluaciones[fecha.weekday()]

        # Dias posteriores al periodo de evaluaciones
        if FIN_EVALUACIONES < fecha <= FIN_PERIODO:
            return 25

        # Periodo normal de clases
        if INICIO_PERIODO <= fecha <= FIN_CLASES:
            return ocupacion_normal[fecha.weekday()]

        return None

    # ---------------------------------
    # SELECCIONAR FECHA
    # ---------------------------------

    fecha_seleccionada = st.date_input(
        "Selecciona una fecha",
        value=date.today(),
        min_value=date(2026, 1, 1),
        max_value=date(2027, 12, 31)
    )

    resultado = predecir_ocupacion(fecha_seleccionada)

    st.divider()

    if resultado is None:

        st.warning(
            "La fecha seleccionada está fuera del periodo académico considerado."
        )

    else:

        st.metric(
            "Ocupación estimada del campus",
            f"{resultado}%"
        )

        st.progress(resultado / 100)

        if resultado >= 80:
            st.error("Alta actividad estimada en el campus.")

        elif resultado >= 50:
            st.warning("Actividad media estimada en el campus.")

        elif resultado > 0:
            st.info("Baja actividad estimada en el campus.")

        else:
            st.info("Sin actividad académica estimada.")

        st.write(
            f"**Día:** {nombres_dias[fecha_seleccionada.weekday()]}"
        )

        st.write(
            f"**Fecha:** {fecha_seleccionada.strftime('%d-%m-%Y')}"
        )

    st.divider()

    st.caption(
        "Nota: estos porcentajes son estimaciones heurísticas iniciales, "
        "no mediciones reales de ocupación. Deben calibrarse con observaciones "
        "y registros del campus."
    )
