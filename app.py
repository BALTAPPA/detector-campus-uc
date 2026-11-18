
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

CARPETA = Path(__file__).parent

# =====================================
# 2. CARGAR FOTOGRAFIAS
# =====================================

def buscar_fotografias(numeros):

    fotografias = []

    for numero in numeros:

        posibles_nombres = [
            f"foto_{numero:02d}.jpg",
            f"foto_{numero:02d}.jpeg",
            f"foto_{numero:02d}.png"
        ]

        for nombre in posibles_nombres:

            ruta = CARPETA / nombre

            if ruta.exists():
                fotografias.append(ruta)
                break

    return fotografias


FOTOS_PATIO_1 = buscar_fotografias([1, 2, 3])
FOTOS_PATIO_2 = buscar_fotografias([4, 5, 6])


# =====================================
# 3. CARGAR MODELO YOLO
# =====================================

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")


# =====================================
# 4. FUNCIONES DE DETECCION
# =====================================

def calcular_interseccion(caja_mesa, caja_persona):

    mx1, my1, mx2, my2 = caja_mesa
    px1, py1, px2, py2 = caja_persona

    return not (
        px2 < mx1 or
        px1 > mx2 or
        py2 < my1 or
        py1 > my2
    )


def analizar_fotografia(ruta_foto, modelo, mesas):

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

        # Ajustar coordenadas a la fotografia

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

        estados[nombre] = (
            "OCUPADA" if ocupada else "DISPONIBLE"
        )

    return estados, len(cajas_personas)


# =====================================
# 5. COORDENADAS DE LAS MESAS
# =====================================

# PATIO 1
# Coordenadas actuales del primer patio

mesas_patio_1 = {
    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]
}


# PATIO 2
# Estas coordenadas son provisionales.
# Hay que ajustarlas a las fotografias del segundo patio.

mesas_patio_2 = {
    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]
}


# =====================================
# 6. FUNCION PARA MOSTRAR CADA PATIO
# =====================================

def mostrar_patio(nombre_patio, fotografias, mesas, clave):

    st.subheader(f"Disponibilidad - {nombre_patio}")

    if len(fotografias) == 0:

        st.error(
            f"No se encontraron fotografias para {nombre_patio}."
        )

        return

    # ---------------------------------
    # SELECCION DE FOTOGRAFIA
    # ---------------------------------

    clave_foto = f"foto_actual_{clave}"
    clave_procesada = f"foto_procesada_{clave}"
    clave_estados = f"estados_mesas_{clave}"
    clave_personas = f"cantidad_personas_{clave}"

    if clave_foto not in st.session_state:

        st.session_state[clave_foto] = random.choice(fotografias)

    col1, col2 = st.columns([1, 3])

    with col1:

        if st.button(
            "🔄 Cambiar fotografia",
            key=f"boton_{clave}"
        ):

            if len(fotografias) > 1:

                opciones = [
                    foto for foto in fotografias
                    if foto != st.session_state[clave_foto]
                ]

                st.session_state[clave_foto] = random.choice(opciones)

            else:

                st.info("Solo hay una fotografia disponible.")

    # ---------------------------------
    # MINIATURA
    # ---------------------------------

    with col1:

        st.image(
            str(st.session_state[clave_foto]),
            width=260
        )

        st.caption(st.session_state[clave_foto].name)

    # ---------------------------------
    # ANALISIS DE LA FOTOGRAFIA
    # ---------------------------------

    modelo = cargar_modelo()

    if (
        clave_procesada not in st.session_state
        or st.session_state[clave_procesada]
        != str(st.session_state[clave_foto])
    ):

        with st.spinner(f"Analizando {nombre_patio}..."):

            estados, cantidad_personas = analizar_fotografia(
                st.session_state[clave_foto],
                modelo,
                mesas
            )

            st.session_state[clave_estados] = estados
            st.session_state[clave_personas] = cantidad_personas

            st.session_state[clave_procesada] = str(
                st.session_state[clave_foto]
            )

    # ---------------------------------
    # RESULTADOS INDEPENDIENTES
    # ---------------------------------

    estados = st.session_state[clave_estados]

    cantidad_personas = st.session_state[clave_personas]

    disponibles = sum(
        1 for estado in estados.values()
        if estado == "DISPONIBLE"
    )

    ocupadas = sum(
        1 for estado in estados.values()
        if estado == "OCUPADA"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total de mesas", len(mesas))
    c2.metric("Mesas disponibles", disponibles)
    c3.metric("Mesas ocupadas", ocupadas)

    st.caption(
        f"Personas detectadas: {cantidad_personas}"
    )

    st.divider()

    # ---------------------------------
    # MAPA DE DISPONIBILIDAD
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

    nombres_mesas = list(mesas.keys())

    # Mostrar las mesas en filas de tres

    for inicio in range(0, len(nombres_mesas), 3):

        columnas = st.columns(3)

        for i, nombre in enumerate(nombres_mesas[inicio:inicio + 3]):

            with columnas[i]:
                mostrar_mesa(nombre)


# =====================================
# 7. PESTAÑAS DE LA APLICACION
# =====================================

pestana1, pestana2, pestana3 = st.tabs([
    "📍 Patio 1",
    "📍 Patio 2",
    "📊 Ocupación del campus"
])


# =====================================
# PESTAÑA 1: PATIO 1
# =====================================

with pestana1:

    mostrar_patio(
        "Patio 1",
        FOTOS_PATIO_1,
        mesas_patio_1,
        "patio1"
    )


# =====================================
# PESTAÑA 2: PATIO 2
# =====================================

with pestana2:

    mostrar_patio(
        "Patio 2",
        FOTOS_PATIO_2,
        mesas_patio_2,
        "patio2"
    )


# =====================================
# PESTAÑA 3: OCUPACION DEL CAMPUS
# =====================================

with pestana3:

    st.subheader("Estimación de ocupación del campus")

    st.write(
        "Estimación referencial de actividad universitaria "
        "según el calendario académico y las actividades de Arquitectura."
    )

    # CALENDARIO ACADEMICO 2026

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

    # ESTIMACION SEGUN DIA

    ocupacion_normal = {
        0: 55,
        1: 90,
        2: 60,
        3: 90,
        4: 80
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

    # FUNCION DE PREDICCION

    def predecir_ocupacion(fecha):

        if fecha < INICIO_PERIODO or fecha > FIN_PERIODO:
            return None

        if fecha.weekday() >= 5:
            return 0

        if fecha in SUSPENSIONES:
            return 0

        if INICIO_RECESO <= fecha <= FIN_RECESO:
            return 10

        if INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:
            return ocupacion_evaluaciones[fecha.weekday()]

        if FIN_EVALUACIONES < fecha <= FIN_PERIODO:
            return 25

        if INICIO_PERIODO <= fecha <= FIN_CLASES:
            return ocupacion_normal[fecha.weekday()]

        return None

    # SELECCIONAR FECHA

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
        "Los porcentajes son estimaciones heurísticas iniciales, "
        "no mediciones reales de ocupación del campus."
    )
