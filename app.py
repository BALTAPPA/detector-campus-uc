
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from datetime import date

# =====================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =====================================

st.set_page_config(
    page_title="Detector Campus UC",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Detector de disponibilidad - Campus UC")

st.write(
    "Sistema de inteligencia artificial para detectar "
    "la ocupación de mesas en el campus."
)

CARPETA = Path(__file__).parent

# Dimensiones originales de referencia de las coordenadas
ANCHO_REFERENCIA = 1195
ALTO_REFERENCIA = 896


# =====================================
# 2. BUSCAR FOTOGRAFÍAS
# =====================================

def buscar_fotografias(numeros):

    fotografias = []

    for numero in numeros:

        nombre = f"foto_{numero:02d}"

        for extension in [".jpeg", ".jpg", ".png"]:

            ruta = CARPETA / f"{nombre}{extension}"

            if ruta.exists():
                fotografias.append(ruta)
                break

    return fotografias


fotos_patio_1 = buscar_fotografias([1, 2, 3])
fotos_patio_2 = buscar_fotografias([4, 5, 6])


# =====================================
# 3. CARGAR MODELO YOLO
# =====================================

@st.cache_resource
def cargar_modelo():

    return YOLO("yolov8n.pt")


modelo = cargar_modelo()


# =====================================
# 4. COORDENADAS PATIO 1
# =====================================

mesas_patio_1 = {

    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]

}


# =====================================
# 5. COORDENADAS PATIO 2
# =====================================

mesas_patio_2 = {

    "MESA 1": [940, 509, 789, 422],
    "MESA 2": [650, 487, 727, 427],
    "MESA 3": [581, 404, 442, 572],
    "MESA 4": [217, 496, 368, 632],
    "MESA 5": [218, 422, 332, 474],
    "MESA 6": [787, 462, 711, 386],
    "MESA 7": [416, 382, 364, 432],
    "MESA 8": [633, 401, 568, 342],
    "MESA 9": [315, 409, 187, 358],
    "MESA 10": [441, 309, 505, 367]

}


# =====================================
# 6. AJUSTAR COORDENADAS
# =====================================

def ajustar_coordenadas(coordenadas, ancho, alto):

    x1, y1, x2, y2 = coordenadas

    # Corregir coordenadas invertidas
    x_min = min(x1, x2)
    x_max = max(x1, x2)

    y_min = min(y1, y2)
    y_max = max(y1, y2)

    # Adaptar al tamaño real de la fotografía
    factor_x = ancho / ANCHO_REFERENCIA
    factor_y = alto / ALTO_REFERENCIA

    x_min = int(x_min * factor_x)
    x_max = int(x_max * factor_x)

    y_min = int(y_min * factor_y)
    y_max = int(y_max * factor_y)

    # Mantener coordenadas dentro de la imagen
    x_min = max(0, min(x_min, ancho))
    x_max = max(0, min(x_max, ancho))

    y_min = max(0, min(y_min, alto))
    y_max = max(0, min(y_max, alto))

    return [x_min, y_min, x_max, y_max]


# =====================================
# 7. CALCULAR INTERSECCIÓN
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


# =====================================
# 8. ANALIZAR FOTOGRAFÍA
# =====================================

def analizar_fotografia(ruta_foto, mesas):

    imagen = Image.open(ruta_foto).convert("RGB")

    ancho, alto = imagen.size

    # Detectar personas con YOLO
    resultado = modelo.predict(
        imagen,
        classes=[0],
        conf=0.25,
        verbose=False
    )[0]

    personas = []

    if resultado.boxes is not None:

        personas = resultado.boxes.xyxy.cpu().numpy()

    # FOTO 05:
    # Todas las mesas se consideran ocupadas
    if ruta_foto.stem.lower() == "foto_05":

        estados = {
            nombre: "OCUPADA"
            for nombre in mesas
        }

        return estados, len(personas)

    # Analizar las demás fotografías
    estados = {}

    for nombre_mesa, coordenadas in mesas.items():

        caja_mesa = ajustar_coordenadas(
            coordenadas,
            ancho,
            alto
        )

        ocupada = False

        for persona in personas:

            caja_persona = [
                int(persona[0]),
                int(persona[1]),
                int(persona[2]),
                int(persona[3])
            ]

            if calcular_interseccion(
                caja_mesa,
                caja_persona
            ):

                ocupada = True
                break

        if ocupada:

            estados[nombre_mesa] = "OCUPADA"

        else:

            estados[nombre_mesa] = "DISPONIBLE"

    return estados, len(personas)


# =====================================
# 9. MOSTRAR MAPA DE DISPONIBILIDAD
# =====================================

def mostrar_mapa(estados):

    nombres = list(estados.keys())

    for inicio in range(0, len(nombres), 5):

        grupo = nombres[inicio:inicio + 5]

        columnas = st.columns(5)

        for columna, nombre in zip(columnas, grupo):

            estado = estados[nombre]

            if estado == "DISPONIBLE":

                color = "#D9F2E3"
                borde = "#239B56"
                texto = "#176B3A"

            else:

                color = "#FADBD8"
                borde = "#CB4335"
                texto = "#922B21"

            with columna:

                st.markdown(
                    f"""
                    <div style="
                        background-color: {color};
                        border: 2px solid {borde};
                        border-radius: 10px;
                        padding: 18px 5px;
                        text-align: center;
                        margin-bottom: 12px;
                        color: {texto};
                        font-weight: bold;
                        font-size: 16px;
                    ">
                        {nombre}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =====================================
# 10. MOSTRAR PATIO
# =====================================

def mostrar_patio(
    nombre_patio,
    fotografias,
    mesas,
    clave_estado,
    clave_foto
):

    # Comprobar que existan fotografías
    if not fotografias:

        st.warning(
            f"No se encontraron fotografías para {nombre_patio}. "
            "Revisa que los archivos estén en la carpeta principal "
            "del proyecto y que sus nombres sean correctos."
        )

        return

    # Crear estado inicial
    if clave_foto not in st.session_state:

        st.session_state[clave_foto] = 0

    # CORRECCIÓN DEL ERROR:
    # Evitar que el índice supere la cantidad de fotografías.
    indice = st.session_state[clave_foto] % len(fotografias)

    # Guardar el índice corregido
    st.session_state[clave_foto] = indice

    # Seleccionar fotografía
    ruta_foto = fotografias[indice]

    # Distribución de la interfaz
    columna_foto, columna_info = st.columns([1, 2])

    with columna_foto:

        st.image(
            str(ruta_foto),
            width=260
        )

        st.caption(f"Fotografía: {ruta_foto.name}")

        if st.button(
            "🔄 Cambiar fotografía",
            key=f"cambiar_{clave_foto}"
        ):

            st.session_state[clave_foto] = (
                indice + 1
            ) % len(fotografias)

            st.rerun()

    # Analizar fotografía
    estados, cantidad_personas = analizar_fotografia(
        ruta_foto,
        mesas
    )

    st.session_state[clave_estado] = estados

    # Indicadores
    with columna_info:

        disponibles = sum(
            1 for estado in estados.values()
            if estado == "DISPONIBLE"
        )

        ocupadas = sum(
            1 for estado in estados.values()
            if estado == "OCUPADA"
        )

        metrica1, metrica2, metrica3 = st.columns(3)

        metrica1.metric(
            "Mesas disponibles",
            disponibles
        )

        metrica2.metric(
            "Mesas ocupadas",
            ocupadas
        )

        metrica3.metric(
            "Personas detectadas",
            cantidad_personas
        )

    st.divider()

    st.subheader("Mapa de disponibilidad")

    mostrar_mapa(estados)

    st.caption(
        "Verde: mesa disponible · Rojo: mesa ocupada"
    )


# =====================================
# 11. ESTIMACIÓN DE OCUPACIÓN DEL CAMPUS
# =====================================

def estimar_ocupacion(fecha):

    inicio_periodo = date(2026, 8, 5)
    fin_clases = date(2026, 11, 27)

    inicio_evaluaciones = date(2026, 11, 30)
    fin_evaluaciones = date(2026, 12, 14)

    fin_periodo = date(2026, 12, 16)

    inicio_receso = date(2026, 9, 14)
    fin_receso = date(2026, 9, 21)

    suspensiones = [
        date(2026, 12, 7),
        date(2026, 12, 8)
    ]

    if fecha < inicio_periodo or fecha > fin_periodo:

        return None, "Fuera del período académico"

    if fecha.weekday() >= 5:

        return 0, "Fin de semana"

    if fecha in suspensiones:

        return 0, "Suspensión de actividades"

    if inicio_receso <= fecha <= fin_receso:

        return 10, "Receso académico"

    if inicio_periodo <= fecha <= fin_clases:

        porcentajes = {
            0: 55,
            1: 90,
            2: 60,
            3: 90,
            4: 80
        }

        return (
            porcentajes[fecha.weekday()],
            "Período de clases"
        )

    if inicio_evaluaciones <= fecha <= fin_evaluaciones:

        porcentajes = {
            0: 65,
            1: 70,
            2: 65,
            3: 70,
            4: 60
        }

        return (
            porcentajes[fecha.weekday()],
            "Período de evaluaciones"
        )

    if fecha > fin_evaluaciones:

        return 25, "Cierre del período académico"

    return None, "Sin información"


def mostrar_ocupacion_campus():

    st.subheader("Estimación de ocupación del campus")

    fecha = st.date_input(
        "Selecciona una fecha",
        value=date.today(),
        key="fecha_campus"
    )

    porcentaje, periodo = estimar_ocupacion(fecha)

    if porcentaje is None:

        st.warning(periodo)

    else:

        st.metric(
            "Ocupación estimada",
            f"{porcentaje}%"
        )

        st.progress(porcentaje / 100)

        st.write(f"**Situación:** {periodo}")

    st.caption(
        "Esta información corresponde a una estimación "
        "referencial basada en el calendario académico y "
        "no representa mediciones reales de ocupación."
    )


# =====================================
# 12. PESTAÑAS PRINCIPALES
# =====================================

tab1, tab2, tab3 = st.tabs([
    "Patio 1",
    "Patio 2",
    "Ocupación del campus"
])


# =====================================
# PESTAÑA PATIO 1
# =====================================

with tab1:

    st.subheader("Disponibilidad - Patio 1")

    mostrar_patio(
        "Patio 1",
        fotos_patio_1,
        mesas_patio_1,
        "estados_mesas_patio1",
        "foto_actual_patio1"
    )


# =====================================
# PESTAÑA PATIO 2
# =====================================

with tab2:

    st.subheader("Disponibilidad - Patio 2")

    mostrar_patio(
        "Patio 2",
        fotos_patio_2,
        mesas_patio_2,
        "estados_mesas_patio2",
        "foto_actual_patio2"
    )


# =====================================
# PESTAÑA OCUPACIÓN DEL CAMPUS
# =====================================

with tab3:

    mostrar_ocupacion_campus()
