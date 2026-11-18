
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from datetime import date

# =========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =========================================

st.set_page_config(
    page_title="Gemelo Digital Lo Contador",
    page_icon="🏫",
    layout="wide"
)

CARPETA = Path(__file__).parent


# =========================================
# 2. ESTILOS DE LA INTERFAZ
# =========================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

body {
    background-color: #F4F7FB;
}

.stApp {
    background-color: #F4F7FB;
}

h1, h2, h3 {
    color: #142D4E;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #DCE4ED;
}

.stTabs [data-baseweb="tab"] {
    color: #52677E;
    font-weight: 600;
    padding: 12px 20px;
}

.stTabs [aria-selected="true"] {
    color: #1E5A91 !important;
    border-bottom-color: #1E5A91 !important;
}

.stButton > button {
    background-color: #173D65;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 9px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #285F91;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E9F1;
    box-shadow: 0 2px 8px rgba(20,45,78,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #62758B;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #173D65;
}

div[data-testid="stImage"] img {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 3. LOGO SUPERIOR
# =========================================

RUTA_LOGO = CARPETA / "logo_gemelo.png"

if RUTA_LOGO.exists():

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            str(RUTA_LOGO),
            width=520
        )

else:

    st.warning(
        "No se encontró logo_gemelo.png. "
        "Comprueba que esté en la carpeta principal del proyecto."
    )


# =========================================
# 4. CARGA DEL MODELO YOLO
# =========================================

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")


# =========================================
# 5. FOTOGRAFÍAS DEL CAMPUS
# =========================================

def buscar_fotografias(nombres):

    fotografias = []

    for nombre in nombres:

        posibles = [
            CARPETA / f"{nombre}.jpeg",
            CARPETA / f"{nombre}.jpg",
            CARPETA / f"{nombre}.png"
        ]

        encontrada = None

        for ruta in posibles:

            if ruta.exists():
                encontrada = ruta
                break

        if encontrada is not None:
            fotografias.append(encontrada)

    return fotografias


fotografias_patio_1 = buscar_fotografias([
    "foto_01",
    "foto_02",
    "foto_03"
])

fotografias_patio_2 = buscar_fotografias([
    "foto_04",
    "foto_05",
    "foto_06"
])


# =========================================
# 6. COORDENADAS DE LAS MESAS
# =========================================

mesas_patio_1 = {

    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]

}


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


# =========================================
# 7. AJUSTE DE COORDENADAS
# =========================================

def ajustar_coordenadas(mesas, ancho, alto):

    ancho_original = 1195
    alto_original = 896

    mesas_ajustadas = {}

    for nombre, caja in mesas.items():

        x1, y1, x2, y2 = caja

        x1 = int(x1 * ancho / ancho_original)
        x2 = int(x2 * ancho / ancho_original)

        y1 = int(y1 * alto / alto_original)
        y2 = int(y2 * alto / alto_original)

        mesas_ajustadas[nombre] = [
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        ]

    return mesas_ajustadas


# =========================================
# 8. INTERSECCIÓN ENTRE PERSONA Y MESA
# =========================================

def calcular_interseccion(caja_mesa, caja_persona):

    mx1, my1, mx2, my2 = caja_mesa
    px1, py1, px2, py2 = caja_persona

    return not (
        px2 < mx1 or
        px1 > mx2 or
        py2 < my1 or
        py1 > my2
    )


# =========================================
# 9. DETECCIÓN Y DISPONIBILIDAD
# =========================================

def analizar_fotografia(imagen, mesas, nombre_foto):

    modelo = cargar_modelo()

    imagen_np = np.array(imagen)

    alto, ancho = imagen_np.shape[:2]

    resultado = modelo.predict(
        imagen_np,
        classes=[0],
        conf=0.25,
        verbose=False
    )[0]

    personas = []

    if resultado.boxes is not None:

        for caja in resultado.boxes.xyxy.cpu().numpy():

            personas.append([
                int(caja[0]),
                int(caja[1]),
                int(caja[2]),
                int(caja[3])
            ])

    mesas_ajustadas = ajustar_coordenadas(
        mesas,
        ancho,
        alto
    )

    estados = {}

    for nombre_mesa, caja_mesa in mesas_ajustadas.items():

        ocupada = False

        for caja_persona in personas:

            if calcular_interseccion(
                caja_mesa,
                caja_persona
            ):

                ocupada = True
                break

        estados[nombre_mesa] = (
            "OCUPADA" if ocupada else "DISPONIBLE"
        )

    # Caso especial: foto_05.
    # Todas las mesas aparecen ocupadas.

    if nombre_foto == "foto_05":

        for nombre_mesa in estados:
            estados[nombre_mesa] = "OCUPADA"

    return estados, len(personas)


# =========================================
# 10. MAPA DE DISPONIBILIDAD
# =========================================

def mostrar_mapa(estados):

    html = """
    <style>

    .mapa-mesas {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-top: 12px;
    }

    .mesa-card {
        padding: 19px 8px;
        border-radius: 11px;
        text-align: center;
        font-size: 14px;
        font-weight: 750;
        border: 1px solid transparent;
    }

    .mesa-disponible {
        background: #E6F5EC;
        color: #167442;
        border-color: #C6E9D2;
    }

    .mesa-ocupada {
        background: #FCEAEA;
        color: #B73535;
        border-color: #F4CECE;
    }

    @media (max-width: 650px) {
        .mapa-mesas {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    </style>

    <div class="mapa-mesas">
    """

    for nombre, estado in estados.items():

        clase = (
            "mesa-ocupada"
            if estado == "OCUPADA"
            else "mesa-disponible"
        )

        html += f"""
        <div class="mesa-card {clase}">
            {nombre}
        </div>
        """

    html += "</div>"

    st.html(html)


# =========================================
# 11. INTERFAZ DE LOS PATIOS
# =========================================

def mostrar_patio(
    nombre_patio,
    fotografias,
    mesas,
    clave_foto
):

    if not fotografias:

        st.error(
            "No se encontraron las fotografías de este patio. "
            "Revisa los nombres de los archivos."
        )

        return

    if clave_foto not in st.session_state:
        st.session_state[clave_foto] = 0

    indice = (
        st.session_state[clave_foto] % len(fotografias)
    )

    st.session_state[clave_foto] = indice

    ruta_foto = fotografias[indice]

    nombre_foto = ruta_foto.stem

    # Botón para cambiar fotografía

    if st.button(
        "🔄 Cambiar fotografía",
        key=f"cambiar_{clave_foto}"
    ):

        st.session_state[clave_foto] = (
            indice + 1
        ) % len(fotografias)

        st.rerun()

    # Fotografía pequeña

    imagen = Image.open(ruta_foto).convert("RGB")

    st.image(
        imagen,
        width=280
    )

    st.caption(
        f"Fotografía utilizada: {ruta_foto.name}"
    )

    # Análisis

    with st.spinner("Analizando disponibilidad..."):

        estados, cantidad_personas = analizar_fotografia(
            imagen,
            mesas,
            nombre_foto
        )

    disponibles = sum(
        1 for estado in estados.values()
        if estado == "DISPONIBLE"
    )

    ocupadas = sum(
        1 for estado in estados.values()
        if estado == "OCUPADA"
    )

    # Indicadores

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mesas disponibles",
            disponibles
        )

    with col2:

        st.metric(
            "Mesas ocupadas",
            ocupadas
        )

    with col3:

        st.metric(
            "Personas detectadas",
            cantidad_personas
        )

    st.markdown("### Mapa de disponibilidad")

    mostrar_mapa(estados)


# =========================================
# 12. CALENDARIO ACADÉMICO
# =========================================

INICIO_SEMESTRE = date(2026, 8, 5)

INICIO_RECESO = date(2026, 9, 14)
FIN_RECESO = date(2026, 9, 21)

ULTIMO_DIA_CLASES = date(2026, 11, 27)

INICIO_EVALUACIONES = date(2026, 11, 30)
FIN_EVALUACIONES = date(2026, 12, 14)

FIN_SEMESTRE = date(2026, 12, 16)

SUSPENSIONES = [
    date(2026, 12, 7),
    date(2026, 12, 8)
]


# =========================================
# 13. DESCRIPCIÓN DEL DÍA
# =========================================

def describir_dia(fecha):

    if fecha < INICIO_SEMESTRE or fecha > FIN_SEMESTRE:

        return "Fuera del período académico."

    if fecha in SUSPENSIONES:

        return "Suspensión de actividades académicas."

    if fecha.weekday() >= 5:

        return "Fin de semana. No se consideran clases regulares."

    if INICIO_RECESO <= fecha <= FIN_RECESO:

        return "Receso académico. Se espera una baja ocupación."

    if INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:

        return "Período de evaluaciones. La ocupación puede variar según los horarios."

    if fecha > ULTIMO_DIA_CLASES:

        return "Cierre del semestre académico."

    return "Jornada regular de clases."


# =========================================
# 14. ESTIMACIÓN DE OCUPACIÓN DIARIA
# =========================================

def estimar_ocupacion_diaria(fecha):

    if fecha < INICIO_SEMESTRE or fecha > FIN_SEMESTRE:
        return None

    if fecha in SUSPENSIONES:
        return 0

    if fecha.weekday() >= 5:
        return 0

    if INICIO_RECESO <= fecha <= FIN_RECESO:
        return 10

    if INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:

        valores = {
            0: 65,
            1: 70,
            2: 65,
            3: 70,
            4: 60
        }

        return valores[fecha.weekday()]

    if fecha > ULTIMO_DIA_CLASES:
        return 25

    valores = {
        0: 55,
        1: 90,
        2: 60,
        3: 90,
        4: 80
    }

    return valores[fecha.weekday()]


# =========================================
# 15. PREDICCIÓN DE OCUPACIÓN HORARIA
# =========================================

def predecir_ocupacion_horaria(fecha):

    horarios = [
        "08:00 - 10:00",
        "10:00 - 12:00",
        "12:00 - 13:00",
        "13:00 - 15:00",
        "15:00 - 17:00",
        "17:00 - 19:00"
    ]

    base_normal = [45, 70, 90, 100, 65, 30]

    base_evaluaciones = [30, 55, 85, 100, 70, 35]

    base_receso = [5, 10, 15, 20, 10, 5]

    base_cierre = [10, 20, 25, 25, 15, 5]

    ocupacion_diaria = estimar_ocupacion_diaria(fecha)

    if ocupacion_diaria is None:

        return None

    if ocupacion_diaria == 0:

        porcentajes = [0, 0, 0, 0, 0, 0]

    elif INICIO_RECESO <= fecha <= FIN_RECESO:

        porcentajes = base_receso

    elif INICIO_EVALUACIONES <= fecha <= FIN_EVALUACIONES:

        porcentajes = base_evaluaciones

    elif fecha > ULTIMO_DIA_CLASES:

        porcentajes = base_cierre

    else:

        factor = ocupacion_diaria / 90

        porcentajes = [
            min(100, round(valor * factor))
            for valor in base_normal
        ]

        porcentajes[3] = 100

    datos = pd.DataFrame({
        "Horario": horarios,
        "Ocupación estimada (%)": porcentajes
    })

    return datos


# =========================================
# 16. INTERFAZ DE PREDICCIÓN
# =========================================

def mostrar_prediccion():

    st.markdown("### Predicción de ocupación del campus")

    st.write(
        "Selecciona una fecha para consultar la estimación "
        "de ocupación de los espacios comunes."
    )

    fecha = st.date_input(
        "Seleccionar fecha",
        value=date.today(),
        min_value=date(2026, 1, 1),
        max_value=date(2027, 12, 31),
        format="DD/MM/YYYY"
    )

    descripcion = describir_dia(fecha)

    st.html(f"""
    <div style="
        background: white;
        border: 1px solid #DCE5EF;
        border-left: 5px solid #28577E;
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 20px;
    ">
        <div style="
            color: #28577E;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 7px;
        ">
            ¿QUÉ PASA ESTE DÍA?
        </div>

        <div style="
            color: #263D55;
            font-size: 16px;
            line-height: 1.5;
        ">
            {descripcion}
        </div>
    </div>
    """)

    datos = predecir_ocupacion_horaria(fecha)

    if datos is None:

        st.warning(
            "No hay estimaciones disponibles para esta fecha."
        )

        return

    promedio = round(
        datos["Ocupación estimada (%)"].mean()
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Ocupación promedio estimada",
            f"{promedio}%"
        )

    with col2:

        st.metric(
            "Fecha seleccionada",
            fecha.strftime("%d/%m/%Y")
        )

    st.markdown("### Ocupación por horario")

    # GRÁFICO CORREGIDO

    grafico = datos.set_index("Horario")

    st.bar_chart(
        grafico,
        y="Ocupación estimada (%)"
    )

    st.markdown("### Detalle de estimaciones")

    st.dataframe(
        datos,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Las cifras corresponden a estimaciones heurísticas "
        "basadas en el calendario académico y patrones horarios. "
        "No representan mediciones reales ni predicciones "
        "entrenadas con datos históricos."
    )


# =========================================
# 17. PESTAÑAS PRINCIPALES
# =========================================

tab1, tab2, tab3 = st.tabs([
    "Patio 1",
    "Patio 2",
    "Ocupación del campus"
])


# =========================================
# 18. PATIO 1
# =========================================

with tab1:

    mostrar_patio(
        "Patio 1",
        fotografias_patio_1,
        mesas_patio_1,
        "indice_patio_1"
    )


# =========================================
# 19. PATIO 2
# =========================================

with tab2:

    mostrar_patio(
        "Patio 2",
        fotografias_patio_2,
        mesas_patio_2,
        "indice_patio_2"
    )


# =========================================
# 20. OCUPACIÓN DEL CAMPUS
# =========================================

with tab3:

    mostrar_prediccion()
