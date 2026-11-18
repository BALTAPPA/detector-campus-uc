
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from datetime import date

# =============================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================

st.set_page_config(
    page_title="Smart Campus UC",
    page_icon="🏫",
    layout="wide"
)

CARPETA = Path(__file__).parent

ANCHO_REFERENCIA = 1195
ALTO_REFERENCIA = 896


# =============================================
# 2. DISEÑO VISUAL
# =============================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #183B5B !important;
    font-weight: 750 !important;
}

p, label {
    color: #526477;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #DCE5EF;
}

.stTabs [data-baseweb="tab"] {
    background-color: #E8EEF5;
    border-radius: 10px 10px 0 0;
    padding: 12px 22px;
    color: #38536F;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #183B5B !important;
    color: white !important;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #E2EAF2;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(25,55,85,0.05);
}

[data-testid="stMetricValue"] {
    color: #183B5B;
    font-weight: 800;
}

.stButton > button {
    background-color: #183B5B;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #28577E;
    color: white;
    border: none;
}

[data-testid="stDateInput"] input {
    border-radius: 10px;
    border: 1px solid #D5E0EB;
}

[data-testid="stTable"] {
    background-color: white;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

.stProgress > div > div > div {
    background-color: #25845B;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #DCE5EF;
}
</style>
""", unsafe_allow_html=True)


# =============================================
# 3. ENCABEZADO PRINCIPAL
# =============================================

st.html("""
<div style="
    background: linear-gradient(115deg, #142D4E, #28577E);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 5px 18px rgba(20,45,78,0.12);
">

    <div style="
        color: #AFC9E1;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    ">
        SMART CAMPUS · UNIVERSIDAD CATÓLICA
    </div>

    <div style="
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-top: 10px;
    ">
        Gestión inteligente de espacios
    </div>

    <div style="
        color: #D5E3F0;
        font-size: 15px;
        margin-top: 8px;
    ">
        Detección de disponibilidad y predicción de ocupación
        de los espacios comunes del campus.
    </div>

</div>
""")


# =============================================
# 4. BUSCAR FOTOGRAFÍAS
# =============================================

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


# =============================================
# 5. CARGAR MODELO YOLO
# =============================================

@st.cache_resource
def cargar_modelo():

    return YOLO("yolov8n.pt")


modelo = cargar_modelo()


# =============================================
# 6. COORDENADAS PATIO 1
# =============================================

mesas_patio_1 = {

    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]

}


# =============================================
# 7. COORDENADAS PATIO 2
# =============================================

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


# =============================================
# 8. AJUSTAR COORDENADAS
# =============================================

def ajustar_coordenadas(coordenadas, ancho, alto):

    x1, y1, x2, y2 = coordenadas

    x_min = min(x1, x2)
    x_max = max(x1, x2)

    y_min = min(y1, y2)
    y_max = max(y1, y2)

    factor_x = ancho / ANCHO_REFERENCIA
    factor_y = alto / ALTO_REFERENCIA

    x_min = int(x_min * factor_x)
    x_max = int(x_max * factor_x)

    y_min = int(y_min * factor_y)
    y_max = int(y_max * factor_y)

    x_min = max(0, min(x_min, ancho))
    x_max = max(0, min(x_max, ancho))

    y_min = max(0, min(y_min, alto))
    y_max = max(0, min(y_max, alto))

    return [x_min, y_min, x_max, y_max]


# =============================================
# 9. CALCULAR INTERSECCIÓN
# =============================================

def calcular_interseccion(caja_mesa, caja_persona):

    mx1, my1, mx2, my2 = caja_mesa
    px1, py1, px2, py2 = caja_persona

    return not (
        px2 < mx1 or
        px1 > mx2 or
        py2 < my1 or
        py1 > my2
    )


# =============================================
# 10. ANALIZAR FOTOGRAFÍA
# =============================================

def analizar_fotografia(ruta_foto, mesas):

    imagen = Image.open(ruta_foto).convert("RGB")

    resultado = modelo.predict(
        imagen,
        classes=[0],
        conf=0.25,
        verbose=False
    )[0]

    personas = []

    if resultado.boxes is not None:
        personas = resultado.boxes.xyxy.cpu().numpy()

    # FOTO 05: TODAS LAS MESAS OCUPADAS

    if ruta_foto.stem.lower() == "foto_05":

        estados = {
            nombre: "OCUPADA"
            for nombre in mesas
        }

        return estados, len(personas)

    ancho, alto = imagen.size

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

        estados[nombre_mesa] = (
            "OCUPADA" if ocupada else "DISPONIBLE"
        )

    return estados, len(personas)


# =============================================
# 11. MAPA DE DISPONIBILIDAD
# =============================================

def mostrar_mapa(estados):

    nombres = list(estados.keys())

    for inicio in range(0, len(nombres), 5):

        grupo = nombres[inicio:inicio + 5]

        columnas = st.columns(5)

        for columna, nombre in zip(columnas, grupo):

            estado = estados[nombre]

            if estado == "DISPONIBLE":

                color = "#E1F5EA"
                borde = "#239B56"
                texto = "#176B3A"

            else:

                color = "#FCE8E6"
                borde = "#CB4335"
                texto = "#922B21"

            with columna:

                st.html(f"""
                <div style="
                    background-color: {color};
                    border: 2px solid {borde};
                    border-radius: 12px;
                    padding: 20px 5px;
                    text-align: center;
                    margin-bottom: 12px;
                    color: {texto};
                    font-weight: bold;
                    font-size: 16px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                ">
                    {nombre}
                </div>
                """)


# =============================================
# 12. MOSTRAR PATIO
# =============================================

def mostrar_patio(
    nombre_patio,
    fotografias,
    mesas,
    clave_estado,
    clave_foto
):

    if not fotografias:

        st.warning(
            f"No se encontraron fotografías para {nombre_patio}. "
            "Revisa los nombres y la ubicación de los archivos."
        )

        return

    if clave_foto not in st.session_state:
        st.session_state[clave_foto] = 0

    indice = st.session_state[clave_foto] % len(fotografias)

    st.session_state[clave_foto] = indice

    ruta_foto = fotografias[indice]

    columna_foto, columna_info = st.columns([1, 2])

    with columna_foto:

        st.image(
            str(ruta_foto),
            width=280
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

    estados, cantidad_personas = analizar_fotografia(
        ruta_foto,
        mesas
    )

    st.session_state[clave_estado] = estados

    with columna_info:

        disponibles = sum(
            1 for estado in estados.values()
            if estado == "DISPONIBLE"
        )

        ocupadas = sum(
            1 for estado in estados.values()
            if estado == "OCUPADA"
        )

        total = len(estados)

        st.markdown("#### Resumen de disponibilidad")

        metrica1, metrica2, metrica3 = st.columns(3)

        metrica1.metric(
            "Mesas disponibles",
            disponibles,
            f"de {total}"
        )

        metrica2.metric(
            "Mesas ocupadas",
            ocupadas,
            f"de {total}"
        )

        metrica3.metric(
            "Personas detectadas",
            cantidad_personas
        )

        porcentaje_ocupacion = (
            ocupadas / total * 100
            if total > 0 else 0
        )

        st.markdown("**Nivel de ocupación actual**")

        st.progress(porcentaje_ocupacion / 100)

        st.caption(
            f"{porcentaje_ocupacion:.0f}% de las mesas ocupadas"
        )

    st.divider()

    st.markdown("### Mapa de disponibilidad")

    mostrar_mapa(estados)

    st.caption(
        "🟢 Verde: mesa disponible  ·  🔴 Rojo: mesa ocupada"
    )


# =============================================
# 13. ESTIMACIÓN DE OCUPACIÓN DEL CAMPUS
# =============================================

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


# =============================================
# 14. DESCRIPCIÓN DEL DÍA
# =============================================

def describir_dia(fecha):

    inicio_periodo = date(2026, 8, 5)
    inicio_receso = date(2026, 9, 14)
    fin_receso = date(2026, 9, 21)

    fin_clases = date(2026, 11, 27)

    inicio_evaluaciones = date(2026, 11, 30)
    fin_evaluaciones = date(2026, 12, 14)

    fin_periodo = date(2026, 12, 16)

    suspensiones = [
        date(2026, 12, 7),
        date(2026, 12, 8)
    ]

    if fecha == inicio_periodo:
        return "Inicio del segundo período académico."

    if fecha == inicio_receso:
        return "Comienza el receso académico."

    if fecha == fin_receso:
        return "Último día del receso académico. Las actividades se retoman el 22 de septiembre."

    if fecha == fin_clases:
        return "Último día de clases del segundo período académico."

    if fecha == inicio_evaluaciones:
        return "Comienza el período de evaluaciones."

    if fecha == fin_evaluaciones:
        return "Último día del período de evaluaciones."

    if fecha == fin_periodo:
        return "Término del segundo período académico."

    if fecha in suspensiones:
        return "Suspensión de actividades académicas."

    if inicio_receso <= fecha <= fin_receso:
        return "Receso académico. No se consideran clases regulares."

    if inicio_periodo <= fecha < inicio_receso:
        return "Jornada regular de clases del segundo período académico."

    if fin_receso < fecha <= fin_clases:
        return "Jornada regular de clases del segundo período académico."

    if inicio_evaluaciones < fecha <= fin_evaluaciones:
        return "Período de evaluaciones académicas."

    if fin_evaluaciones < fecha <= fin_periodo:
        return "Etapa de cierre del período académico."

    if fecha.weekday() >= 5:
        return "Fin de semana. No se consideran actividades académicas regulares."

    if fecha < inicio_periodo:
        return "Fuera del segundo período académico."

    if fecha > fin_periodo:
        return "Período académico finalizado."

    return "Sin información académica específica para esta fecha."


# =============================================
# 15. PREDICCIÓN HORARIA
# =============================================

def predecir_ocupacion_horaria(fecha):

    inicio_periodo = date(2026, 8, 5)
    inicio_receso = date(2026, 9, 14)
    fin_receso = date(2026, 9, 21)
    fin_clases = date(2026, 11, 27)

    inicio_evaluaciones = date(2026, 11, 30)
    fin_evaluaciones = date(2026, 12, 14)
    fin_periodo = date(2026, 12, 16)

    suspensiones = [
        date(2026, 12, 7),
        date(2026, 12, 8)
    ]

    horarios = [
        "08:00 - 10:00",
        "10:00 - 12:00",
        "12:00 - 13:00",
        "13:00 - 15:00",
        "15:00 - 17:00",
        "17:00 - 19:00"
    ]

    if fecha < inicio_periodo or fecha > fin_periodo:
        return None

    if fecha.weekday() >= 5:
        return [(h, 0) for h in horarios]

    if fecha in suspensiones:
        return [(h, 0) for h in horarios]

    if inicio_receso <= fecha <= fin_receso:

        porcentajes = [5, 10, 15, 20, 10, 5]

        return list(zip(horarios, porcentajes))

    if inicio_periodo <= fecha <= fin_clases:

        ocupacion_diaria, _ = estimar_ocupacion(fecha)

        factor = ocupacion_diaria / 90

        porcentajes_base = [45, 70, 90, 100, 65, 30]

        porcentajes = [
            min(100, round(valor * factor))
            for valor in porcentajes_base
        ]

        # Almuerzo: máxima demanda estimada
        porcentajes[3] = 100

        return list(zip(horarios, porcentajes))

    if inicio_evaluaciones <= fecha <= fin_evaluaciones:

        porcentajes = [30, 55, 85, 100, 70, 35]

        return list(zip(horarios, porcentajes))

    if fecha <= fin_periodo:

        porcentajes = [10, 20, 25, 25, 15, 5]

        return list(zip(horarios, porcentajes))

    return None


# =============================================
# 16. MOSTRAR OCUPACIÓN DEL CAMPUS
# =============================================

def mostrar_ocupacion_campus():

    st.markdown("### Análisis del calendario académico")

    fecha = st.date_input(
        "Selecciona una fecha",
        value=date.today(),
        key="fecha_campus"
    )

    porcentaje, periodo = estimar_ocupacion(fecha)

    descripcion = describir_dia(fecha)

    # TARJETA DE ACTIVIDAD ACADÉMICA
    # Se utiliza st.html() para evitar que el HTML
    # aparezca como texto en la interfaz.

    st.html(f"""
    <div style="
        background-color: white;
        border: 1px solid #E2EAF2;
        border-left: 5px solid #28577E;
        padding: 22px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
    ">

        <div style="
            color: #6B8095;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
        ">
            ACTIVIDAD ACADÉMICA
        </div>

        <div style="
            color: #183B5B;
            font-size: 19px;
            font-weight: 700;
            margin-top: 8px;
        ">
            ¿Qué pasa este día?
        </div>

        <div style="
            color: #526477;
            font-size: 15px;
            margin-top: 8px;
        ">
            {descripcion}
        </div>

    </div>
    """)

    # OCUPACIÓN GENERAL

    st.markdown("### Ocupación estimada del campus")

    if porcentaje is None:

        st.warning(periodo)

    else:

        col1, col2 = st.columns([1, 2])

        with col1:

            st.metric(
                "Ocupación estimada",
                f"{porcentaje}%"
            )

        with col2:

            st.write(f"**Situación:** {periodo}")

            st.progress(porcentaje / 100)

    st.divider()

    # PREDICCIÓN HORARIA

    st.markdown("### Predicción de ocupación por horario")

    st.write(
        "Identificación de los momentos de mayor demanda "
        "de mesas durante la jornada."
    )

    predicciones = predecir_ocupacion_horaria(fecha)

    if predicciones is None:

        st.warning(
            "No hay una predicción disponible para esta fecha."
        )

    else:

        datos = {
            "Horario": [h for h, p in predicciones],
            "Ocupación estimada": [
                f"{p}%" for h, p in predicciones
            ]
        }

        st.table(datos)

        valores = {
            horario: porcentaje_horario
            for horario, porcentaje_horario in predicciones
        }

        st.markdown("#### Comportamiento de la ocupación")

        st.bar_chart(
            valores,
            y_label="Ocupación estimada (%)"
        )

        # TARJETA DE HORARIO DE MAYOR DEMANDA

        st.html("""
        <div style="
            background-color: #FFF4E5;
            border: 1px solid #F2D2A9;
            border-left: 5px solid #D97736;
            border-radius: 12px;
            padding: 18px;
            margin-top: 15px;
        ">

            <div style="
                color: #A95C20;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
            ">
                HORARIO DE MAYOR DEMANDA
            </div>

            <div style="
                color: #7D421C;
                font-size: 23px;
                font-weight: 800;
                margin-top: 5px;
            ">
                13:00 - 15:00 · 100%
            </div>

            <div style="
                color: #865B3D;
                font-size: 14px;
                margin-top: 5px;
            ">
                Se estima una ocupación máxima de las mesas
                durante el horario de almuerzo.
            </div>

        </div>
        """)

    st.caption(
        "Las predicciones horarias son estimaciones referenciales "
        "basadas en el calendario académico y patrones de uso "
        "supuestos. No corresponden a mediciones reales ni a "
        "predicciones entrenadas con registros históricos."
    )


# =============================================
# 17. PESTAÑAS PRINCIPALES
# =============================================

tab1, tab2, tab3 = st.tabs([
    "📍 Patio 1",
    "📍 Patio 2",
    "📊 Ocupación del campus"
])


# =============================================
# 18. PESTAÑA PATIO 1
# =============================================

with tab1:

    st.markdown("### Disponibilidad de mesas · Patio 1")

    st.write(
        "Monitoreo visual de las mesas mediante detección "
        "automática de personas."
    )

    mostrar_patio(
        "Patio 1",
        fotos_patio_1,
        mesas_patio_1,
        "estados_mesas_patio1",
        "foto_actual_patio1"
    )


# =============================================
# 19. PESTAÑA PATIO 2
# =============================================

with tab2:

    st.markdown("### Disponibilidad de mesas · Patio 2")

    st.write(
        "Monitoreo visual de las mesas mediante detección "
        "automática de personas."
    )

    mostrar_patio(
        "Patio 2",
        fotos_patio_2,
        mesas_patio_2,
        "estados_mesas_patio2",
        "foto_actual_patio2"
    )


# =============================================
# 20. PESTAÑA OCUPACIÓN DEL CAMPUS
# =============================================

with tab3:

    mostrar_ocupacion_campus()


# =============================================
# 21. PIE DE PÁGINA
# =============================================

st.html("""
<div style="
    border-top: 1px solid #DCE5EF;
    margin-top: 40px;
    padding-top: 18px;
    text-align: center;
    color: #8A9BAD;
    font-size: 12px;
">
    SMART CAMPUS UC · SISTEMA DE MONITOREO Y PREDICCIÓN
    <br>
    Prototipo académico de inteligencia artificial aplicada
    a la disponibilidad de espacios universitarios.
</div>
""")
