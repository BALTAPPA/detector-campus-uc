
import streamlit as st
import random
from pathlib import Path
from PIL import Image, ImageDraw
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
# 2. BUSCAR FOTOGRAFIAS
# =====================================

def buscar_fotografias(numeros):

    fotografias = []

    for numero in numeros:

        nombre_base = f"foto_{numero:02d}"

        for archivo in CARPETA.iterdir():

            if (
                archivo.stem.lower() == nombre_base
                and archivo.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ):
                fotografias.append(archivo)
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

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

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
# 5. COORDENADAS DEL PATIO 1
# =====================================

mesas_patio_1 = {
    "MESA 1": [100, 200, 240, 330],
    "MESA 2": [430, 220, 540, 320],
    "MESA 3": [570, 250, 680, 350],
    "MESA 4": [760, 380, 880, 480],
    "MESA 5": [240, 380, 500, 520]
}


# =====================================
# 6. COLORES DE LAS MESAS
# =====================================

COLORES_MESAS = [
    (230, 50, 50),
    (30, 120, 230),
    (20, 160, 80),
    (230, 150, 20),
    (150, 70, 210),
    (20, 170, 170),
    (220, 70, 150),
    (120, 120, 30),
    (80, 80, 220),
    (230, 100, 40)
]


# =====================================
# 7. CONFIGURAR LAS 10 MESAS DEL PATIO 2
# =====================================

def configurar_mesas_patio_2(foto):

    st.subheader("⚙️ Configuración de mesas - Patio 2")

    imagen = Image.open(foto).convert("RGB")

    ancho, alto = imagen.size

    st.write(
        f"Dimensiones de la fotografía: {ancho} x {alto} píxeles."
    )

    st.write(
        "Ajusta las coordenadas de cada mesa. "
        "Los rectángulos de colores se actualizarán sobre la fotografía."
    )

    # ---------------------------------
    # INICIALIZAR COORDENADAS
    # ---------------------------------

    if "coordenadas_patio_2" not in st.session_state:

        st.session_state.coordenadas_patio_2 = {}

        ancho_caja = max(30, int(ancho * 0.12))
        alto_caja = max(30, int(alto * 0.12))

        for i in range(1, 11):

            columna = (i - 1) % 5
            fila = (i - 1) // 5

            x1 = int((columna + 0.5) * ancho / 5 - ancho_caja / 2)
            y1 = int((fila + 0.5) * alto / 2 - alto_caja / 2)

            x1 = max(0, min(x1, ancho - ancho_caja))
            y1 = max(0, min(y1, alto - alto_caja))

            st.session_state.coordenadas_patio_2[f"MESA {i}"] = [
                x1,
                y1,
                x1 + ancho_caja,
                y1 + alto_caja
            ]

    # ---------------------------------
    # AJUSTAR COORDENADAS CON SLIDERS
    # ---------------------------------

    mesas_actualizadas = {}

    for i in range(1, 11):

        nombre = f"MESA {i}"

        coordenadas = st.session_state.coordenadas_patio_2[nombre]

        x1, y1, x2, y2 = coordenadas

        # Evitar valores fuera de la fotografía

        x1 = max(0, min(x1, ancho))
        x2 = max(0, min(x2, ancho))

        y1 = max(0, min(y1, alto))
        y2 = max(0, min(y2, alto))

        # Inicializar valores de los controles

        valores_iniciales = {
            f"x1_p2_{i}": x1,
            f"y1_p2_{i}": y1,
            f"x2_p2_{i}": x2,
            f"y2_p2_{i}": y2
        }

        for clave, valor in valores_iniciales.items():

            if clave not in st.session_state:
                st.session_state[clave] = valor

            else:
                st.session_state[clave] = max(
                    0,
                    min(st.session_state[clave],
                        ancho if clave.startswith("x") else alto)
                )

        # ---------------------------------
        # CONTROLES DE CADA MESA
        # ---------------------------------

        color = COLORES_MESAS[i - 1]

        color_hex = "#{:02X}{:02X}{:02X}".format(*color)

        with st.expander(f"🟦 MESA {i}", expanded=False):

            st.markdown(
                f"**Mesa {i}** — Color de referencia: "
                f"<span style='color:{color_hex};'>●</span>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:

                nuevo_x1 = st.slider(
                    "X inicial",
                    0,
                    ancho,
                    key=f"x1_p2_{i}"
                )

                nuevo_y1 = st.slider(
                    "Y inicial",
                    0,
                    alto,
                    key=f"y1_p2_{i}"
                )

            with col2:

                nuevo_x2 = st.slider(
                    "X final",
                    0,
                    ancho,
                    key=f"x2_p2_{i}"
                )

                nuevo_y2 = st.slider(
                    "Y final",
                    0,
                    alto,
                    key=f"y2_p2_{i}"
                )

            st.caption(
                f"Coordenadas: ({nuevo_x1}, {nuevo_y1}) "
                f"hasta ({nuevo_x2}, {nuevo_y2})"
            )

        mesas_actualizadas[nombre] = [
            nuevo_x1,
            nuevo_y1,
            nuevo_x2,
            nuevo_y2
        ]

    # Guardar las coordenadas

    st.session_state.coordenadas_patio_2 = mesas_actualizadas

    # =================================
    # 8. DIBUJAR RECTANGULOS SOBRE FOTO
    # =================================

    imagen_previa = imagen.copy()

    dibujo = ImageDraw.Draw(imagen_previa)

    grosor = max(3, int(min(ancho, alto) * 0.005))

    for i in range(1, 11):

        nombre = f"MESA {i}"

        x1, y1, x2, y2 = mesas_actualizadas[nombre]

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        color = COLORES_MESAS[i - 1]

        # Rectangulo de la mesa

        dibujo.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=grosor
        )

        # Etiqueta de la mesa

        etiqueta = f"MESA {i}"

        dibujo.text(
            (x1 + 4, y1 + 4),
            etiqueta,
            fill=color,
            stroke_width=2,
            stroke_fill=(255, 255, 255)
        )

    # ---------------------------------
    # MOSTRAR FOTO CON RECTANGULOS
    # ---------------------------------

    st.divider()

    st.subheader("Vista previa de calibración")

    st.image(
        imagen_previa,
        caption="Rectángulos de colores: ubicación de las 10 mesas",
        use_container_width=True
    )

    st.caption(
        "Cada color representa una mesa. "
        "Los rectángulos se actualizan al mover los controles."
    )

    return mesas_actualizadas


# =====================================
# 9. MOSTRAR CADA PATIO
# =====================================

def mostrar_patio(
    nombre_patio,
    fotografias,
    mesas,
    clave,
    configurar=False
):

    st.subheader(f"Disponibilidad - {nombre_patio}")

    if len(fotografias) == 0:

        st.error(
            f"No se encontraron fotografías para {nombre_patio}."
        )

        return

    # ---------------------------------
    # CLAVES INDEPENDIENTES
    # ---------------------------------

    clave_foto = f"foto_actual_{clave}"
    clave_procesada = f"foto_procesada_{clave}"
    clave_estados = f"estados_mesas_{clave}"
    clave_personas = f"cantidad_personas_{clave}"
    clave_configuracion = f"configuracion_procesada_{clave}"

    # ---------------------------------
    # SELECCIONAR FOTOGRAFIA
    # ---------------------------------

    if (
        clave_foto not in st.session_state
        or st.session_state[clave_foto] not in fotografias
    ):

        st.session_state[clave_foto] = random.choice(fotografias)

    col1, col2 = st.columns([1, 3])

    with col1:

        if st.button(
            "🔄 Cambiar fotografía",
            key=f"boton_{clave}"
        ):

            opciones = [
                foto for foto in fotografias
                if foto != st.session_state[clave_foto]
            ]

            if opciones:
                st.session_state[clave_foto] = random.choice(opciones)

    with col1:

        st.image(
            str(st.session_state[clave_foto]),
            width=260
        )

        st.caption(st.session_state[clave_foto].name)

    # ---------------------------------
    # CONFIGURACION VISUAL PATIO 2
    # ---------------------------------

    if configurar:

        mesas = configurar_mesas_patio_2(
            st.session_state[clave_foto]
        )

        st.divider()

    # ---------------------------------
    # IDENTIFICAR CAMBIOS
    # ---------------------------------

    firma_foto = str(st.session_state[clave_foto])

    firma_coordenadas = tuple(
        (nombre, tuple(coordenadas))
        for nombre, coordenadas in mesas.items()
    )

    foto_cambio = (
        clave_procesada not in st.session_state
        or st.session_state[clave_procesada] != firma_foto
    )

    coordenadas_cambiadas = (
        clave_configuracion not in st.session_state
        or st.session_state[clave_configuracion] != firma_coordenadas
    )

    # ---------------------------------
    # BOTON DE ANALISIS
    # ---------------------------------

    analizar_ahora = False

    if configurar:

        analizar_ahora = st.button(
            "🔍 Analizar disponibilidad con estas coordenadas",
            key=f"analizar_{clave}",
            type="primary"
        )

    # Analisis inicial o cuando cambia la fotografia

    if (
        clave_estados not in st.session_state
        or foto_cambio
    ):

        analizar_ahora = True

    # ---------------------------------
    # EJECUTAR YOLO
    # ---------------------------------

    if analizar_ahora:

        modelo = cargar_modelo()

        with st.spinner(f"Analizando {nombre_patio}..."):

            estados, cantidad_personas = analizar_fotografia(
                st.session_state[clave_foto],
                modelo,
                mesas
            )

            st.session_state[clave_estados] = estados
            st.session_state[clave_personas] = cantidad_personas

            st.session_state[clave_procesada] = firma_foto

            st.session_state[clave_configuracion] = firma_coordenadas

    elif configurar and coordenadas_cambiadas:

        st.info(
            "Has modificado las coordenadas. "
            "Cuando termines de ajustar los rectángulos, "
            "presiona el botón de análisis."
        )

    # ---------------------------------
    # MOSTRAR RESULTADOS
    # ---------------------------------

    if clave_estados not in st.session_state:
        return

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

    st.subheader("Resultados del análisis")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total de mesas", len(mesas))
    c2.metric("Mesas disponibles", disponibles)
    c3.metric("Mesas ocupadas", ocupadas)

    st.caption(
        f"Personas detectadas: {cantidad_personas}"
    )

    if configurar and coordenadas_cambiadas and not analizar_ahora:

        st.warning(
            "El mapa todavía muestra el último análisis realizado. "
            "Pulsa el botón de análisis para actualizarlo."
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

    for inicio in range(0, len(nombres_mesas), 3):

        columnas = st.columns(3)

        for i, nombre in enumerate(
            nombres_mesas[inicio:inicio + 3]
        ):

            with columnas[i]:
                mostrar_mesa(nombre)


# =====================================
# 10. PESTAÑAS
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
        {
            f"MESA {i}": [100, 100, 200, 200]
            for i in range(1, 11)
        },
        "patio2",
        configurar=True
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
