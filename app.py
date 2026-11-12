
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

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
    "Sistema de inteligencia artificial para detectar "
    "la ocupacion de mesas mediante fotografias del campus."
)

# =====================================
# 2. CARGAR MODELO YOLOv8
# =====================================

@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")

with st.spinner("Cargando inteligencia artificial..."):
    modelo = cargar_modelo()

# =====================================
# 3. SUBIR FOTOGRAFIA
# =====================================

archivo = st.file_uploader(
    "Sube una fotografia del campus",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# 4. CALCULAR INTERSECCION
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
# 5. PROCESAR FOTOGRAFIA
# =====================================

if archivo is not None:

    imagen = Image.open(archivo).convert("RGB")

    ancho, alto = imagen.size

    st.subheader("Fotografia original")
    st.image(imagen, use_container_width=True)

    # Detectar personas
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

    st.success(
        f"Personas detectadas: {len(personas)}"
    )

    # =================================
    # 6. CALIBRAR LAS 5 MESAS
    # =================================

    st.divider()
    st.subheader("Calibracion de las 5 mesas")

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

        # Adaptar coordenadas al tamaño de la imagen
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

            with st.expander(nombre, expanded=(i == 0)):

                x1, x2 = st.slider(
                    f"Coordenadas X - {nombre}",
                    0, ancho,
                    (x_min, x_max),
                    key=f"x_{i}"
                )

                y1, y2 = st.slider(
                    f"Coordenadas Y - {nombre}",
                    0, alto,
                    (y_min, y_max),
                    key=f"y_{i}"
                )

                mesas[nombre] = [x1, y1, x2, y2]

    # =================================
    # 7. ANALIZAR DISPONIBILIDAD
    # =================================

    if st.button("ANALIZAR DISPONIBILIDAD", type="primary"):

        estados = {}

        for nombre, coordenadas in mesas.items():

            ocupada = False

            for persona in personas:

                if calcular_interseccion(coordenadas, persona):
                    ocupada = True
                    break

            estados[nombre] = (
                "OCUPADA" if ocupada else "DISPONIBLE"
            )

        # =================================
        # 8. DIBUJAR RESULTADOS
        # =================================

        capa = Image.new(
            "RGBA",
            imagen.size,
            (0, 0, 0, 0)
        )

        dibujo = ImageDraw.Draw(capa)

        # Dibujar mesas
        for nombre, coordenadas in mesas.items():

            x1, y1, x2, y2 = coordenadas

            if estados[nombre] == "OCUPADA":
                color = (255, 0, 0, 85)
                borde = "red"
            else:
                color = (0, 255, 0, 85)
                borde = "lime"

            dibujo.rectangle(
                [x1, y1, x2, y2],
                fill=color,
                outline=borde,
                width=4
            )

            dibujo.text(
                (x1 + 5, y1 + 5),
                f"{nombre}: {estados[nombre]}",
                fill="white",
                stroke_width=2,
                stroke_fill="black"
            )

        # Dibujar personas detectadas
        for persona in personas:

            px1, py1, px2, py2 = map(int, persona)

            dibujo.rectangle(
                [px1, py1, px2, py2],
                outline="yellow",
                width=3
            )

        imagen_final = Image.alpha_composite(
            imagen.convert("RGBA"),
            capa
        ).convert("RGB")

        # =================================
        # 9. MOSTRAR RESULTADOS
        # =================================

        st.divider()
        st.subheader("Resultado del analisis")

        st.image(
            imagen_final,
            caption="Deteccion de ocupacion",
            use_container_width=True
        )

        ocupadas = sum(
            estado == "OCUPADA"
            for estado in estados.values()
        )

        disponibles = 5 - ocupadas

        col1, col2, col3 = st.columns(3)

        col1.metric("Total de mesas", 5)
        col2.metric("Disponibles", disponibles)
        col3.metric("Ocupadas", ocupadas)

        st.subheader("Reporte de ocupacion")

        for mesa, estado in estados.items():

            if estado == "OCUPADA":
                st.error(f"🔴 {mesa}: OCUPADA")
            else:
                st.success(f"🟢 {mesa}: DISPONIBLE")

        st.info(
            "El sistema estima la ocupacion mediante "
            "la interseccion entre personas detectadas "
            "y las zonas definidas para cada mesa."
        )

else:

    st.info("Sube una fotografia para comenzar.")
