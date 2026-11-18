
import streamlit as st
import numpy as np
from PIL import Image
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

    st.success("Fotografia cargada correctamente.")

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

    st.write(f"Personas detectadas: **{len(personas)}**")

    # =================================
    # 6. CALIBRAR LAS 5 MESAS
    # =================================

    st.divider()
    st.subheader("Configuracion de las mesas")

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
        # 8. CALCULAR ESTADISTICAS
        # =================================

        ocupadas = sum(
            estado == "OCUPADA"
            for estado in estados.values()
        )

        disponibles = 5 - ocupadas

        # =================================
        # 9. MOSTRAR MAPA ESQUEMATICO
        # =================================

        st.divider()
        st.subheader("Mapa de disponibilidad")

        st.markdown("""
        <style>
        .mapa {
            background-color: #f1f2f3;
            border: 1px solid #d1d5db;
            border-radius: 15px;
            padding: 35px;
            min-height: 350px;
        }

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

        # Posiciones esquematicas del mapa
        posiciones = [
            (0, 0),  # Mesa 1
            (0, 1),  # Mesa 2
            (0, 2),  # Mesa 3
            (1, 0),  # Mesa 4
            (1, 1),  # Mesa 5
        ]

        # Distribucion en dos filas
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
                        st.markdown("<div style='height:131px'></div>",
                                    unsafe_allow_html=True)

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
        # 10. ESTADISTICAS
        # =================================

        st.divider()
        st.subheader("Estado general")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total de mesas", 5)
        col2.metric("Disponibles", disponibles)
        col3.metric("Ocupadas", ocupadas)

        st.info(
            "El sistema estima la ocupacion mediante "
            "la interseccion entre personas detectadas "
            "y las zonas definidas para cada mesa."
        )

else:

    st.info("Sube una fotografia para comenzar.")
