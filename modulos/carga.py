"""Fase 1 · Carga de datos: archivos locales (CSV/Excel), URL y datasets de ejemplo."""
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes, load_iris, load_wine

from .utils import mostrar_df

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- lectura
def _leer_csv(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV detectando el separador y probando codificaciones comunes."""
    for codificacion in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise ValueError("No se pudo decodificar el archivo CSV.")


def leer_archivo(nombre: str, contenido: bytes) -> dict:
    """Devuelve {nombre_tabla: DataFrame}. Un Excel con varias hojas produce varias tablas."""
    extension = nombre.lower().rsplit(".", 1)[-1]
    if extension in ("xlsx", "xls"):
        hojas = pd.read_excel(io.BytesIO(contenido), sheet_name=None)
        if len(hojas) == 1:
            return {nombre: next(iter(hojas.values()))}
        return {f"{nombre} › {hoja}": df for hoja, df in hojas.items()}
    return {nombre: _leer_csv(contenido)}


def _url_directa(url: str) -> str:
    """Convierte enlaces compartidos de Google Drive / Google Sheets / GitHub en descargas directas."""
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r"docs\.google\.com/spreadsheets/d/([^/]+)", url)
    if m:
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def leer_url(url: str) -> dict:
    """Descarga un CSV o Excel desde una URL pública."""
    import urllib.request

    directa = _url_directa(url.strip())
    peticion = urllib.request.Request(directa, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(peticion, timeout=30) as respuesta:
        contenido = respuesta.read()
    nombre = directa.split("?")[0].rstrip("/").split("/")[-1] or "datos_url.csv"
    if not nombre.lower().endswith((".csv", ".xlsx", ".xls")):
        # Si la URL no indica la extensión, se detecta por la firma del archivo (ZIP = xlsx)
        nombre += ".xlsx" if contenido[:2] == b"PK" else ".csv"
    return leer_archivo(nombre, contenido)


def datasets_ejemplo() -> dict:
    """Catálogo de datasets de prueba (incluye los dos del taller)."""
    def iris():
        d = load_iris(as_frame=True)
        df = d.frame.rename(columns={"target": "especie"})
        df["especie"] = df["especie"].map(dict(enumerate(d.target_names)))
        return df

    def vino():
        d = load_wine(as_frame=True)
        df = d.frame.rename(columns={"target": "clase_vino"})
        df["clase_vino"] = "clase_" + df["clase_vino"].astype(str)
        return df

    def diabetes():
        return load_diabetes(as_frame=True).frame.rename(columns={"target": "progresion"})

    catalogo = {
        "Clima – ¿lloverá? (Clasificación, taller)": (lambda: pd.read_csv(CARPETA_DATOS / "weather_forecast_data.csv"), "CLASIFICACIÓN",
                                                       "2.500 días con temperatura, humedad, viento, nubosidad y presión. Clases desbalanceadas."),
        "Gimnasio – calorías (Regresión, taller)": (lambda: pd.read_csv(CARPETA_DATOS / "gym_members_exercise.csv"), "REGRESIÓN",
                                                     "1.800 miembros con nulos y textos sucios. Ideal para practicar limpieza."),
        "Iris Flowers (3 clases)": (iris, "CLASIFICACIÓN", "150 flores con 4 medidas morfológicas."),
        "Vinos (3 clases)": (vino, "CLASIFICACIÓN", "178 vinos con 13 propiedades químicas."),
        "Diabetes (progresión)": (diabetes, "REGRESIÓN", "442 pacientes, 10 variables clínicas normalizadas."),
    }
    return catalogo


OBJETIVOS_EJEMPLO = {"Gimnasio – calorías (Regresión, taller)": "Calories_Burned"}


def _registrar(tablas: dict):
    """Agrega tablas al estado y selecciona la última como activa."""
    for nombre, df in tablas.items():
        df.columns = [str(c).strip() for c in df.columns]
        st.session_state["datasets"][nombre] = df
        st.session_state["activo"] = nombre
    # Al cambiar de datos se invalidan la configuración y los modelos previos
    for clave in ("config", "resultados"):
        st.session_state.pop(clave, None)


# ---------------------------------------------------------------- interfaz
def render():
    st.subheader("1. Carga de Datos")
    izquierda, derecha = st.columns([2, 1], gap="large")

    with izquierda:
        local, remoto = st.tabs(["📁 Archivos locales", "🌐 Cargar desde URL"])
        with local:
            archivos = st.file_uploader(
                "Selecciona o arrastra uno o varios archivos (.csv, .xlsx, .xls)",
                type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="uploader")
            if archivos:
                nuevos = [a for a in archivos if a.file_id not in st.session_state["archivos_leidos"]]
                for archivo in nuevos:
                    try:
                        _registrar(leer_archivo(archivo.name, archivo.getvalue()))
                        st.session_state["archivos_leidos"].add(archivo.file_id)
                        st.toast(f"✅ {archivo.name} cargado")
                    except Exception as error:  # archivo malformado, hoja vacía, etc.
                        st.error(f"No se pudo leer **{archivo.name}**: {error}")
        with remoto:
            url = st.text_input("URL pública de un CSV o Excel",
                                placeholder="https://.../datos.csv  ·  enlaces de Google Drive, Google Sheets o GitHub")
            if st.button("Descargar datos", disabled=not url):
                try:
                    with st.spinner("Descargando..."):
                        _registrar(leer_url(url))
                    st.success("Datos descargados correctamente.")
                except Exception as error:
                    st.error(f"No se pudo descargar o leer la URL: {error}")

    with derecha:
        st.markdown("##### 🗂️ Datasets de ejemplo")
        for nombre, (cargador, tipo, descripcion) in datasets_ejemplo().items():
            with st.container(border=True):
                st.markdown(f"**{nombre}**  \n`{tipo}`  \n<small>{descripcion}</small>", unsafe_allow_html=True)
                if st.button("Cargar", key=f"ej_{nombre}"):
                    _registrar({nombre: cargador()})
                    if nombre in OBJETIVOS_EJEMPLO:
                        st.session_state.setdefault("objetivo_sugerido", {})[nombre] = OBJETIVOS_EJEMPLO[nombre]
                    st.rerun()

    datasets = st.session_state["datasets"]
    if not datasets:
        return

    st.divider()
    nombres = list(datasets)
    activo = st.selectbox("Dataset a analizar", nombres, index=nombres.index(st.session_state["activo"]))
    if activo != st.session_state["activo"]:
        st.session_state["activo"] = activo
        for clave in ("config", "resultados"):
            st.session_state.pop(clave, None)
        st.rerun()

    df = datasets[activo]
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{len(df):,}".replace(",", "."))
    c2.metric("Columnas", df.shape[1])
    c3.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024:,.0f} KB")
    filas = st.slider("Filas a previsualizar", 5, 100, 10)
    mostrar_df(df.head(filas))
