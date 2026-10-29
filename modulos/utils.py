"""Funciones de apoyo compartidas por todos los módulos de la aplicación."""
import re

import numpy as np
import pandas as pd
import streamlit as st

# Paleta usada en todos los gráficos
COLORES = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#0EA5E9", "#A855F7", "#64748B", "#EC4899"]

# Patrón para limpiar secuencias de escape escritas como texto literal (ej. "\\t182", "\\nYoga")
_PATRON_ESCAPES = re.compile(r"(?:\\[tnr])+")


def mostrar_df(df, **kwargs):
    """Muestra un DataFrame ocupando todo el ancho (compatible con varias versiones de Streamlit)."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def mostrar_fig(fig, **kwargs):
    """Muestra una figura Plotly ocupando todo el ancho."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), colorway=COLORES)
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def df_activo():
    """Devuelve el DataFrame seleccionado actualmente (o None si no hay datos)."""
    nombre = st.session_state.get("activo")
    if not nombre:
        return None
    return st.session_state["datasets"].get(nombre)


def requiere_datos():
    """Muestra un aviso y devuelve False si aún no se han cargado datos."""
    if df_activo() is None:
        st.info("Primero carga un dataset en la pestaña **1. Carga de Datos**.")
        return False
    return True


def columnas_por_tipo(df):
    """Separa las columnas en numéricas y categóricas."""
    numericas = df.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in df.columns if c not in numericas]
    return numericas, categoricas


def limpiar_texto(serie):
    """Quita espacios y caracteres de escape (\\t, \\n) tanto reales como escritos de forma literal."""
    if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
        return serie
    limpia = serie.astype("string").str.replace(_PATRON_ESCAPES, "", regex=True).str.strip()
    limpia = limpia.replace("", pd.NA)
    return limpia.astype(object).where(limpia.notna(), np.nan)


def intentar_numerico(serie, umbral=0.9):
    """Convierte una columna de texto a número si al menos el `umbral` de sus valores son numéricos."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie, False
    no_nulos = serie.dropna()
    if no_nulos.empty:
        return serie, False
    convertida = pd.to_numeric(limpiar_texto(serie), errors="coerce")
    proporcion = convertida.notna().sum() / len(no_nulos)
    if proporcion >= umbral:
        return convertida, True
    return serie, False


def cadenas_vacias(serie):
    """Cuenta celdas de texto vacías o formadas solo por espacios."""
    if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
        return 0
    s = serie.dropna().astype(str)
    return int((s.str.strip() == "").sum())


def texto_sucio(serie):
    """Cuenta valores con espacios sobrantes o secuencias de escape (ej. ' Cardio', '\\tYoga')."""
    if not (pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)):
        return 0
    s = serie.dropna().astype(str)
    s = s[s.str.strip() != ""]
    return int(((s != s.str.strip()) | s.str.contains(_PATRON_ESCAPES)).sum())


def tarjeta(titulo, texto, tipo="info"):
    """Caja de explicación pedagógica."""
    {"info": st.info, "ok": st.success, "warn": st.warning, "error": st.error}[tipo](f"**{titulo}** — {texto}")
