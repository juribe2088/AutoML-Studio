"""Fase 2 · Preparación y validación: nulos, duplicados, texto sucio, tipos y tamaño de muestra."""
import math

import pandas as pd
import plotly.express as px
import streamlit as st

from .utils import (cadenas_vacias, df_activo, intentar_numerico, mostrar_df, mostrar_fig,
                    requiere_datos, texto_sucio)


def perfil_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de diagnóstico por columna."""
    filas = []
    for col in df.columns:
        s = df[col]
        _, es_num_como_texto = intentar_numerico(s) if not pd.api.types.is_numeric_dtype(s) else (s, False)
        filas.append({
            "Columna": col,
            "Tipo": str(s.dtype),
            "Nulos": int(s.isna().sum()),
            "% Nulos": round(100 * s.isna().mean(), 2),
            "Valores únicos": int(s.nunique(dropna=True)),
            "Cadenas vacías": cadenas_vacias(s),
            "Texto con espacios/escapes": texto_sucio(s),
            "Número guardado como texto": "⚠️ Sí" if es_num_como_texto else "",
            "Ejemplo": str(s.dropna().iloc[0]) if s.notna().any() else "",
        })
    return pd.DataFrame(filas)


def validez_estadistica(n_filas: int, n_columnas: int) -> dict:
    """Evalúa si el tamaño del dataset es suficiente para un análisis básico / modelado.

    Criterios usados (reglas prácticas ampliamente aceptadas):
    * n ≥ 30  → mínimo para que apliquen aproximaciones estadísticas (Teorema del Límite Central).
    * n ≥ 10 × nº de variables predictoras → regla de "10 observaciones por variable".
    * Margen de error al 95 % para una proporción: 1,96·√(0,25/n). Con n = 385 el margen es ±5 %.
    """
    predictoras = max(n_columnas - 1, 1)
    margen = 1.96 * math.sqrt(0.25 / n_filas) * 100 if n_filas else 100
    minimo_reglas = max(30, 10 * predictoras)
    if n_filas < 30:
        nivel = "error"
    elif n_filas < minimo_reglas or n_filas < 100:
        nivel = "warn"
    else:
        nivel = "ok"
    return {"nivel": nivel, "margen": margen, "minimo": minimo_reglas, "predictoras": predictoras,
            "filas_por_variable": n_filas / predictoras}


def render():
    st.subheader("2. Preparación y Validación Estadística")
    if not requiere_datos():
        return
    df = df_activo()
    perfil = perfil_columnas(df)

    # ---- tarjetas resumen
    duplicados = int(df.duplicated().sum())
    nulos = int(df.isna().sum().sum())
    filas_con_nulos = int(df.isna().any(axis=1).sum())
    sucios = int(perfil["Texto con espacios/escapes"].sum())
    vacias = int(perfil["Cadenas vacías"].sum())
    c = st.columns(5)
    c[0].metric("Filas", f"{len(df):,}".replace(",", "."))
    c[1].metric("Celdas nulas", nulos, help=f"{filas_con_nulos} filas tienen al menos un valor vacío")
    st.caption(f"Filas con al menos un nulo: **{filas_con_nulos}** ({100 * filas_con_nulos / max(len(df), 1):.1f} %)")
    c[2].metric("Filas duplicadas", duplicados, delta_color="off")
    c[3].metric("Cadenas vacías", vacias)
    c[4].metric("Textos sucios", sucios, help="Valores con espacios al inicio/final o secuencias \\t, \\n")

    # ---- validez estadística
    st.markdown("#### 📏 Validación del tamaño de la muestra")
    v = validez_estadistica(len(df), df.shape[1])
    texto = (f"{len(df):,} filas para {v['predictoras']} posibles variables predictoras "
             f"({v['filas_por_variable']:.0f} filas por variable). Mínimo recomendado: **{v['minimo']:,} filas**. "
             f"Margen de error al 95 %: **±{v['margen']:.1f} %**.").replace(",", ".")
    if v["nivel"] == "error":
        st.error("🚨 **Dataset demasiado pequeño.** " + texto + " Con menos de 30 registros los resultados no son confiables.")
    elif v["nivel"] == "warn":
        st.warning("⚠️ **Tamaño limitado.** " + texto + " Los modelos pueden sobreajustarse; interpreta con cautela.")
    else:
        st.success("✅ **Tamaño estadísticamente adecuado.** " + texto)
    with st.expander("¿Cómo se calcula?"):
        st.markdown(
            "- **n ≥ 30**: mínimo clásico para aplicar el Teorema del Límite Central.\n"
            "- **10 filas por variable**: regla práctica para que un modelo no memorice los datos.\n"
            "- **Margen de error** = 1,96 × √(0,25 / n). Con 385 filas se alcanza ±5 % con 95 % de confianza.")

    # ---- detalle por columna
    st.markdown("#### 🔍 Diagnóstico por columna")
    mostrar_df(perfil, hide_index=True)

    con_nulos = perfil[perfil["Nulos"] > 0].sort_values("Nulos")
    if not con_nulos.empty:
        fig = px.bar(con_nulos, x="Nulos", y="Columna", orientation="h", text="Nulos",
                     title="Valores faltantes por columna")
        fig.update_layout(height=max(250, 28 * len(con_nulos)))
        mostrar_fig(fig)

    # ---- ejemplos de texto sucio
    sucias = perfil[perfil["Texto con espacios/escapes"] > 0]["Columna"].tolist()
    if sucias:
        st.markdown("#### 🧹 Ejemplos de texto con formato incorrecto")
        for col in sucias:
            s = df[col].dropna().astype(str)
            malos = s[(s != s.str.strip()) | s.str.contains(r"\\[tnr]")].value_counts().head(8)
            st.write(f"**{col}**:", ", ".join(f"`{repr(k)}` ×{v}" for k, v in malos.items()))
        st.caption("Estos valores se corrigen automáticamente en la pestaña **5. Prep. Categórica & Balanceo**.")

    if duplicados:
        with st.expander(f"Ver las {duplicados} filas duplicadas"):
            mostrar_df(df[df.duplicated(keep=False)].head(200))
