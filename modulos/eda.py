"""Fase 3 · Análisis exploratorio: filtros, agrupaciones (conteo, mín, máx, promedio) y gráficos."""
import pandas as pd
import plotly.express as px
import streamlit as st

from .utils import columnas_por_tipo, df_activo, intentar_numerico, limpiar_texto, mostrar_df, mostrar_fig, requiere_datos


def _vista_rapida(df: pd.DataFrame) -> pd.DataFrame:
    """Copia ligera con textos limpios y números reconocidos, solo para explorar (no altera los datos)."""
    vista = df.copy()
    for col in vista.columns:
        if not pd.api.types.is_numeric_dtype(vista[col]):
            vista[col], convertida = intentar_numerico(vista[col])
            if not convertida:
                vista[col] = limpiar_texto(vista[col])
    return vista


def _filtros(df, numericas, categoricas):
    """Filtros dinámicos por categoría y rango numérico."""
    with st.expander("🔎 Filtros", expanded=False):
        f1, f2 = st.columns(2)
        col_cat = f1.selectbox("Filtrar por categoría", ["(ninguna)"] + categoricas)
        if col_cat != "(ninguna)":
            valores = sorted(df[col_cat].dropna().astype(str).unique())[:200]
            elegidos = f1.multiselect("Valores", valores, default=valores)
            df = df[df[col_cat].astype(str).isin(elegidos)]
        col_num = f2.selectbox("Filtrar por rango numérico", ["(ninguna)"] + numericas)
        if col_num != "(ninguna)" and df[col_num].notna().any():
            mn, mx = float(df[col_num].min()), float(df[col_num].max())
            if mn < mx:
                rango = f2.slider("Rango", mn, mx, (mn, mx))
                df = df[df[col_num].between(*rango)]
    return df


def render():
    st.subheader("3. Análisis Exploratorio (EDA) & Correlaciones")
    if not requiere_datos():
        return
    df = _vista_rapida(df_activo())
    numericas, categoricas = columnas_por_tipo(df)
    df = _filtros(df, numericas, categoricas)
    st.caption(f"Registros después de filtros: **{len(df):,}**".replace(",", "."))
    if df.empty:
        st.warning("Los filtros no dejan registros.")
        return

    resumen, agrupacion, graficos = st.tabs(["📋 Resumen", "🧮 Agrupaciones", "📊 Gráficos"])

    with resumen:
        if numericas:
            st.markdown("**Variables numéricas**")
            tabla = df[numericas].describe().T.rename(columns={
                "count": "Conteo", "mean": "Promedio", "std": "Desv. estándar", "min": "Mínimo",
                "25%": "P25", "50%": "Mediana", "75%": "P75", "max": "Máximo"})
            mostrar_df(tabla.round(2))
        if categoricas:
            st.markdown("**Variables categóricas**")
            filas = [{"Columna": c, "Categorías": df[c].nunique(), "Más frecuente": df[c].mode().iloc[0] if df[c].notna().any() else "",
                      "Frecuencia": int(df[c].value_counts().iloc[0]) if df[c].notna().any() else 0} for c in categoricas]
            mostrar_df(pd.DataFrame(filas), hide_index=True)

    with agrupacion:
        candidatas = categoricas + [c for c in numericas if df[c].nunique() <= 15]
        if not candidatas or not numericas:
            st.info("Se necesita al menos una columna categórica y una numérica para agrupar.")
        else:
            a1, a2 = st.columns(2)
            grupo = a1.selectbox("Agrupar por", candidatas)
            medida = a2.selectbox("Variable a medir", numericas, index=len(numericas) - 1)
            tabla = df.groupby(grupo, dropna=False)[medida].agg(
                Conteo="count", Mínimo="min", Máximo="max", Promedio="mean", Mediana="median", Suma="sum").round(2)
            tabla = tabla.sort_values("Conteo", ascending=False)
            mostrar_df(tabla)
            mayor, menor = tabla["Promedio"].idxmax(), tabla["Promedio"].idxmin()
            st.info(f"Mayor promedio de **{medida}**: **{mayor}** ({tabla.loc[mayor, 'Promedio']:,}). "
                    f"Menor: **{menor}** ({tabla.loc[menor, 'Promedio']:,}). "
                    f"Máximo absoluto: {tabla['Máximo'].max():,} · mínimo absoluto: {tabla['Mínimo'].min():,}.")

    with graficos:
        tipo = st.radio("Tipo de gráfico", ["Distribución (histograma)", "Boxplot", "Barras", "Torta",
                                             "Dispersión", "Mapa de correlación"], horizontal=True)
        g1, g2 = st.columns(2)
        if tipo == "Distribución (histograma)" and numericas:
            x = g1.selectbox("Variable", numericas)
            color = g2.selectbox("Color por", ["(ninguno)"] + categoricas)
            fig = px.histogram(df, x=x, color=None if color == "(ninguno)" else color, nbins=40,
                               marginal="box", title=f"Distribución de {x}", barmode="overlay", opacity=0.75)
            mostrar_fig(fig)
        elif tipo == "Boxplot" and numericas:
            y = g1.selectbox("Variable numérica", numericas)
            x = g2.selectbox("Comparar por", ["(ninguno)"] + candidatas_box(df, categoricas, numericas))
            fig = px.box(df, y=y, x=None if x == "(ninguno)" else x, points="outliers", title=f"Boxplot de {y}")
            mostrar_fig(fig)
        elif tipo in ("Barras", "Torta"):
            col = g1.selectbox("Variable categórica", categoricas + numericas)
            conteo = df[col].astype(str).value_counts().head(25).reset_index()
            conteo.columns = [col, "Registros"]
            if tipo == "Barras":
                fig = px.bar(conteo, x=col, y="Registros", text="Registros", title=f"Registros por {col}")
            else:
                fig = px.pie(conteo, names=col, values="Registros", hole=0.35, title=f"Proporción de {col}")
            mostrar_fig(fig)
        elif tipo == "Dispersión" and len(numericas) >= 2:
            x = g1.selectbox("Eje X", numericas)
            y = g2.selectbox("Eje Y", numericas, index=min(1, len(numericas) - 1))
            color = st.selectbox("Color por", ["(ninguno)"] + categoricas)
            fig = px.scatter(df, x=x, y=y, color=None if color == "(ninguno)" else color, opacity=0.6,
                             title=f"{y} vs {x}")
            mostrar_fig(fig)
            st.caption(f"Correlación de Pearson: **{df[[x, y]].corr().iloc[0, 1]:.3f}** "
                       "(−1 relación inversa perfecta · 0 sin relación · +1 relación directa perfecta)")
        elif tipo == "Mapa de correlación" and len(numericas) >= 2:
            corr = df[numericas].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Matriz de correlación", aspect="auto")
            fig.update_layout(height=max(400, 40 * len(numericas)))
            mostrar_fig(fig)
        else:
            st.info("No hay columnas suficientes del tipo necesario para este gráfico.")


def candidatas_box(df, categoricas, numericas):
    """Columnas con pocas categorías, útiles para comparar boxplots."""
    return [c for c in categoricas if df[c].nunique() <= 20] + [c for c in numericas if df[c].nunique() <= 10]
