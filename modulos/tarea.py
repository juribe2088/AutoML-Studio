"""Fase 4 · Selección de la tarea (regresión / clasificación), variable objetivo y predictoras."""
import pandas as pd
import plotly.express as px
import streamlit as st

from .eda import _vista_rapida
from .utils import df_activo, mostrar_fig, requiere_datos


def sugerir_tarea(y: pd.Series) -> str:
    """Clasificación si el objetivo es texto o tiene pocos valores distintos; si no, regresión."""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
        return "Clasificación"
    return "Regresión"


def render():
    st.subheader("4. Selección de Tarea y Variables")
    if not requiere_datos():
        return
    df = _vista_rapida(df_activo())
    clave = st.session_state["activo"]  # las claves dependen del dataset para no mezclar selecciones

    columnas = df.columns.tolist()
    t1, t2 = st.columns(2)
    sugerido = st.session_state.get("objetivo_sugerido", {}).get(clave)
    indice = columnas.index(sugerido) if sugerido in columnas else len(columnas) - 1
    target = t1.selectbox("🎯 Variable objetivo (target)", columnas, index=indice, key=f"target_{clave}")
    sugerida = sugerir_tarea(df[target])
    tarea = t2.radio("Tipo de problema", ["Regresión", "Clasificación"],
                     index=["Regresión", "Clasificación"].index(sugerida), horizontal=True, key=f"tarea_{clave}_{target}",
                     help="Regresión = predecir un número · Clasificación = predecir una categoría")
    t2.caption(f"Sugerencia automática: **{sugerida}** "
               f"({'texto o pocos valores distintos' if sugerida == 'Clasificación' else 'variable numérica continua'}).")

    disponibles = [c for c in columnas if c != target]
    features = st.multiselect("🧩 Variables predictoras (features)", disponibles, default=disponibles,
                              key=f"features_{clave}_{target}")

    y = df[target]
    errores = []
    if not features:
        errores.append("Selecciona al menos una variable predictora.")
    if tarea == "Regresión" and not pd.api.types.is_numeric_dtype(y):
        errores.append(f"**{target}** no es numérica: no se puede usar en regresión. Elige Clasificación.")
    if tarea == "Clasificación" and y.nunique() > 50:
        errores.append(f"**{target}** tiene {y.nunique()} valores distintos: es demasiado para clasificación. ¿Querías Regresión?")
    for e in errores:
        st.error(e)

    if y.isna().any():
        st.warning(f"{int(y.isna().sum())} filas no tienen valor en **{target}** y se excluirán del entrenamiento "
                   "(no se puede aprender de una respuesta desconocida).")

    # ---- análisis de la variable objetivo
    st.markdown("#### Variable objetivo")
    if tarea == "Clasificación":
        conteo = y.astype(str).value_counts().reset_index()
        conteo.columns = ["Clase", "Registros"]
        conteo["%"] = (100 * conteo["Registros"] / conteo["Registros"].sum()).round(1)
        c1, c2 = st.columns([2, 1])
        with c1:
            mostrar_fig(px.bar(conteo, x="Clase", y="Registros", text="%", title=f"Distribución de clases de {target}"))
        ratio = conteo["Registros"].max() / max(conteo["Registros"].min(), 1)
        c2.metric("Clases", len(conteo))
        c2.metric("Desbalance (mayor/menor)", f"{ratio:.1f} : 1")
        if conteo["Registros"].min() < 10:
            st.error("Hay clases con menos de 10 registros: el modelo casi no tendrá ejemplos para aprenderlas.")
        if ratio >= 3:
            st.warning(f"⚖️ **Clases desbalanceadas.** Un modelo que siempre responda **{conteo.iloc[0]['Clase']}** "
                       f"acertaría el {conteo.iloc[0]['%']} % sin aprender nada. Revisa Recall/F1 y considera "
                       "aplicar balanceo en la pestaña 5.")
    else:
        mostrar_fig(px.histogram(df, x=target, nbins=40, marginal="box", title=f"Distribución de {target}"))

    # ---- guardar configuración
    if not errores:
        nueva = {"dataset": clave, "tarea": tarea, "target": target, "features": features}
        if st.session_state.get("config") != nueva:
            st.session_state["config"] = nueva
            st.session_state.pop("resultados", None)  # los modelos anteriores ya no aplican
        st.success(f"Configuración lista: **{tarea}** de **{target}** con {len(features)} variables predictoras.")
    else:
        st.session_state.pop("config", None)
