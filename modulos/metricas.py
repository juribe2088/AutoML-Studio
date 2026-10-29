"""Fase 7 · Métricas, comparación e interpretabilidad (importancia de variables y predicción de un caso)."""
import io

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)

from .modelos import LINEA_BASE
from .utils import mostrar_df, mostrar_fig, requiere_datos


# ---------------------------------------------------------------- métricas
def tabla_regresion(res) -> pd.DataFrame:
    y = res["y_test"]
    filas = []
    for nombre, m in res["modelos"].items():
        mse = mean_squared_error(y, m["pred"])
        filas.append({"Modelo": nombre, "MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y, m["pred"]),
                      "R²": r2_score(y, m["pred"])})
    return pd.DataFrame(filas).set_index("Modelo")


def config_clasificacion(res):
    """En binario se mide la clase minoritaria (la más difícil e importante); en multiclase, el promedio macro."""
    clases = res["codificador"].classes_
    if len(clases) == 2:
        minoritaria = int(pd.Series(res["y_test"]).value_counts().idxmin())  # y_test conserva la proporción real
        return {"average": "binary", "pos_label": minoritaria}, f"clase positiva = «{clases[minoritaria]}» (minoritaria)"
    return {"average": "macro"}, "promedio macro de todas las clases"


def tabla_clasificacion(res) -> pd.DataFrame:
    y = res["y_test"]
    kw, _ = config_clasificacion(res)
    filas = []
    for nombre, m in res["modelos"].items():
        p = m["pred"]
        filas.append({"Modelo": nombre, "Accuracy": accuracy_score(y, p),
                      "Precision": precision_score(y, p, zero_division=0, **kw),
                      "Recall": recall_score(y, p, zero_division=0, **kw), "F1-Score": f1_score(y, p, zero_division=0, **kw)})
    return pd.DataFrame(filas).set_index("Modelo")


def interpretar_regresion(t: pd.DataFrame, cfg) -> None:
    modelos = t.drop(index=LINEA_BASE, errors="ignore")
    mejor = modelos["RMSE"].idxmin()
    base = t.loc[LINEA_BASE, "RMSE"] if LINEA_BASE in t.index else np.nan
    mejora = 100 * (1 - modelos.loc[mejor, "RMSE"] / base) if base else np.nan
    r2 = modelos.loc[mejor, "R²"]
    st.markdown(f"🏆 **Mejor modelo: {mejor}** · error típico (RMSE) de **{modelos.loc[mejor, 'RMSE']:,.2f}** en "
                f"**{cfg['target']}**, {abs(mejora):.1f} % {'menor' if mejora >= 0 else 'mayor'} que la línea base · explica el **{100 * max(r2, 0):.1f} %** de la variación (R²).")
    if r2 < 0.1:
        st.error("🚨 **Ningún modelo aprende patrones útiles** (R² < 0,10): no superan con claridad a predecir siempre el promedio. "
                 "Suele indicar que las variables no tienen relación con el objetivo o que los datos tienen problemas de calidad. "
                 "Revisa la EDA (correlaciones) antes de confiar en estas predicciones.")
    elif r2 < 0.5:
        st.warning("El modelo capta parte del patrón, pero queda mucha variación sin explicar.")
    else:
        st.success("El modelo explica la mayor parte de la variación: buen desempeño.")


def interpretar_clasificacion(t: pd.DataFrame, descripcion) -> None:
    modelos = t.drop(index=LINEA_BASE, errors="ignore")
    mejor = modelos["F1-Score"].idxmax()
    base_acc = t.loc[LINEA_BASE, "Accuracy"] if LINEA_BASE in t.index else np.nan
    st.markdown(f"🏆 **Mejor modelo (por F1): {mejor}** · Accuracy {100 * modelos.loc[mejor, 'Accuracy']:.1f} % · "
                f"Recall {100 * modelos.loc[mejor, 'Recall']:.1f} % · F1 {100 * modelos.loc[mejor, 'F1-Score']:.1f} %  \n"
                f"<small>Precision, Recall y F1 calculados con {descripcion}.</small>", unsafe_allow_html=True)
    st.info(f"La línea base (siempre la clase más frecuente) logra **{100 * base_acc:.1f} %** de accuracy sin aprender nada: "
            "por eso, con clases desbalanceadas, Recall y F1 son más informativos que el Accuracy.")
    if modelos.loc[mejor, "F1-Score"] < 0.5:
        st.warning("El F1 es bajo: el modelo tiene dificultades para reconocer la clase de interés.")


# ---------------------------------------------------------------- interpretabilidad
def importancia_permutacion(res, nombre) -> pd.Series:
    """Cuánto empeora el modelo al desordenar cada variable original (método válido para cualquier modelo)."""
    cache = res.setdefault("_perm", {})
    if nombre not in cache:
        scoring = "r2" if res["cfg"]["tarea"] == "Regresión" else "f1_macro"
        X, y = res["X_test"], res["y_test"]
        if len(X) > 2000:  # acelera en datasets grandes
            idx = np.random.default_rng(0).choice(len(X), 2000, replace=False)
            X, y = X.iloc[idx], np.asarray(y)[idx]
        r = permutation_importance(res["modelos"][nombre]["pipeline"], X, y, n_repeats=5, random_state=0,
                                   scoring=scoring, n_jobs=-1)
        cache[nombre] = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    return cache[nombre]


def importancia_nativa(pipe):
    """Importancia propia del modelo (árboles) o magnitud de coeficientes (modelos lineales)."""
    modelo = pipe.named_steps["modelo"]
    nombres = pipe.named_steps["preparacion"].get_feature_names_out()
    if hasattr(modelo, "feature_importances_"):
        return pd.Series(modelo.feature_importances_, index=nombres).sort_values(ascending=False), "Importancia (reducción de impureza)"
    if hasattr(modelo, "coef_"):
        coef = np.atleast_2d(modelo.coef_)
        return pd.Series(np.abs(coef).mean(axis=0), index=nombres).sort_values(ascending=False), "|Coeficiente| (variables escaladas)"
    return None, None


def render_interpretabilidad(res, metricas: pd.DataFrame, columna_score: str):
    cfg = res["cfg"]
    candidatos = [n for n in res["modelos"] if n != LINEA_BASE]
    mejor = metricas.drop(index=LINEA_BASE, errors="ignore")[columna_score]
    mejor = mejor.idxmin() if columna_score == "RMSE" else mejor.idxmax()
    nombre = st.selectbox("Modelo a explicar", candidatos, index=candidatos.index(mejor))

    with st.spinner("Calculando importancia por permutación…"):
        perm = importancia_permutacion(res, nombre)
    c1, c2 = st.columns(2)
    with c1:
        top = perm.head(15)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h", title="Importancia por permutación",
                     labels={"x": "Caída del desempeño al desordenar la variable", "y": ""})
        mostrar_fig(fig)
    with c2:
        nativa, etiqueta = importancia_nativa(res["modelos"][nombre]["pipeline"])
        if nativa is not None:
            top = nativa.head(15)[::-1]
            mostrar_fig(px.bar(x=top.values, y=top.index, orientation="h", title="Importancia interna del modelo",
                               labels={"x": etiqueta, "y": ""}))
        else:
            st.info(f"**{nombre}** no expone una importancia interna; usa la importancia por permutación (izquierda), "
                    "que funciona para cualquier modelo.")

    positivas = perm[perm > 0]
    if positivas.empty:
        st.error("Ninguna variable mejora las predicciones de forma consistente: el modelo no encontró patrones reales.")
    else:
        top3 = ", ".join(f"**{k}**" for k in positivas.head(3).index)
        debil = (columna_score == "RMSE" and metricas.loc[nombre, "R²"] < 0.1) or \
                (columna_score == "F1-Score" and metricas.loc[nombre, columna_score] < 0.5)
        if debil:
            st.markdown(f"🧠 **En palabras simples:** el modelo *{nombre}* usa un poco más {top3}, pero su efecto es tan "
                        f"pequeño que no permite predecir **{cfg['target']}** con confianza.")
        else:
            st.markdown(f"🧠 **En palabras simples:** para predecir **{cfg['target']}**, el modelo *{nombre}* se apoya sobre todo "
                        f"en {top3}. Si se desordena cualquiera de ellas, las predicciones empeoran claramente.")
        sin_aporte = perm[perm <= 0].index.tolist()
        if sin_aporte:
            st.caption("Variables sin aporte medible: " + ", ".join(sin_aporte[:12]) + ("…" if len(sin_aporte) > 12 else ""))
    score = metricas.loc[nombre, columna_score]
    if (columna_score == "RMSE" and metricas.loc[nombre, "R²"] < 0.1) or (columna_score == "F1-Score" and score < 0.5):
        st.warning("⚠️ Este modelo tiene bajo desempeño: su ranking de importancia es poco confiable. "
                   "Una explicación solo vale tanto como el modelo que explica.")

    # ---- simulador de un caso
    st.markdown("#### 🔮 Predecir un caso nuevo")
    X = res["X_train"]
    with st.form("caso"):
        columnas = st.columns(3)
        valores = {}
        for i, col in enumerate(X.columns):
            serie = X[col]
            with columnas[i % 3]:
                if pd.api.types.is_numeric_dtype(serie):
                    valores[col] = st.number_input(col, value=float(serie.median()), format="%.3f")
                else:
                    opciones = serie.dropna().astype(str).value_counts().index.tolist()
                    valores[col] = st.selectbox(col, opciones)
        enviar = st.form_submit_button("Predecir")
    if enviar:
        caso = pd.DataFrame([valores])
        pipe = res["modelos"][nombre]["pipeline"]
        pred = pipe.predict(caso)[0]
        if cfg["tarea"] == "Clasificación":
            clase = res["codificador"].inverse_transform([int(pred)])[0]
            st.success(f"Predicción de **{cfg['target']}**: **{clase}**")
            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pd.Series(pipe.predict_proba(caso)[0], index=res["codificador"].classes_)
                    mostrar_fig(px.bar(x=proba.index, y=proba.values, title="Probabilidad por clase",
                                       labels={"x": "", "y": "Probabilidad"}))
                except Exception:
                    pass
        else:
            st.success(f"Predicción de **{cfg['target']}**: **{pred:,.2f}** (promedio de entrenamiento: {np.mean(res['y_train']):,.2f})")


# ---------------------------------------------------------------- interfaz
def render():
    st.subheader("7. Métricas & Explicabilidad")
    if not requiere_datos():
        return
    res = st.session_state.get("resultados")
    if not res:
        st.info("Entrena los modelos en la pestaña **6. Construcción de Modelos**.")
        return
    cfg = res["cfg"]
    comparacion, explicabilidad, descargas = st.tabs(["📊 Comparación de modelos", "🧠 Interpretabilidad", "⬇️ Descargas"])

    with comparacion:
        if cfg["tarea"] == "Regresión":
            t = tabla_regresion(res)
            interpretar_regresion(t, cfg)
            mostrar_df(t.style.format("{:,.3f}").highlight_min(subset=["MSE", "RMSE", "MAE"], color="#D1FAE5")
                       .highlight_max(subset=["R²"], color="#D1FAE5"))
            st.caption("MSE: error cuadrático medio · RMSE: raíz del MSE, en las unidades del objetivo · "
                       "MAE: error absoluto medio · R²: proporción de la variación explicada (1 = perfecto, 0 = igual que el promedio).")
            mostrar_fig(px.bar(t.reset_index(), x="Modelo", y="R²", text_auto=".3f", title="R² por modelo"))
            nombre = st.selectbox("Ver predicciones de", [n for n in res["modelos"] if n != LINEA_BASE])
            y, p = res["y_test"], res["modelos"][nombre]["pred"]
            g1, g2 = st.columns(2)
            with g1:
                fig = px.scatter(x=y, y=p, opacity=0.5, labels={"x": "Valor real", "y": "Predicción"}, title="Real vs. predicción")
                lim = [float(min(y.min(), p.min())), float(max(y.max(), p.max()))]
                fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Predicción perfecta", line=dict(dash="dash", color="gray")))
                mostrar_fig(fig)
            with g2:
                mostrar_fig(px.histogram(x=y - p, nbins=40, title="Distribución de errores (real − predicción)",
                                         labels={"x": "Error"}))
            score_col = "RMSE"
        else:
            t = tabla_clasificacion(res)
            kw, descripcion = config_clasificacion(res)
            interpretar_clasificacion(t, descripcion)
            mostrar_df(t.style.format("{:.1%}").highlight_max(color="#D1FAE5"))
            st.caption("Accuracy: % total de aciertos · Precision: de lo que el modelo predijo como positivo, cuánto era cierto · "
                       "Recall: de los positivos reales, cuántos detectó · F1: equilibrio entre Precision y Recall.")
            st.markdown("#### Matrices de confusión")
            clases = res["codificador"].classes_
            nombres = list(res["modelos"])
            columnas = st.columns(min(3, len(nombres)))
            for i, nombre in enumerate(nombres):
                cm = confusion_matrix(res["y_test"], res["modelos"][nombre]["pred"], labels=range(len(clases)))
                fig = px.imshow(cm, text_auto=True, x=[f"Pred: {c}" for c in clases], y=[f"Real: {c}" for c in clases],
                                color_continuous_scale="Blues", title=nombre, aspect="auto")
                fig.update_layout(coloraxis_showscale=False, height=320)
                with columnas[i % len(columnas)]:
                    mostrar_fig(fig)
            with st.expander("Reporte detallado por clase"):
                nombre = st.selectbox("Modelo", nombres)
                st.code(classification_report(res["y_test"], res["modelos"][nombre]["pred"], labels=range(len(clases)),
                                              target_names=[str(c) for c in clases], zero_division=0))
            score_col = "F1-Score"

    with explicabilidad:
        render_interpretabilidad(res, t, score_col)

    with descargas:
        st.download_button("⬇️ Tabla de métricas (CSV)", t.to_csv().encode("utf-8"), "metricas_modelos.csv", "text/csv")
        pred = res["X_test"].copy()
        real = res["y_test"]
        if cfg["tarea"] == "Clasificación":
            dec = res["codificador"].inverse_transform
            pred[f"{cfg['target']} (real)"] = dec(real)
            for n, m in res["modelos"].items():
                pred[f"Pred · {n}"] = dec(m["pred"].astype(int))
        else:
            pred[f"{cfg['target']} (real)"] = real
            for n, m in res["modelos"].items():
                pred[f"Pred · {n}"] = m["pred"]
        st.download_button("⬇️ Predicciones del conjunto de prueba (CSV)", pred.to_csv(index=False).encode("utf-8"),
                           "predicciones.csv", "text/csv")
        modelo = st.selectbox("Modelo entrenado a exportar", [n for n in res["modelos"] if n != LINEA_BASE])
        buffer = io.BytesIO()
        joblib.dump({"pipeline": res["modelos"][modelo]["pipeline"], "codificador_objetivo": res["codificador"],
                     "config": cfg}, buffer)
        st.download_button(f"⬇️ Modelo «{modelo}» (.joblib)", buffer.getvalue(), "modelo_entrenado.joblib")
        st.caption("Cargar luego con `joblib.load('modelo_entrenado.joblib')['pipeline'].predict(nuevos_datos)`.")
