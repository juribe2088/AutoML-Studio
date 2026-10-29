"""Fase 6 · Construcción de modelos: catálogo, entrenamiento simultáneo y comparación con una línea base."""
import time

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import preparacion
from .utils import requiere_datos

try:  # XGBoost es opcional
    from xgboost import XGBClassifier, XGBRegressor
    HAY_XGB = True
except Exception:
    HAY_XGB = False

LINEA_BASE = "Línea base (sin aprender)"


def catalogo(tarea: str, pesos: bool, semilla: int) -> dict:
    """Modelos disponibles según la tarea. `pesos` activa class_weight='balanced' cuando el modelo lo admite."""
    cw = "balanced" if pesos else None
    if tarea == "Clasificación":
        modelos = {
            "Regresión Logística": lambda: LogisticRegression(max_iter=2000, class_weight=cw),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=semilla, n_jobs=-1),
            "SVM (Support Vector Machine)": lambda: SVC(kernel="rbf", class_weight=cw, random_state=semilla),
            "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=semilla),
            "Árbol de Decisión": lambda: DecisionTreeClassifier(max_depth=6, class_weight=cw, random_state=semilla),
            "K Vecinos (KNN)": lambda: KNeighborsClassifier(n_neighbors=7),
        }
        if HAY_XGB:
            modelos["XGBoost"] = lambda: XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=semilla, verbosity=0)
    else:
        modelos = {
            "Regresión Lineal": lambda: LinearRegression(),
            "Random Forest": lambda: RandomForestRegressor(n_estimators=300, random_state=semilla, n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=semilla),
            "Ridge (lineal regularizada)": lambda: Ridge(alpha=1.0),
            "Árbol de Decisión": lambda: DecisionTreeRegressor(max_depth=6, random_state=semilla),
            "K Vecinos (KNN)": lambda: KNeighborsRegressor(n_neighbors=7),
            "SVR (Support Vector)": lambda: SVR(),
        }
        if HAY_XGB:
            modelos["XGBoost"] = lambda: XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=semilla, verbosity=0)
    return modelos


def entrenar(seleccion: list) -> dict:
    """Prepara los datos, entrena cada modelo dentro de un Pipeline y guarda predicciones sobre el conjunto de prueba."""
    cfg, prep = st.session_state["config"], st.session_state["prep"]
    datos, _, X_tr, X_te, y_tr, y_te = preparacion.preparar_todo()

    codificador = None
    if cfg["tarea"] == "Clasificación":
        codificador = LabelEncoder().fit(pd.concat([y_tr, y_te]))
        y_tr_m, y_te_m = codificador.transform(y_tr), codificador.transform(y_te)
    else:
        y_tr_m, y_te_m = y_tr.to_numpy(), y_te.to_numpy()

    pesos = prep["balanceo"].startswith("Pesos")
    fabricas = catalogo(cfg["tarea"], pesos, prep["semilla"])
    fabricas[LINEA_BASE] = (lambda: DummyClassifier(strategy="most_frequent")) if cfg["tarea"] == "Clasificación" \
        else (lambda: DummyRegressor(strategy="mean"))

    modelos, barra = {}, st.progress(0.0, "Entrenando…")
    nombres = seleccion + [LINEA_BASE]
    for i, nombre in enumerate(nombres, 1):
        barra.progress(i / len(nombres), f"Entrenando {nombre}…")
        pipe = Pipeline([("preparacion", preparacion.construir_preprocesador(X_tr, prep)), ("modelo", fabricas[nombre]())])
        inicio = time.perf_counter()
        try:
            pipe.fit(X_tr, y_tr_m)
            modelos[nombre] = {"pipeline": pipe, "pred": pipe.predict(X_te), "segundos": time.perf_counter() - inicio}
        except Exception as error:
            st.error(f"{nombre} falló: {error}")
    barra.empty()
    return {"cfg": dict(cfg), "prep": dict(prep), "modelos": modelos, "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr_m, "y_test": y_te_m, "codificador": codificador, "datos": datos}


def render():
    st.subheader("6. Construcción de Modelos")
    if not requiere_datos():
        return
    cfg = st.session_state.get("config")
    if not cfg:
        st.info("Primero define la tarea y las variables en la pestaña **4. Tarea y Variables**.")
        return
    prep = st.session_state.setdefault("prep", dict(preparacion.OPCIONES_DEFECTO))

    st.markdown(f"Tarea: **{cfg['tarea']}** · Objetivo: **{cfg['target']}** · {len(cfg['features'])} variables · "
                f"Prueba: **{int(prep['test_size'] * 100)} %** · Balanceo: **{prep['balanceo']}**")
    disponibles = list(catalogo(cfg["tarea"], False, 0))
    seleccion = st.multiselect("Modelos a entrenar (mínimo 3)", disponibles, default=disponibles[:3])
    st.caption(f"Se agrega automáticamente una **{LINEA_BASE.lower()}**: "
               + ("siempre predice la clase más frecuente." if cfg["tarea"] == "Clasificación" else "siempre predice el promedio.")
               + " Un modelo útil debe superarla con claridad.")
    if not HAY_XGB:
        st.caption("XGBoost no está instalado (`pip install xgboost` para habilitarlo).")

    if st.button("🚀 Entrenar modelos", type="primary", disabled=len(seleccion) < 3):
        with st.spinner("Preparando datos y entrenando…"):
            st.session_state["resultados"] = entrenar(seleccion)
    if len(seleccion) < 3:
        st.warning("Selecciona al menos 3 modelos.")

    res = st.session_state.get("resultados")
    if res:
        tiempos = pd.DataFrame({"Modelo": list(res["modelos"]),
                                "Tiempo de entrenamiento (s)": [round(m["segundos"], 2) for m in res["modelos"].values()]})
        st.success(f"✅ {len(res['modelos']) - 1} modelos entrenados con {len(res['X_train']):,} filas y evaluados con "
                   f"{len(res['X_test']):,} filas que nunca vieron. Revisa la pestaña **7. Métricas & Explicabilidad**."
                   .replace(",", "."))
        st.dataframe(tiempos, hide_index=True)
