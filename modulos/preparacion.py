"""Fase 5 · Preparación categórica y balanceo.

Incluye limpieza de texto, tipado, duplicados, imputación, codificación de categorías,
escalado, división train/test y balanceo de clases. La imputación, codificación y escalado
se ajustan SOLO con el conjunto de entrenamiento (dentro de un Pipeline) para evitar fuga de datos.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from .utils import df_activo, intentar_numerico, limpiar_texto, mostrar_df, mostrar_fig, requiere_datos

OPCIONES_DEFECTO = {
    "limpiar_texto": True, "duplicados": True, "imp_num": "Mediana", "imp_cat": "Moda (más frecuente)",
    "codificacion": "One-Hot", "escalado": "Estandarización (z-score)", "balanceo": "Ninguno",
    "test_size": 0.2, "semilla": 42, "max_categorias": 50,
}


# ---------------------------------------------------------------- lógica
def limpiar_dataset(df: pd.DataFrame, cfg: dict, prep: dict):
    """Aplica las limpiezas que no dependen del entrenamiento. Devuelve (df_limpio, bitácora)."""
    log = []
    datos = df[cfg["features"] + [cfg["target"]]].copy()

    if prep["limpiar_texto"]:
        for col in datos.columns:
            if pd.api.types.is_numeric_dtype(datos[col]):
                continue
            convertida, ok = intentar_numerico(datos[col])
            if ok:
                datos[col] = convertida
                log.append(f"🔢 **{col}**: convertida de texto a número.")
            else:
                antes = datos[col].astype(str).nunique()
                datos[col] = limpiar_texto(datos[col])
                despues = datos[col].nunique()
                if despues < antes:
                    log.append(f"🧹 **{col}**: texto normalizado ({antes} → {despues} categorías distintas).")

    sin_target = int(datos[cfg["target"]].isna().sum())
    if sin_target:
        datos = datos.dropna(subset=[cfg["target"]])
        log.append(f"🗑️ {sin_target} filas eliminadas por no tener valor en la variable objetivo.")

    if prep["duplicados"]:
        n = int(datos.duplicated().sum())
        datos = datos.drop_duplicates()
        log.append(f"🗑️ {n} filas duplicadas eliminadas." if n else "✅ No se encontraron filas duplicadas.")

    features = [c for c in datos.columns if c != cfg["target"]]
    num = datos[features].select_dtypes("number").columns.tolist()
    cat = [c for c in features if c not in num]
    if prep["imp_num"] == "Eliminar filas" and num:
        n = len(datos)
        datos = datos.dropna(subset=num)
        log.append(f"🗑️ {n - len(datos)} filas eliminadas por nulos en columnas numéricas.")
    if prep["imp_cat"] == "Eliminar filas" and cat:
        n = len(datos)
        datos = datos.dropna(subset=cat)
        log.append(f"🗑️ {n - len(datos)} filas eliminadas por nulos en columnas categóricas.")

    # Columnas inútiles: vacías, constantes o identificadores de texto con demasiadas categorías
    for col in features:
        unicos = datos[col].nunique(dropna=True)
        if unicos <= 1:
            datos = datos.drop(columns=col)
            log.append(f"✂️ **{col}** excluida: es constante o está vacía.")
        elif col in cat and unicos > prep["max_categorias"]:
            datos = datos.drop(columns=col)
            log.append(f"✂️ **{col}** excluida: {unicos} categorías distintas (parece un identificador o texto libre).")

    imputables = [c for c in datos.columns if c != cfg["target"] and datos[c].isna().any()]
    if imputables:
        log.append(f"🩹 Se imputarán nulos en {len(imputables)} columnas ({', '.join(imputables[:6])}"
                   f"{'…' if len(imputables) > 6 else ''}) con valores calculados solo en entrenamiento.")
    return datos, log


def construir_preprocesador(X: pd.DataFrame, prep: dict) -> ColumnTransformer:
    """Crea el transformador de columnas: imputación + escalado (numéricas) e imputación + codificación (categóricas)."""
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]

    pasos_num = [("imputar", SimpleImputer(strategy="mean" if prep["imp_num"] == "Media" else "median"))]
    if prep["escalado"].startswith("Estandarización"):
        pasos_num.append(("escalar", StandardScaler()))
    elif prep["escalado"].startswith("Min-Max"):
        pasos_num.append(("escalar", MinMaxScaler()))

    if prep["imp_cat"].startswith("Categoría"):
        imputador_cat = SimpleImputer(strategy="constant", fill_value="Desconocido")
    else:
        imputador_cat = SimpleImputer(strategy="most_frequent")
    if prep["codificacion"] == "One-Hot":
        codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    transformadores = []
    if num:
        transformadores.append(("num", Pipeline(pasos_num), num))
    if cat:
        transformadores.append(("cat", Pipeline([("imputar", imputador_cat), ("codificar", codificador)]), cat))
    return ColumnTransformer(transformadores, remainder="drop", verbose_feature_names_out=False)


def dividir(datos: pd.DataFrame, cfg: dict, prep: dict):
    """Separa entrenamiento y prueba (estratificado en clasificación cuando es posible)."""
    X = datos.drop(columns=cfg["target"])
    y = datos[cfg["target"]]
    if cfg["tarea"] == "Clasificación":
        y = y.astype(str)
    estratificar = y if cfg["tarea"] == "Clasificación" and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=prep["test_size"], random_state=prep["semilla"], stratify=estratificar)


def balancear(X: pd.DataFrame, y: pd.Series, metodo: str, semilla: int):
    """Sobremuestreo o submuestreo aleatorio aplicado SOLO al conjunto de entrenamiento."""
    if metodo not in ("Sobremuestreo aleatorio", "Submuestreo aleatorio"):
        return X, y
    rng = np.random.default_rng(semilla)
    conteo = y.value_counts()
    objetivo = conteo.max() if metodo.startswith("Sobre") else conteo.min()
    indices = []
    for clase, n in conteo.items():
        idx = y.index[y == clase].to_numpy()
        reemplazo = objetivo > n
        indices.extend(rng.choice(idx, size=objetivo, replace=reemplazo))
    indices = rng.permutation(indices)
    return X.loc[indices].reset_index(drop=True), y.loc[indices].reset_index(drop=True)


def preparar_todo():
    """Ejecuta limpieza + división + balanceo con la configuración vigente. Usado por el módulo de modelos."""
    cfg, prep = st.session_state["config"], st.session_state["prep"]
    datos, log = limpiar_dataset(df_activo(), cfg, prep)
    X_tr, X_te, y_tr, y_te = dividir(datos, cfg, prep)
    if cfg["tarea"] == "Clasificación":
        X_tr, y_tr = balancear(X_tr, y_tr, prep["balanceo"], prep["semilla"])
    return datos, log, X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------- interfaz
def render():
    st.subheader("5. Preparación Categórica & Balanceo")
    if not requiere_datos():
        return
    cfg = st.session_state.get("config")
    if not cfg:
        st.info("Primero define la tarea y las variables en la pestaña **4. Tarea y Variables**.")
        return
    prep = st.session_state.setdefault("prep", dict(OPCIONES_DEFECTO))
    previo = dict(prep)

    st.markdown("##### 🧹 Limpieza")
    a, b, c = st.columns(3)
    prep["limpiar_texto"] = a.toggle("Normalizar texto y convertir números guardados como texto", prep["limpiar_texto"],
                                     help="Quita espacios y \\t \\n; 'Cardio', ' Cardio' y '\\tCardio' pasan a ser una sola categoría.")
    prep["duplicados"] = b.toggle("Eliminar filas duplicadas", prep["duplicados"])
    prep["max_categorias"] = c.number_input("Máx. categorías por columna de texto", 5, 1000, prep["max_categorias"],
                                            help="Columnas con más categorías (ej. nombres, IDs) se excluyen.")

    st.markdown("##### 🩹 Valores faltantes · 🔤 Codificación · 📐 Escalado")
    a, b, c, d = st.columns(4)
    opciones_num = ["Mediana", "Media", "Eliminar filas"]
    opciones_cat = ["Moda (más frecuente)", "Categoría 'Desconocido'", "Eliminar filas"]
    prep["imp_num"] = a.selectbox("Nulos numéricos", opciones_num, opciones_num.index(prep["imp_num"]),
                                  help="La mediana resiste mejor los valores extremos que la media.")
    prep["imp_cat"] = b.selectbox("Nulos categóricos", opciones_cat, opciones_cat.index(prep["imp_cat"]))
    opciones_cod = ["One-Hot", "Ordinal"]
    prep["codificacion"] = c.selectbox("Codificación categórica", opciones_cod, opciones_cod.index(prep["codificacion"]),
                                       help="One-Hot crea una columna 0/1 por categoría (recomendado). "
                                            "Ordinal asigna 0, 1, 2… (útil con muchos valores o árboles).")
    opciones_esc = ["Estandarización (z-score)", "Min-Max (0 a 1)", "Ninguno"]
    prep["escalado"] = d.selectbox("Escalado numérico", opciones_esc, opciones_esc.index(prep["escalado"]),
                                   help="Necesario para Regresión Logística, SVM y KNN; indiferente para árboles.")

    st.markdown("##### ✂️ División y ⚖️ Balanceo")
    a, b, c = st.columns(3)
    prep["test_size"] = a.slider("% de datos para prueba", 10, 40, int(prep["test_size"] * 100), 5) / 100
    prep["semilla"] = b.number_input("Semilla aleatoria", 0, 9999, prep["semilla"],
                                     help="Fija el azar para que los resultados sean reproducibles.")
    opciones_bal = ["Ninguno", "Pesos de clase (class_weight)", "Sobremuestreo aleatorio", "Submuestreo aleatorio"]
    if cfg["tarea"] == "Clasificación":
        prep["balanceo"] = c.selectbox("Balanceo de clases", opciones_bal, opciones_bal.index(prep["balanceo"]),
                                       help="Se aplica solo al conjunto de entrenamiento; la prueba conserva la proporción real.")
    else:
        prep["balanceo"] = "Ninguno"
        c.caption("El balanceo solo aplica a problemas de clasificación.")

    if prep != previo:
        st.session_state.pop("resultados", None)

    # ---- vista previa del resultado
    try:
        datos, log, X_tr, X_te, y_tr, y_te = preparar_todo()
    except Exception as error:
        st.error(f"No se pudo preparar el dataset: {error}")
        return

    st.divider()
    st.markdown("#### Resultado de la preparación")
    for linea in log:
        st.markdown(f"- {linea}")

    original = df_activo()
    m = st.columns(4)
    m[0].metric("Filas originales", f"{len(original):,}".replace(",", "."))
    m[1].metric("Filas limpias", f"{len(datos):,}".replace(",", "."), f"{len(datos) - len(original)}")
    m[2].metric("Entrenamiento", f"{len(X_tr):,}".replace(",", "."))
    m[3].metric("Prueba", f"{len(X_te):,}".replace(",", "."))

    try:
        pre = construir_preprocesador(X_tr, prep).fit(X_tr)
        matriz = pd.DataFrame(pre.transform(X_te.head(8)), columns=pre.get_feature_names_out()).round(3)
        st.markdown(f"**Matriz que recibe el modelo** · {X_tr.shape[1]} variables originales → "
                    f"**{matriz.shape[1]} columnas** después de codificar")
        mostrar_df(matriz)
    except Exception as error:
        st.error(f"Error al codificar: {error}")

    if cfg["tarea"] == "Clasificación":
        antes = dividir(datos, cfg, prep)[2].value_counts()
        despues = y_tr.value_counts()
        fig = go.Figure([go.Bar(name="Antes del balanceo", x=antes.index, y=antes.values, text=antes.values),
                         go.Bar(name="Después del balanceo", x=despues.index, y=despues.values, text=despues.values)])
        fig.update_layout(barmode="group", title="Clases en el conjunto de entrenamiento")
        mostrar_fig(fig)
        if prep["balanceo"].startswith("Pesos"):
            st.caption("Con *pesos de clase* no se duplican filas: el modelo penaliza más los errores en la clase minoritaria.")

    with st.expander("Ver dataset limpio completo"):
        mostrar_df(datos.head(500))
        st.download_button("⬇️ Descargar dataset limpio (CSV)", datos.to_csv(index=False).encode("utf-8"),
                           "dataset_limpio.csv", "text/csv")
