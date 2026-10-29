# AutoML Studio · Machine Learning guiado

Aplicación web en **Streamlit** para entrenar modelos de Machine Learning paso a paso, sin escribir código.

| Pestaña | Qué hace |
|---|---|
| 1. Carga | Archivos CSV/Excel (varios a la vez, todas las hojas), URL pública (Google Drive, Google Sheets, GitHub) y 5 datasets de ejemplo |
| 2. Validación | Nulos, duplicados, cadenas vacías, texto sucio, números guardados como texto y **validez estadística del tamaño de muestra** |
| 3. EDA | Filtros, agrupaciones (conteo, mín, máx, promedio), histogramas, boxplot, barras, torta, dispersión y mapa de correlación |
| 4. Tarea y Variables | Regresión o Clasificación (con sugerencia automática), variable objetivo y predictoras, alerta de desbalance |
| 5. Preparación & Balanceo | Limpieza de texto, imputación, One-Hot / Ordinal, escalado, train/test y balanceo (pesos, sobre/submuestreo) |
| 6. Modelos | Entrena 3 o más modelos a la vez + una línea base para comparar |
| 7. Métricas & Explicabilidad | MSE, RMSE, MAE, R² · Accuracy, Precision, Recall, F1, matrices de confusión · importancia de variables · predicción de un caso · descargas |

## Estructura

```
automl_studio/
├── app.py                  # Punto de entrada (pestañas y barra lateral)
├── modulos/
│   ├── utils.py            # Funciones compartidas (limpieza de texto, gráficos)
│   ├── carga.py            # 1. Carga de datos
│   ├── validacion.py       # 2. Validación y calidad
│   ├── eda.py              # 3. Análisis exploratorio
│   ├── tarea.py            # 4. Tarea y variables
│   ├── preparacion.py      # 5. Codificación, escalado, división y balanceo
│   ├── modelos.py          # 6. Catálogo y entrenamiento
│   └── metricas.py         # 7. Métricas e interpretabilidad
├── data/                   # Datasets del taller (gimnasio y clima)
├── .streamlit/config.toml  # Tema visual y límite de carga (200 MB)
├── requirements.txt
└── Dockerfile
```

## Opción 1 · Ejecutar en tu computador

Requiere Python 3.10 o superior.

```bash
cd automl_studio
python -m venv .venv
# Windows:  .venv\Scripts\activate      ·   Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. Para que otros equipos de la red la vean: `streamlit run app.py --server.address 0.0.0.0` y entrar por `http://IP-del-equipo:8501`.

## Opción 2 · Streamlit Community Cloud (gratis, URL pública)

1. Sube la carpeta a un repositorio de GitHub (puede ser privado).
2. Entra a <https://share.streamlit.io> con tu cuenta de GitHub → **Create app**.
3. Elige el repositorio, la rama y el archivo `app.py` → **Deploy**.
4. Obtienes una URL del tipo `https://tu-app.streamlit.app`.

> La app se "duerme" tras unos días sin uso; se reactiva con un clic.

## Opción 3 · Servidor interno con Docker

```bash
docker build -t automl-studio .
docker run -d --name automl-studio --restart unless-stopped -p 8501:8501 automl-studio
```

Para publicarla con un nombre interno (ej. `automl.empresa.local`) usa un proxy inverso (IIS con URL Rewrite + ARR, o nginx) que reenvíe al puerto 8501 **con soporte WebSocket** (Streamlit lo necesita).

Sin Docker en Windows Server: instala Python, crea el entorno virtual como en la opción 1 y regístrala como servicio con NSSM:

```bat
nssm install AutoMLStudio "C:\automl_studio\.venv\Scripts\streamlit.exe" "run app.py --server.port 8501 --server.headless true"
nssm set AutoMLStudio AppDirectory C:\automl_studio
nssm start AutoMLStudio
```

## Usar un modelo exportado

```python
import joblib, pandas as pd
paquete = joblib.load("modelo_entrenado.joblib")
predicciones = paquete["pipeline"].predict(pd.read_csv("nuevos_datos.csv"))
# En clasificación: paquete["codificador_objetivo"].inverse_transform(predicciones)
```
