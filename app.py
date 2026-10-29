"""AutoML Studio · Machine Learning guiado con Streamlit.

Ejecutar con:  streamlit run app.py
"""
import streamlit as st

from modulos import carga, eda, metricas, modelos, preparacion, tarea, validacion

st.set_page_config(page_title="AutoML Studio", page_icon="🤖", layout="wide")

# ---- estado inicial de la sesión
st.session_state.setdefault("datasets", {})
st.session_state.setdefault("activo", None)
st.session_state.setdefault("archivos_leidos", set())
st.session_state.setdefault("prep", dict(preparacion.OPCIONES_DEFECTO))

# ---- estilos
st.markdown("""
<style>
.block-container {padding-top: 3.5rem;}
.encabezado {display:flex; align-items:center; gap:14px; margin-bottom:.4rem;}
.logo {width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg,#6366F1,#4F46E5);
       display:flex; align-items:center; justify-content:center; font-size:26px;}
.etiqueta {background:#EEF2FF; color:#4338CA; border-radius:999px; padding:2px 10px; font-size:12px; font-weight:600;}
.stTabs [data-baseweb="tab-list"] {gap: 4px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {background:#F1F5F9; border-radius:8px; padding:6px 14px;}
.stTabs [aria-selected="true"] {background:#4F46E5 !important; color:white !important;}
</style>
<div class="encabezado">
  <div class="logo">✨</div>
  <div><span style="font-size:1.5rem;font-weight:700">AutoML Studio</span> <span class="etiqueta">ML Modular Guiado</span><br>
  <span style="color:#64748B">Carga, validación estadística, EDA, codificación categórica, balanceo, modelos y explicabilidad</span></div>
</div>
""", unsafe_allow_html=True)

pestanas = st.tabs(["1. Carga", "2. Validación", "3. EDA", "4. Tarea y Variables",
                    "5. Preparación & Balanceo", "6. Modelos", "7. Métricas & Explicabilidad"])
for pestana, modulo in zip(pestanas, [carga, validacion, eda, tarea, preparacion, modelos, metricas]):
    with pestana:
        try:
            modulo.render()
        except Exception as error:  # una falla en un módulo no tumba toda la app
            st.error(f"Ocurrió un error en este módulo: {error}")
            st.exception(error)

# ---- barra lateral: progreso del flujo (se dibuja al final para reflejar el estado actualizado)
with st.sidebar:
    st.markdown("### 🧭 Progreso")
    cfg = st.session_state.get("config")
    pasos = [
        ("Datos cargados", st.session_state["activo"] is not None),
        ("Tarea y variables definidas", cfg is not None),
        ("Modelos entrenados", "resultados" in st.session_state),
    ]
    for texto, hecho in pasos:
        st.markdown(f"{'✅' if hecho else '⬜'} {texto}")
    if st.session_state["activo"]:
        st.divider()
        st.caption("Dataset activo")
        st.markdown(f"**{st.session_state['activo']}**")
    if cfg:
        st.caption("Configuración")
        st.markdown(f"{cfg['tarea']} · objetivo **{cfg['target']}**")
    st.divider()
    if st.button("🔄 Reiniciar sesión"):
        st.session_state.clear()
        st.rerun()
