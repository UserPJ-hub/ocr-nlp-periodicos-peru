"""
App Streamlit: OCR + NLP sobre noticias de periódicos peruanos
Ciencia de Datos II – Examen de medio curso
"""
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import spacy
import streamlit as st

BASE = Path(__file__).parent
import sys; sys.path.insert(0, str(BASE))
import utils_ocr_nlp as U

MOD = BASE / "modelos"

st.set_page_config(page_title="OCR + NLP · Periódicos peruanos", page_icon="📰", layout="wide")


# ---------------------------------------------------------------------
# Recursos (se cargan una sola vez)
# ---------------------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    nlp = spacy.load("es_core_news_md")
    clf = joblib.load(MOD / "clasificador_tfidf_logreg.joblib")
    tfidf_kw = joblib.load(MOD / "tfidf_keywords.joblib")
    cfg_path = MOD / "config_preproc.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return nlp, clf, tfidf_kw, cfg


nlp, clf, tfidf_kw, CONFIG = cargar_recursos()
CFG_DEF = CONFIG.get("preprocesamiento", dict(redim=True, enderezar=True, ruido=True,
                                              sombras=True, contraste=False, binarizacion="otsu"))

NOMBRES = {"politica": "Política", "economia": "Economía", "policial": "Policial",
           "deporte": "Deportes", "espectaculo": "Espectáculos"}

# ---------------------------------------------------------------------
# Barra lateral: parámetros del preprocesamiento
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Preprocesamiento")
    if CONFIG.get("variante"):
        st.caption(f"Valores por defecto = mejor variante del experimento: **{CONFIG['variante']}**")
    redim = st.checkbox("Redimensionar", CFG_DEF.get("redim", True))
    enderezar = st.checkbox("Corregir inclinación", CFG_DEF.get("enderezar", True))
    ruido = st.checkbox("Eliminar ruido", CFG_DEF.get("ruido", True))
    sombras = st.checkbox("Eliminar sombras", CFG_DEF.get("sombras", True))
    contraste = st.checkbox("Mejorar contraste (CLAHE)", CFG_DEF.get("contraste", False))
    opciones_bin = ["ninguna", "otsu", "adaptativa"]
    bin_def = CFG_DEF.get("binarizacion") or "ninguna"
    binar = st.selectbox("Binarización", opciones_bin, index=opciones_bin.index(bin_def))
    st.header("🔤 OCR")
    psm = st.selectbox("Modo de segmentación (Tesseract PSM)", [3, 4, 6],
                       index=[3, 4, 6].index(CONFIG.get("psm", 3)),
                       help="3 = automático (columnas) · 4 = una columna · 6 = bloque uniforme")
    n_oraciones = st.slider("Oraciones del resumen", 1, 5, 2)

cfg = dict(redim=redim, enderezar=enderezar, ruido=ruido, sombras=sombras,
           contraste=contraste, binarizacion=None if binar == "ninguna" else binar)

# ---------------------------------------------------------------------
# Cabecera y carga
# ---------------------------------------------------------------------
st.title("📰 Lector inteligente de noticias peruanas")
st.write("Sube la foto de **una nota de periódico** (JPG, PNG o PDF escaneado). "
         "La app la preprocesa, extrae el texto con OCR y lo analiza con NLP.")

archivo = st.file_uploader("Imagen o PDF", type=["jpg", "jpeg", "png", "pdf"])
ejemplos = sorted((BASE / "ejemplos").glob("*.*")) if (BASE / "ejemplos").exists() else []
if not archivo and ejemplos:
    elegido = st.selectbox("…o prueba con un ejemplo", ["—"] + [e.name for e in ejemplos])
    if elegido != "—":
        ruta = BASE / "ejemplos" / elegido
        datos, nombre = ruta.read_bytes(), elegido
    else:
        st.stop()
elif archivo:
    datos, nombre = archivo.getvalue(), archivo.name
else:
    st.info("Esperando una imagen…")
    st.stop()

# ---------------------------------------------------------------------
# 1. Preprocesamiento + OCR
# ---------------------------------------------------------------------
try:
    paginas = U.cargar_desde_bytes(datos, nombre)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

t0 = time.time()
with st.spinner("Preprocesando y leyendo el texto…"):
    textos, confs, etapas_p1 = [], [], None
    for i, img in enumerate(paginas):
        proc, etapas = U.preprocesar(img, **cfg)
        if i == 0:
            etapas_p1 = etapas
        textos.append(U.ocr_tesseract(proc, psm=psm))
        confs.append(U.confianza_tesseract(proc, psm=psm))
texto_crudo = "\n\n".join(textos)
confianza = float(np.mean(confs))
t_ocr = time.time() - t0

st.subheader("1 · Preprocesamiento de la imagen")
c1, c2 = st.columns(2)
c1.image(cv2.cvtColor(paginas[0], cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
final = list(etapas_p1.values())[-1]
c2.image(final, caption="Preprocesada (entrada del OCR)", use_container_width=True, clamp=True)
with st.expander("Ver todas las etapas"):
    nombres = [k for k in etapas_p1 if k != "original"]
    cols = st.columns(min(4, len(nombres)))
    for j, k in enumerate(nombres):
        cols[j % len(cols)].image(etapas_p1[k], caption=k, use_container_width=True, clamp=True)

# ---------------------------------------------------------------------
# 2. Texto
# ---------------------------------------------------------------------
texto_rep = U.reparar_texto_ocr(texto_crudo)
texto_limpio = U.limpiar_texto(texto_rep)
tokens = U.tokenizar(texto_limpio, nlp)
tokens_str = " ".join(tokens)

st.subheader("2 · Texto extraído")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Confianza OCR", f"{confianza:.0f}/100")
m2.metric("Palabras", len(texto_limpio.split()))
m3.metric("Tokens útiles", len(tokens))
m4.metric("Tiempo OCR", f"{t_ocr:.1f} s")
if confianza < 60:
    st.warning("Confianza baja: prueba con una foto más cercana, nítida y sin sombras.")

tab1, tab2, tab3 = st.tabs(["Texto reparado", "Texto crudo (OCR)", "Texto limpio + tokens"])
tab1.text_area("Reparado", texto_rep, height=220, label_visibility="collapsed")
tab2.text_area("Crudo", texto_crudo, height=220, label_visibility="collapsed")
tab3.text_area("Limpio", texto_limpio, height=120, label_visibility="collapsed")
tab3.write(", ".join(tokens[:80]) + (" …" if len(tokens) > 80 else ""))

if len(tokens) < 5:
    st.error("No se extrajo suficiente texto para el análisis NLP.")
    st.stop()

# ---------------------------------------------------------------------
# 3. NLP
# ---------------------------------------------------------------------
st.subheader("3 · Análisis de lenguaje natural")
izq, der = st.columns([1, 1])

with izq:
    st.markdown("**Sección estimada**")
    proba = pd.Series(clf.predict_proba([tokens_str])[0], index=clf.classes_).sort_values(ascending=False)
    pred = proba.index[0]
    st.success(f"### {NOMBRES.get(pred, pred)}  ·  {proba.iloc[0]:.0%}")
    st.bar_chart(proba.rename(index=lambda k: NOMBRES.get(k, k)), horizontal=True, height=200)

    st.markdown("**Palabras clave (TF-IDF)**")
    kw = U.palabras_clave(tokens_str, tfidf_kw, 10)
    st.write(" · ".join(f"`{w}`" for w, _ in kw) if kw else "—")

with der:
    st.markdown("**Resumen automático**")
    resumen = U.resumen_extractivo(texto_rep, nlp, n_oraciones)
    st.info(resumen or "—")

    st.markdown("**Entidades**")
    ents = U.extraer_entidades(texto_rep, nlp)
    etiquetas = {"PER": "👤 Personas", "ORG": "🏛️ Organizaciones", "LOC": "📍 Lugares"}
    for k, tit in etiquetas.items():
        if ents[k]:
            st.write(f"{tit}: " + ", ".join(e for e, _ in ents[k]))
    if ents["REGIONES_PERU"]:
        st.write("🇵🇪 Regiones del Perú: " + ", ".join(ents["REGIONES_PERU"]))

st.markdown("**Nube de palabras**")
try:
    from wordcloud import WordCloud
    wc = WordCloud(width=1000, height=350, background_color="white",
                   colormap="viridis", collocations=False).generate(tokens_str)
    st.image(wc.to_array(), use_container_width=True)
except ValueError:
    st.caption("Texto insuficiente para la nube de palabras.")

# ---------------------------------------------------------------------
# 4. Descargas
# ---------------------------------------------------------------------
st.subheader("4 · Descargar resultados")
resultado = {
    "archivo": nombre, "confianza_ocr": round(confianza, 1), "seccion_estimada": pred,
    "probabilidades": {k: round(float(v), 3) for k, v in proba.items()},
    "palabras_clave": [w for w, _ in kw], "resumen": resumen,
    "entidades": {k: [e for e, _ in v] if k != "REGIONES_PERU" else v for k, v in ents.items()},
    "texto_reparado": texto_rep, "texto_crudo": texto_crudo,
    "preprocesamiento": cfg, "psm": psm,
}
d1, d2, d3 = st.columns(3)
stem = Path(nombre).stem
d1.download_button("📄 Texto (.txt)", texto_rep, f"{stem}_texto.txt", "text/plain")
d2.download_button("🧾 Resultados (.json)", json.dumps(resultado, ensure_ascii=False, indent=2),
                   f"{stem}_resultados.json", "application/json")
fila = pd.DataFrame([{**{k: v for k, v in resultado.items() if k not in ("entidades", "probabilidades", "preprocesamiento")},
                      "palabras_clave": ", ".join(resultado["palabras_clave"])}])
d3.download_button("📊 Resultados (.csv)", fila.to_csv(index=False).encode("utf-8-sig"),
                   f"{stem}_resultados.csv", "text/csv")

st.caption("Proyecto académico · Ciencia de Datos II · UNALM")
