"""
utils_ocr_nlp.py
Funciones compartidas entre el notebook de Colab y la app de Streamlit.
Proyecto: OCR + NLP sobre noticias de periódicos peruanos.
"""
import re
import unicodedata
from collections import Counter

import cv2
import numpy as np
import pytesseract

# =====================================================================
# 1. CARGA DE IMÁGENES (JPG, PNG y PDF escaneado)
# =====================================================================
def cargar_desde_bytes(datos: bytes, nombre: str = ""):
    """Devuelve una lista de imágenes BGR (una por página si es PDF)."""
    if nombre.lower().endswith(".pdf"):
        import fitz  # PyMuPDF
        paginas = []
        with fitz.open(stream=datos, filetype="pdf") as doc:
            for pag in doc:
                pix = pag.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
                paginas.append(img)
        return paginas
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo leer la imagen")
    return [img]


# =====================================================================
# 2. PREPROCESAMIENTO DE IMÁGENES
# =====================================================================
def a_gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def redimensionar(gray, ancho_objetivo=2000):
    """Tesseract rinde mejor con letras de ~25-35 px de alto.
    Agranda fotos pequeñas (INTER_CUBIC) y reduce las enormes (INTER_AREA)."""
    h, w = gray.shape[:2]
    escala = ancho_objetivo / w
    if 0.9 < escala < 1.1:
        return gray
    interp = cv2.INTER_CUBIC if escala > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=escala, fy=escala, interpolation=interp)


def corregir_inclinacion(gray, max_angulo=6, paso=0.5):
    """Endereza la foto buscando el ángulo que maximiza la nitidez
    del perfil de proyección horizontal (las líneas de texto)."""
    peq = cv2.resize(gray, (800, int(800 * gray.shape[0] / gray.shape[1])))
    _, binv = cv2.threshold(peq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    centro = (binv.shape[1] // 2, binv.shape[0] // 2)
    mejor_ang, mejor_score = 0.0, -1
    for ang in np.arange(-max_angulo, max_angulo + paso, paso):
        M = cv2.getRotationMatrix2D(centro, ang, 1.0)
        rot = cv2.warpAffine(binv, M, (binv.shape[1], binv.shape[0]), borderValue=0)
        perfil = rot.sum(axis=1).astype(np.float64)
        score = np.sum((perfil[1:] - perfil[:-1]) ** 2)
        if score > mejor_score:
            mejor_ang, mejor_score = ang, score
    if mejor_ang == 0:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), mejor_ang, 1.0)
    fondo = int(np.median(gray))  # rellena bordes con el color del papel
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fondo)


def eliminar_ruido(gray, h=10):
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def quitar_sombras(gray, k=31):
    """Normalización de fondo: estima la iluminación del papel con un
    cierre morfológico + desenfoque y la divide. Elimina sombras y degradados."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fondo = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    fondo = cv2.medianBlur(fondo, k if k % 2 else k + 1)
    norm = cv2.divide(gray, fondo, scale=255)
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)


def mejorar_contraste(gray, clip=1.5):
    """CLAHE: ecualización adaptativa, corrige sombras e iluminación irregular."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def binarizar(gray, metodo="adaptativa"):
    if metodo == "otsu":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return b
    if metodo == "adaptativa":
        bloque = max(31, (gray.shape[1] // 40) | 1)       # impar y proporcional
        b = cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, bloque, 20)
        return quitar_motas(b)
    return gray


def quitar_motas(binaria, area_min=40):
    """Elimina puntos negros aislados (ruido) tras binarizar."""
    inv = 255 - binaria
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    limpio = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            limpio[lab == i] = 255
    return 255 - limpio


def preprocesar(img, redim=True, enderezar=True, ruido=True, sombras=True,
                contraste=False, binarizacion="otsu"):
    """Pipeline completo. Devuelve (imagen_final, dict_de_etapas)."""
    etapas = {"original": img}
    g = a_gris(img);                      etapas["gris"] = g
    if redim:     g = redimensionar(g);   etapas["redimensionada"] = g
    if enderezar: g = corregir_inclinacion(g); etapas["enderezada"] = g
    if ruido:     g = eliminar_ruido(g);  etapas["sin_ruido"] = g
    if sombras:   g = quitar_sombras(g);  etapas["sin_sombras"] = g
    if contraste: g = mejorar_contraste(g); etapas["contraste"] = g
    if binarizacion:
        g = binarizar(g, binarizacion);   etapas[f"binarizada_{binarizacion}"] = g
    return g, etapas


# Variantes para el experimento (qué paso aporta realmente)
_B = dict(redim=False, enderezar=False, ruido=False, sombras=False, contraste=False, binarizacion=None)
VARIANTES = {
    "V0_solo_gris":        {**_B},
    "V1_+redim":           {**_B, "redim": True},
    "V2_+enderezar":       {**_B, "redim": True, "enderezar": True},
    "V3_+ruido":           {**_B, "redim": True, "enderezar": True, "ruido": True},
    "V4_+sombras":         {**_B, "redim": True, "enderezar": True, "ruido": True, "sombras": True},
    "V5_+CLAHE":           {**_B, "redim": True, "enderezar": True, "ruido": True, "sombras": True, "contraste": True},
    "V6_+otsu":            {**_B, "redim": True, "enderezar": True, "ruido": True, "sombras": True, "binarizacion": "otsu"},
    "V7_+adaptativa":      {**_B, "redim": True, "enderezar": True, "ruido": True, "sombras": True, "binarizacion": "adaptativa"},
}


# =====================================================================
# 3. OCR
# =====================================================================
def ocr_tesseract(img, psm=3):
    """psm 3 = segmentación automática (bueno para columnas de periódico)."""
    return pytesseract.image_to_string(img, lang="spa", config=f"--oem 3 --psm {psm}")


def confianza_tesseract(img, psm=3):
    """Confianza media (0-100) de las palabras reconocidas."""
    d = pytesseract.image_to_data(img, lang="spa", config=f"--oem 3 --psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(d["conf"], d["text"]) if float(c) >= 0 and t.strip()]
    return float(np.mean(confs)) if confs else 0.0


# =====================================================================
# 4. LIMPIEZA DE TEXTO
# =====================================================================
def reparar_texto_ocr(texto):
    """Corrige artefactos típicos del OCR en periódicos, conservando
    mayúsculas y puntuación (útil para NER, resumen y clasificador)."""
    t = unicodedata.normalize("NFC", texto)
    t = t.replace("\x0c", "\n")
    t = re.sub(r"[“”«»„]", '"', t)
    t = re.sub(r"[‘’´`]", "'", t)
    t = re.sub(r"[—–]", "-", t)
    # palabras cortadas al final de línea: "presi-\ndente" -> "presidente"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    # basura frecuente: |, ~, _, ^, •, etc.
    t = re.sub(r"[|~_^•■□▪¦\\{}<>]", " ", t)
    # saltos simples dentro de un párrafo -> espacio; dobles = párrafo
    t = re.sub(r"\n{2,}", "<<P>>", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    t = t.replace("<<P>>", "\n\n")
    # falso salto de párrafo: la línea no cerró oración y la siguiente va en minúscula
    t = re.sub(r'([^.!?:"])\s*\n\n\s*(?=[a-záéíóúüñ0-9(])', r"\1 ", t)
    # espacios antes de signos y espacios múltiples
    t = re.sub(r"\s+([,.;:!?)])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    # líneas de 1-2 caracteres (ruido de bordes/fotos)
    lineas = [l.strip() for l in t.split("\n") if len(l.strip()) > 2 or l.strip() == ""]
    return "\n".join(lineas).strip()


def limpiar_texto(texto):
    """Minúsculas + solo letras (con tildes y ñ), números y espacios."""
    t = texto.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)          # números sueltos
    return re.sub(r"\s+", " ", t).strip()


STOPWORDS_EXTRA = {
    "según", "además", "dijo", "indicó", "señaló", "afirmó", "explicó", "precisó",
    "año", "años", "día", "días", "vez", "ser", "haber", "tener", "hacer", "poder",
    "decir", "ir", "así", "solo", "sólo", "mismo", "misma", "pues", "embargo",
    "foto", "difusión", "redacción", "nota", "página",
}


def tokenizar(texto_limpio, nlp, lematizar=True):
    """Tokens sin stopwords, con lematización de spaCy."""
    doc = nlp(texto_limpio)
    toks = []
    for t in doc:
        if not t.is_alpha or t.is_stop or len(t.text) <= 2:
            continue
        palabra = t.lemma_.lower() if lematizar else t.text
        if palabra not in STOPWORDS_EXTRA:
            toks.append(palabra)
    return toks


# =====================================================================
# 5. NLP
# =====================================================================
REGIONES_PERU = [
    "Amazonas", "Áncash", "Apurímac", "Arequipa", "Ayacucho", "Cajamarca", "Callao",
    "Cusco", "Huancavelica", "Huánuco", "Ica", "Junín", "La Libertad", "Lambayeque",
    "Lima", "Loreto", "Madre de Dios", "Moquegua", "Pasco", "Piura", "Puno",
    "San Martín", "Tacna", "Tumbes", "Ucayali",
]


def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def regiones_mencionadas(texto):
    t = _sin_tildes(texto.lower())
    return [r for r in REGIONES_PERU
            if re.search(rf"\b{re.escape(_sin_tildes(r.lower()))}\b", t)]


def extraer_entidades(texto_reparado, nlp):
    """NER de spaCy (PER, LOC, ORG, MISC) + regiones del Perú."""
    doc = nlp(texto_reparado)
    ents = {"PER": Counter(), "LOC": Counter(), "ORG": Counter(), "MISC": Counter()}
    for e in doc.ents:
        txt = e.text.strip()
        if e.label_ in ents and len(txt) > 2 and not txt.isdigit():
            ents[e.label_][txt] += 1
    ents = {k: v.most_common(10) for k, v in ents.items()}
    ents["REGIONES_PERU"] = regiones_mencionadas(texto_reparado)
    return ents


def resumen_extractivo(texto_reparado, nlp, n_oraciones=3):
    """Resumen por frecuencia de palabras: elige las oraciones con mayor
    puntaje y las devuelve en su orden original. Robusto a errores de OCR."""
    doc = nlp(texto_reparado)
    oraciones = [s for s in doc.sents if len([t for t in s if t.is_alpha]) >= 6]
    if len(oraciones) <= n_oraciones:
        return " ".join(s.text.strip() for s in oraciones)
    freq = Counter(t.lemma_.lower() for t in doc
                   if t.is_alpha and not t.is_stop and len(t.text) > 2)
    maxf = max(freq.values()) if freq else 1
    puntajes = []
    for i, s in enumerate(oraciones):
        palabras = [t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop]
        p = sum(freq[w] / maxf for w in palabras) / (len(palabras) ** 0.5 + 1e-9)
        puntajes.append((p, i))
    elegidas = sorted(i for _, i in sorted(puntajes, reverse=True)[:n_oraciones])
    return " ".join(oraciones[i].text.strip() for i in elegidas)


def palabras_clave(texto_tokens_unidos, vectorizador, n=10):
    """Top-n términos por TF-IDF usando el vectorizador entrenado con el corpus."""
    v = vectorizador.transform([texto_tokens_unidos]).toarray()[0]
    vocab = np.array(vectorizador.get_feature_names_out())
    idx = v.argsort()[::-1][:n]
    return [(vocab[i], round(float(v[i]), 3)) for i in idx if v[i] > 0]


# =====================================================================
# 6. MÉTRICAS DE OCR
# =====================================================================
def normalizar_para_metrica(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


def metricas_ocr(referencia, prediccion):
    import jiwer
    r, p = normalizar_para_metrica(referencia), normalizar_para_metrica(prediccion)
    return {"CER": jiwer.cer(r, p), "WER": jiwer.wer(r, p)}
