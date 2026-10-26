# 📰 Lector inteligente de noticias peruanas (OCR + NLP)

Aplicación que recibe la foto de una nota de periódico peruano, la preprocesa con OpenCV,
extrae el texto con Tesseract OCR y lo analiza con NLP: clasificación por sección,
entidades (personas, organizaciones, lugares y regiones del Perú), palabras clave,
resumen automático y nube de palabras.

**App desplegada:** _(pegar enlace de Streamlit Cloud)_

## Estructura
```
├── app.py                  # interfaz Streamlit
├── utils_ocr_nlp.py        # pipeline compartido (preprocesamiento, OCR, limpieza, NLP)
├── modelos/                # clasificador y vectorizador entrenados en el notebook
├── ejemplos/               # imágenes de muestra para probar la app
├── notebooks/              # análisis completo, experimentos y métricas
├── requirements.txt        # dependencias de Python
└── packages.txt            # dependencias del sistema (Tesseract + idioma español)
```

## Ejecutar localmente
```bash
sudo apt install tesseract-ocr tesseract-ocr-spa   # en Windows: instalar Tesseract y agregarlo al PATH
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline
1. **Preprocesamiento:** escala de grises, redimensionamiento, corrección de inclinación,
   eliminación de ruido, eliminación de sombras, CLAHE y binarización (Otsu / adaptativa).
2. **OCR:** Tesseract (`spa`), comparado con EasyOCR mediante CER y WER.
3. **Limpieza:** reparación de guiones y saltos de línea, minúsculas, eliminación de símbolos,
   tokenización, stopwords y lematización con spaCy.
4. **NLP:** TF-IDF + Regresión Logística (comparado con Naive Bayes, SVM y zero-shot),
   NER, palabras clave, resumen extractivo y nube de palabras.

## Integrantes
- Blacido Hernandez, Pamela Brescia
- Chavez Paucar, Carla Andre
- Condor Montes, Paola Judith
