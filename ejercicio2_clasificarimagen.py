import cv2  # Biblioteca OpenCV para procesamiento de imágenes.
import pytesseract  # Biblioteca para OCR (reconocimiento óptico de caracteres).
from PIL import Image  # Biblioteca Pillow para manipulación de imágenes.
import pdf2image  # Biblioteca para convertir archivos PDF en imágenes.
import numpy as np  # Biblioteca NumPy para manipulación de matrices y cálculos numéricos.
import os  # Biblioteca estándar de Python para manipulación de rutas y archivos.
from sklearn.feature_extraction.text import CountVectorizer  # Herramienta de Scikit-learn para convertir texto en vectores.
from sklearn.naive_bayes import MultinomialNB  # Modelo Naive Bayes multinomial de Scikit-learn.
from sklearn.pipeline import make_pipeline  # Herramienta de Scikit-learn para crear pipelines de procesamiento.
from sklearn.model_selection import train_test_split  # Función de Scikit-learn para dividir datos en entrenamiento y prueba.
import matplotlib.pyplot as plt  # Biblioteca Matplotlib para visualización de datos.
import csv # Bibliotec estandar de python para trabajar con archivos csv.
import re  # Biblioteca para trabajar con expresiones regulares.

# Configuración de la ruta de Tesseract OCR.
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Función para convertir archivos PDF en imágenes.
def pdf_a_imagenes(ruta_pdf):
    """
    Convierte un archivo PDF en una lista de imágenes.
    Origen: pdf2image
    """
    try:
        imagenes = pdf2image.convert_from_path(ruta_pdf, poppler_path='C:/Poppler/poppler-24.08.0/Library/bin')
        return imagenes
    except Exception as e:
        print(f"Error al convertir PDF a imágenes: {e}")
        return []

# Función para preprocesar imágenes.
def preprocesar_imagen(imagen):
    try:
        if isinstance(imagen, Image.Image):
            imagen_np = np.array(imagen)
        else:
            imagen_np = imagen

        imagen_gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
        imagen_contraste = cv2.convertScaleAbs(imagen_gris, alpha=1.5, beta=0)
        imagen_suavizada = cv2.GaussianBlur(imagen_contraste, (5, 5), 0)
        _, imagen_binaria = cv2.threshold(imagen_suavizada, 128, 255, cv2.THRESH_BINARY)
        return imagen_binaria
    except Exception as e:
        print(f"Error al preprocesar la imagen: {e}")
        return None

# Función para extraer texto de una imagen usando OCR.
def extraer_texto(imagen):
    """
    Extrae texto de una imagen usando Tesseract OCR.
    Origen: pytesseract.
    """
    try:
        texto = pytesseract.image_to_string(imagen, lang='spa')  # Extrae texto en español.
        return texto
    except Exception as e:
        print(f"Error al extraer texto de la imagen: {e}")
        return ""

def extraer_numero_documento(texto):
    """
    Busca y extrae el número de documento de identidad en el texto.
    El número puede estar separado por puntos o no.
    """
    try:
        # Buscar el texto "Número de documento de identidad" seguido de un número.
        patron = r"(?i)identidad.*?(\d[\d\.]*)"
        coincidencia = re.search(patron, texto)
        if coincidencia:
            # Eliminar los puntos del número para normalizarlo.
            numero_documento = coincidencia.group(1).replace(".", "")
            return numero_documento
        else:
            return "No encontrado"
    except Exception as e:
        print(f"Error al extraer el número de documento: {e}")
        return "Error"

import pytesseract

def extraer_numero_documento_por_coordenadas(imagen):
    """
    Busca el número de documento de identidad o identificación en una región específica de la imagen.
    """
    try:
        # Obtener datos detallados del texto en la imagen
        datos = pytesseract.image_to_data(imagen, lang='spa', output_type=pytesseract.Output.DICT)

        # Palabras clave a buscar
        palabras_clave = ["Identidad", "Identificación"]

        # Buscar la posición de las palabras clave
        for i, palabra in enumerate(datos['text']):
            if any(clave in palabra.lower() for clave in palabras_clave):  # Verifica si la palabra coincide con alguna clave
                # Coordenadas de la palabra encontrada
                x, y, ancho, alto = datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i]
                print(f"Palabra clave encontrada en coordenadas: ({x}, {y}, {ancho}, {alto})")

                # Definir una región cercana para buscar el número
                margen_x = 50  # Aumentar el margen horizontal
                margen_y = 100  # Aumentar el margen vertical

                x_inicio = max(0, x - margen_x)
                y_inicio = max(0, y + alto)  # Buscar debajo de la palabra clave
                x_fin = x + ancho + margen_x
                y_fin = y + alto + margen_y

                # Recortar la región de interés (ROI) de la imagen
                roi = imagen[y_inicio:y_fin, x_inicio:x_fin]

                # Extraer texto de la región recortada
                texto_roi = pytesseract.image_to_string(roi, lang='spa')
                print(f"Texto en la región de interés:\n{texto_roi}")

                # Buscar el número en el texto de la región
                patron = r"(\d[\d\.]*)"
                coincidencia = re.search(patron, texto_roi)
                if coincidencia:
                    numero_documento = coincidencia.group(1).replace(".", "")
                    return numero_documento

        return "No encontrado"
    except Exception as e:
        print(f"Error al extraer el número de documento: {e}")
        return "Error"

# Ruta de los datos de entrenamiento.
ruta_imagenes = 'C:/Documentos/Proyectos/Desarrollo/dataentrenamiento'

# Lista de archivos PDF para entrenamiento.
imagenes = [
    'F1006011280.pdf', 'F1006464727.pdf', 'C1060357877.pdf', 'F1105366125.pdf', 'F1110287795.pdf',
    'F1112466914.pdf', 'C1006011280.pdf', 'C1006464727.pdf', 'CC1105366125.pdf',
    'F1060357877.pdf', 'O1113654813-3.pdf', 'F1113654813-1.pdf', 'F1116258605.pdf',
    'F1116917375.pdf', 'F1130615179.pdf', 'F1130678621.pdf', 'RC16845155-5.pdf',
    'RC16845155-6.pdf','TI16845155-3.pdf','TI16845155-4.pdf','CC16456814.pdf',
    'CC19326390(2).pdf','CC31474193(1-2).pdf','CC1112483642(2).pdf','CC1112483642(3).pdf',
    'CC16841797.pdf','C16456814.pdf','F16456814.pdf','C6333219.pdf','F6333219.pdf',
    'C5605304.pdf','F5605304.pdf'
    
]

# Etiquetas correspondientes a los archivos PDF.
etiquetas = ['formulario', 'formulario', 'encuesta', 'formulario',
             'formulario', 'formulario', 'encuesta',
             'encuesta', 'cedula', 'formulario', 'otros', 'formulario',
             'formulario', 'formulario', 'formulario', 'formulario','registrocivil',
             'registrocivil','tarjetaidentidad','tarjetaidentidad','cedula',
             'cedula','cedula','cedula','cedula','cedula','encuesta',
             'formulario','encuesta','formulario','encuesta','formulario'
]

# Procesar las imágenes y extraer el texto
textos = []
etiquetas_validas = []
for archivo, etiqueta in zip(imagenes, etiquetas):
    """
    Itera sobre los archivos PDF y sus etiquetas.
    """
    ruta_pdf = os.path.join(ruta_imagenes, archivo)  # Construye la ruta completa del archivo PDF.
    imagenes_pdf = pdf_a_imagenes(ruta_pdf)  # Convierte el PDF en una lista de imágenes.

    for imagen in imagenes_pdf:
        imagen_procesada = preprocesar_imagen(imagen)  # Preprocesa cada imagen.
        if imagen_procesada is None:
            continue

        texto = extraer_texto(imagen_procesada)  # Extrae el texto de la imagen preprocesada.
        if texto.strip() == "":
            continue

        textos.append(texto)  # Añade el texto extraído a la lista.
        etiquetas_validas.append(etiqueta)  # Añade la etiqueta correspondiente.

# Verificar que las listas tengan la misma longitud.
if len(textos) != len(etiquetas_validas):
    print(f"Error: El número de textos ({len(textos)}) no coincide con el número de etiquetas ({len(etiquetas_validas)}).")
else:
    # Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(textos, etiquetas_validas, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo.
    model = make_pipeline(CountVectorizer(), MultinomialNB())  # Pipeline con vectorizador y modelo Naive Bayes.
    model.fit(X_train, y_train)  # Entrena el modelo.

    # Evaluar el modelo.
    accuracy = model.score(X_test, y_test)  # Calcula la precisión del modelo.
    print(f'Precisión del modelo: {accuracy:.2f}')

    # Clasificar un conjunto de archivos en un directorio.
    ruta_directorio = input("Ingrese la ruta del directorio que contiene los documentos o imágenes a clasificar: ")
    archivos_pdf = [f for f in os.listdir(ruta_directorio) if f.endswith('.pdf')]  # Lista de archivos PDF en el directorio.

    # Diccionario para almacenar los resultados de la clasificación.
    resultados_clasificacion = {}

   # Lista para almacenar los datos que se escribirán en el archivo CSV.
datos_csv = [["Archivo", "Ruta", "Clasificación", "Número de Documento", "Texto Extraído"]]  # Encabezados del archivo CSV.

for archivo_pdf in archivos_pdf:
    ruta_pdf = os.path.join(ruta_directorio, archivo_pdf)
    imagenes = pdf_a_imagenes(ruta_pdf)

    for imagen in imagenes:
        imagen_procesada = preprocesar_imagen(imagen)  # Preprocesa la nueva imagen.
        if imagen_procesada is None:
            continue

        texto = extraer_texto(imagen_procesada)  # Extrae el texto de la imagen preprocesada.
        print("Procesando imagen...")
        print(f"Texto extraído:\n{texto}")
        prediccion = model.predict([texto])[0]  # Clasifica el texto extraído.
        print(f'El archivo {archivo_pdf} ha sido clasificado como: {prediccion}')

        # Extraer el número de documento usando coordenadas
        numero_documento = extraer_numero_documento_por_coordenadas(imagen_procesada)
        print(f'Número de documento extraído: {numero_documento}')

        # Agregar los datos al arreglo para el archivo CSV
        datos_csv.append([archivo_pdf, ruta_pdf, prediccion, numero_documento, texto])

# Guardar los resultados en un archivo CSV.
ruta_csv = os.path.join(ruta_directorio, "resultados_clasificacion.csv")
with open(ruta_csv, mode="w", newline="", encoding="utf-8") as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    escritor_csv.writerows(datos_csv)

print(f"Resultados de la clasificación guardados en: {ruta_csv}")

 # Graficar los resultados de la clasificación.
etiquetas = list(resultados_clasificacion.keys())  # Etiquetas únicas.
cantidades = list(resultados_clasificacion.values())  # Cantidades por etiqueta.

    # Definir colores para cada barra.
colores = plt.colormaps['tab10'](range(len(etiquetas)))

plt.bar(etiquetas, cantidades, color=colores)  # Crear gráfica de barras.
plt.xlabel('Etiquetas')  # Etiqueta del eje X.
plt.ylabel('Cantidad de documentos clasificados')  # Etiqueta del eje Y.
plt.title('Resultados de la clasificación de documentos')  # Título de la gráfica.
plt.show()  # Mostrar la gráfica.