
import re
import unicodedata
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sentence_transformers import SentenceTransformer # <-- ¡Importamos SentenceTransformer de nuevo!
import os # <-- ¡Añadido import os!
import json # <-- Añadido import json para load_corpus

# ================== LISTAS Y UTILIDADES PREVIAS ==================
LEMA_EMOCIONES = {
    "feliz": ["contento", "alegre", "satisfecho", "entusiasmado", "animado"],
    "triste": ["deprimido", "desanimado", "abatido", "melancolico", "infeliz"], # <-- Amplía si quieres
    "ansioso": ["nervioso", "preocupado", "inquieto", "estresado", "agobiado"], # <-- Añadido "agobiado"
    "enojado": ["molesto", "furioso", "irritado", "frustrado"], # <-- Amplía si quieres
    "agradecido": ["gracias", "agradezco", "agradecida", "muy amable"], # <-- Amplía si quieres
    "cansado": ["agotado", "fatigado", "exhausto"], # <-- Añadido
    "neutro": ["normal", "ok", "bien"], # <-- Amplía si quieres
    # ... agrega más emociones si es necesario ...
}
LEMA_INTENCIONES = {
    "expresar_emocion": ["me siento", "estoy", "me noto", "siento", "hoy"], # <-- Amplía si quieres
    "pedir_ayuda": ["ayuda", "necesito", "podrías", "quisiera", "socorro", "auxilio"], # <-- Amplía si quieres
    "agradecer": ["gracias", "agradezco", "muy amable"], # <-- Amplía si quieres
    "saludar": ["hola", "buenos dias", "que tal"], # <-- Añadido
    "despedirse": ["adios", "hasta luego"], # <-- Añadido
    # ... agrega más intenciones si es necesario ...
}
SINONIMOS_EMOCIONES = defaultdict(list)
for k, vals in LEMA_EMOCIONES.items():
    for v in vals:
        SINONIMOS_EMOCIONES[v] = k
SINONIMOS_INTENCIONES = defaultdict(list)
for k, vals in LEMA_INTENCIONES.items():
    for v in vals:
        SINONIMOS_INTENCIONES[v] = k

# =============== Lista de frases críticas (para el Ranker) ===============
# Estas son frases que *si aparecen en una RESPUESTA candidata*, pueden ser features para el ranking.
# ¡OJO! Esto es diferente de los patrones de crisis que activan la respuesta de emergencia.
# Puedes expandir esta lista con frases que, si están en una respuesta del bot, son relevantes para rankear.
CRITICAL_PHRASES = [
    "Entiendo esa sensación", # Frases de validación
    "Es válido sentir",
    "Qué bien que", # Frases de celebración de logros
    "Me alegra escuchar",
    "Puedes contar conmigo", # Frases de apoyo
    "Estoy aquí para ti",
    # ... añade frases que identifiquen tipos de respuesta importantes ...
]


def normalize_text(s):
    """
    Normaliza texto: minúsculas, sin tildes, sin espacios extra.
    """
    s = s.lower()
    s = s.strip()
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s)
    s = re.sub(r"\s+", " ", s).strip() # Añadido .strip()
    return s

def keywords_in_text(keywords, text):
    """
    Devuelve un score (0-1) basado en la cantidad de keywords encontradas en el texto normalizado.
    Simple match de subcadena, no token exacto.
    """
    if not keywords or not text:
        return 0.0
    norm_text = normalize_text(text)
    found = 0
    for kw in keywords:
        if kw and normalize_text(kw) in norm_text: # Normaliza también la keyword a buscar
            found += 1
    # Evitar división por cero si keywords está vacío (aunque ya se chequea arriba)
    return found / len(keywords) if keywords else 0.0


def fuzzy_similarity(a, b):
    """
    Implementación simple de similitud 'fuzzy' (aproximada) basada en palabras compartidas.
    Retorna un score (0-1).
    """
    if not a or not b:
        return 0.0
    a_norm, b_norm = normalize_text(a), normalize_text(b)
    words_a = set(a_norm.split())
    words_b = set(b_norm.split())
    if not words_a and not words_b:
        return 1.0 # Ambos vacíos son 100% similares
    if not words_a or not words_b: # Si uno está vacío pero el otro no
        return 0.0
    shared_words = words_a.intersection(words_b)
    all_words = words_a.union(words_b) # Todas las palabras únicas
    # Score: proporción de palabras compartidas respecto al total de palabras únicas
    # Usar max(1.0, len(all_words)) para evitar división por cero si all_words está vacío (aunque cubierto por chequeos previos)
    return len(shared_words) / len(all_words)


def tokenize_and_lemmatize(text):
    """
    Tokeniza el texto normalizado y reemplaza tokens por lemas/sinónimos conocidos.
    Si un token no es sinónimo conocido, lo mantiene.
    """
    tokens = normalize_text(text).split()
    processed_tokens = []
    for t in tokens:
        if t in SINONIMOS_EMOCIONES:
            processed_tokens.append(SINONIMOS_EMOCIONES[t])
        elif t in SINONIMOS_INTENCIONES:
            processed_tokens.append(SINONIMOS_INTENCIONES[t])
        else:
            processed_tokens.append(t)
    # Aplanar lista de listas si los sinónimos son listas (aunque el dict actual no las usa)
    # El código actual no necesita aplanamiento porque los valores del dict son strings
    # flat_tokens = []
    # for item in processed_tokens:
    #     if isinstance(item, list):
    #         flat_tokens.extend(item)
    #     else:
    #         flat_tokens.append(item)
    # return flat_tokens
    return processed_tokens # Directamente devuelve la lista de tokens procesados


def load_corpus(db_path="safary_emopet_db.json"): # <-- Usa el nombre de base consistente
    """
    Carga texto del db.json para entrenar el vectorizador TF-IDF.
    """
    corpus = []
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for emo, intents in data.items():
                for intent, entries in intents.items():
                    for entry in entries:
                        # Incluye user_input_example y response_options en el corpus
                        if entry.get("user_input_example"):
                            corpus.append(entry["user_input_example"])
                        for resp in entry.get("response_options", []):
                            corpus.append(resp)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARNING] No se pudo cargar {db_path} para el corpus TF-IDF: {e}. Usando corpus base.")
        # Corpus base si el archivo no existe o está vacío/malo
        corpus = ["me siento feliz", "estoy triste", "necesito ayuda", "¿cómo estás?", "hola", "gracias", "estoy ansioso", "mucha gente me agobia"] # <-- Añade ejemplos base relevantes

    return list(set(corpus)) # Retorna solo entradas únicas

# =============== TF-IDF Vectorizer Logic ===============
_tfidf_vectorizer = None
TFIDF_VECTORIZER_PATH = "tfidf_vectorizer.joblib" # Nombre de archivo consistente

def get_or_train_tfidf_vectorizer(db_path="safary_emopet_db.json", force_retrain=False):
    """
    Carga o entrena el vectorizador TF-IDF.
    """
    global _tfidf_vectorizer
    if _tfidf_vectorizer is not None and not force_retrain:
        return _tfidf_vectorizer

    if not force_retrain and os.path.exists(TFIDF_VECTORIZER_PATH):
        try:
            _tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            print("[INFO] Vectorizador TF-IDF cargado.")
            return _tfidf_vectorizer
        except Exception as e:
            print(f"[WARNING] No se pudo cargar el vectorizador TF-IDF: {e}. Re-entrenando.")
            _tfidf_vectorizer = None # Reset para forzar re-entrenamiento

    print("[INFO] Entrenando vectorizador TF-IDF...")
    corpus = load_corpus(db_path)
    # token_pattern=None para usar el tokenizer directamente y evitar warning
    # lowercase=False porque tokenize_and_lemmatize ya hace lowercase
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize, lowercase=False, token_pattern=None)

    if not corpus:
        print("[WARNING] Corpus vacío, no se pudo entrenar el vectorizador TF-IDF.")
        # Crear un vectorizador dummy si no hay corpus
        # Necesita fitting aunque sea con una frase base para tener vocabulario
        vectorizer.fit(["placeholder"]) # Ajusta con un placeholder para inicializar
    else:
        vectorizer.fit(corpus)

    try:
        joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
        print(f"[INFO] Vectorizador TF-IDF entrenado y guardado en {TFIDF_VECTORIZER_PATH}.")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el vectorizador TF-IDF: {e}.")

    _tfidf_vectorizer = vectorizer
    return _tfidf_vectorizer

def get_tfidf_vector(text):
    """
    Devuelve un vector TF-IDF para el texto dado usando el vectorizador entrenado.
    """
    vectorizer = get_or_train_tfidf_vectorizer() # Asegura que esté cargado/entrenado
    vocab_size = len(vectorizer.get_feature_names_out())

    if not text:
        # Retorna un vector de ceros del tamaño del vocabulario si el texto está vacío
        return np.zeros(vocab_size)

    # vectorizer.transform espera una lista de strings
    # Normaliza antes de transformar
    tfidf_matrix = vectorizer.transform([normalize_text(text)])

    # Verifica si la transformación produjo una matriz vacía (podría pasar si el texto solo contiene stopwords o tokens fuera del vocabulario)
    if tfidf_matrix.shape[1] == 0:
         return np.zeros(vocab_size)

    # toarray() convierte la matriz dispersa a un array numpy denso
    # [0] toma la primera (y única) fila del array resultante
    return tfidf_matrix.toarray()[0]

# Entrenar/Cargar vectorizador TF-IDF al inicio (opcional, se hace lazy en get_or_train_tfidf_vectorizer)
# get_or_train_tfidf_vectorizer() # Puedes llamarlo aquí si quieres asegurar que se cargue/entrene al importar el módulo


# =============== Sentence Transformer Embeddings Logic ===============
_sentence_transformer_model = None
# Modelo que estábamos usando (asegúrate que esté instalado: pip install sentence-transformers)
SENTENCE_TRANSFORMER_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# Nota: El modelo ST se descarga automáticamente por la librería la primera vez

def get_sentence_transformer_model():
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        print(f"[INFO] Cargando modelo Sentence Transformer: {SENTENCE_TRANSFORMER_MODEL_NAME}...")
        try:
            _sentence_transformer_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
            print("[INFO] Modelo Sentence Transformer cargado.")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo Sentence Transformer: {e}.")
            print("[ERROR] Las funciones que dependan de embeddings semánticos podrían fallar o retornar ceros.")
            # Considerar retornar un objeto mock o None si falla
            _sentence_transformer_model = None # Asegura que siga siendo None si falla
    return _sentence_transformer_model


def get_sentence_embedding(text):
    """
    Devuelve un embedding semántico para el texto dado usando Sentence Transformer.
    Retorna un vector de ceros si el modelo no se pudo cargar o el texto está vacío.
    """
    model = get_sentence_transformer_model() # Intenta obtener/cargar el modelo

    # Si el modelo no se pudo cargar, retorna ceros
    if model is None:
        # Retorna un vector de ceros con un tamaño estándar si el modelo falló en cargar
        # El tamaño 384 es el de 'paraphrase-multilingual-MiniLM-L12-v2'
        # Es mejor obtener la dimensión si es posible, pero si el modelo es None, no se puede.
        # Usar un tamaño conocido o manejarlo aguas arriba.
        print("[WARNING] Modelo Sentence Transformer no disponible. Retornando vector de ceros.")
        return np.zeros(384) # Tamaño hardcodeado como fallback

    # Si el texto está vacío, retorna ceros con la dimensión correcta del modelo
    if not text:
        return np.zeros(model.get_sentence_embedding_dimension())

    # El modelo encode ya maneja la normalización interna, pero podemos asegurar minúsculas
    # Aunque Sentence Transformers funciona mejor con texto más natural, no rígidamente normalizado
    # encoding = model.encode([normalize_text(text)], convert_to_numpy=True) # Opcional: normalizar
    try:
        # Usualmente mejor sin normalización agresiva para ST
        encoding = model.encode([text], convert_to_numpy=True)
        return encoding[0]
    except Exception as e:
        print(f"[ERROR] Error al generar embedding para el texto: '{text}'. Error: {e}")
        return np.zeros(model.get_sentence_embedding_dimension())


# =============== Función de similitud de Coseno (sirve para ambos tipos de vectores) ===============
def cosine_similarity(vec1, vec2):
    """Calcula la similitud de coseno entre dos vectores numpy."""
    # Asegura que los vectores sean numpy arrays
    vec1 = np.asarray(vec1, dtype=np.float32) # Usar float32 es común para embeddings
    vec2 = np.asarray(vec2, dtype=np.float32)

    # Verifica si los vectores tienen la misma dimensión
    if vec1.shape != vec2.shape:
        # print(f"[WARNING] Intentando calcular similitud de coseno entre vectores de diferente dimensión: {vec1.shape} vs {vec2.shape}")
        # Considerar redimensionar o retornar un score bajo o error
        return 0.0 # Retorna 0 si las dimensiones no coinciden (no son comparables)

    # Verifica si algún vector es un vector de ceros (o cerca de cero)
    # Usa np.linalg.norm para calcular la magnitud (norma L2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Si alguna norma es cero (o muy cercana a cero), la similitud es 0
    # np.finfo(vec1.dtype).eps es la épsilon de la máquina para el tipo de datos del vector
    if norm1 < np.finfo(vec1.dtype).eps or norm2 < np.finfo(vec2.dtype).eps:
        return 0.0

    # Calcula el producto punto
    dot_product = np.dot(vec1, vec2)

    # Calcula la similitud de coseno
    similarity = dot_product / (norm1 * norm2)

    # La similitud de coseno está en el rango [-1, 1]. Para algunos casos, quieres [0, 1].
    # Si quieres [0, 1], puedes hacer: (similarity + 1) / 2
    # Para ranking y búsqueda, [-1, 1] suele ser suficiente, o simplemente el valor tal cual.

    # Asegura que el resultado esté dentro del rango [-1.0, 1.0] debido a posibles errores de punto flotante
    return float(np.clip(similarity, -1.0, 1.0))


# =============== Inicialización del modelo Sentence Transformer al inicio ===============
# Es buena idea llamar a esto una vez cuando el módulo se carga para que el modelo esté listo.
get_sentence_transformer_model()

# =============== Inicialización del vectorizador TF-IDF al inicio ===============
# También es buena idea cargarlo/entrenarlo al inicio.
get_or_train_tfidf_vectorizer()

# Puedes añadir un bloque if __name__ == "__main__": para pruebas si lo deseas
# Ejemplo:
# if __name__ == "__main__":
#     print("Probando normalización:", normalize_text(" ¡Hola, cómo ESTÁS! "))
#     print("Probando keywords:", keywords_in_text(["ayuda", "necesito"], "Necesito tu ayuda urgente"))
#     print("Probando fuzzy similarity:", fuzzy_similarity("me siento feliz hoy", "estoy muy feliz"))
#     print("Probando tokenización/lematización:", tokenize_and_lemmatize("Estoy muy contento y agradecido"))
#
#     # Probar TF-IDF (asegúrate que safary_emopet_db.json exista o se use el corpus base)
#     vector_tfidf = get_tfidf_vector("estoy feliz")
#     print("Vector TF-IDF (primeros 10):", vector_tfidf[:10])
#     print("Dimensión TF-IDF:", len(vector_tfidf))
#
#     # Probar Sentence Transformer
#     vector_st = get_sentence_embedding("estoy feliz")
#     print("Vector ST (primeros 10):", vector_st[:10])
#     print("Dimensión ST:", len(vector_st))
#
#     # Probar similitud coseno
#     vec1 = get_sentence_embedding("hola mundo")
#     vec2 = get_sentence_embedding("hello world")
#     vec3 = get_sentence_embedding("adiós")
#     print(f"Similitud ('hola mundo', 'hello world'): {cosine_similarity(vec1, vec2):.4f}")
#     print(f"Similitud ('hola mundo', 'adiós'): {cosine_similarity(vec1, vec3):.4f}")
