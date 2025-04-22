from core import (
    normalize_text,
    tokenize_and_lemmatize,
    LEMA_EMOCIONES,
    LEMA_INTENCIONES,
    keywords_in_text,
    get_sentence_embedding,
    cosine_similarity,
)
import re

# =============== Palabras y patrones críticos para crisis ===============
CRISIS_PATTERNS = [
    r"\b(suicidio|suicidarme|quitarme la vida|me quiero morir|no quiero vivir|no quiero estar aqui|quiero desaparecer|no aguanto más|no aguanto mas|todo terminará|todo terminara|no le encuentro sentido|no encuentro sentido|me rindo|no puedo más|no puedo mas|no soporto más|no soporto mas|necesito ayuda urgente|emergencia)\b",
    r"\b(acabar con todo|terminar con todo|estoy desesperad[ao]|estoy al límite|estoy al limite|quiero rendirme)\b",
]

CRISIS_KEYWORDS = [
    "suicidio", "suicidarme", "quitarme la vida", "me quiero morir", "no quiero vivir", "no quiero estar aqui",
    "quiero desaparecer", "no aguanto más", "no aguanto mas", "todo terminará", "todo terminara", "no le encuentro sentido",
    "no encuentro sentido", "me rindo", "no puedo más", "no puedo mas", "no soporto más", "no soporto mas",
    "necesito ayuda urgente", "emergencia", "acabar con todo", "terminar con todo", "estoy desesperado", "estoy desesperada",
    "estoy al límite", "estoy al limite", "quiero rendirme"
]

EMOTION_PROTOTYPES = {
    emo: get_sentence_embedding(" ".join([emo] + LEMA_EMOCIONES[emo]))
    for emo in LEMA_EMOCIONES
}
INTENTION_PROTOTYPES = {
    intent: get_sentence_embedding(" ".join([intent] + LEMA_INTENCIONES[intent]))
    for intent in LEMA_INTENCIONES
}

class EmotionIntentionClassifier:
    """
    Clasificador robusto basado en scoring: combina keywords del core, contexto y embeddings.
    Incluye heurística explícita para detección de crisis y fallback seguro.
    """
    def __init__(
        self,
        context_window=5,
        keywords_weight=1.0,
        context_weight=0.7,
        embedding_weight=1.0,
        crisis_patterns=CRISIS_PATTERNS,
        crisis_keywords=CRISIS_KEYWORDS,
    ):
        self.context_window = context_window
        self.keywords_weight = keywords_weight
        self.context_weight = context_weight
        self.embedding_weight = embedding_weight
        self.crisis_patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in crisis_patterns]
        self.crisis_keywords = set(crisis_keywords)

    def detect_crisis(self, user_text):
        """
        Detecta crisis usando patrones regex y keywords robustos.
        Retorna True si se detecta crisis, False si no.
        """
        text_lower = user_text.lower()
        for pat in self.crisis_patterns:
            if pat.search(text_lower):
                return True
        for kw in self.crisis_keywords:
            if kw in text_lower:
                return True
        return False

    def _score_emotion(self, user_text, user_history=None):
        norm_text = normalize_text(user_text)
        scores = {}
        user_emb = get_sentence_embedding(user_text)

        for emo in LEMA_EMOCIONES:
            keywords = [emo] + LEMA_EMOCIONES[emo]
            keyword_score = keywords_in_text(keywords, norm_text)
            context_score = 0
            if user_history and hasattr(user_history, "recency_weight"):
                context_score = user_history.recency_weight(target_emotion=emo, window=self.context_window)
            prototype_emb = EMOTION_PROTOTYPES[emo]
            embedding_score = cosine_similarity(user_emb, prototype_emb)
            total = (
                self.keywords_weight * keyword_score
                + self.context_weight * context_score
                + self.embedding_weight * embedding_score
            )
            scores[emo] = total
        best_emo = max(scores, key=scores.get)
        return best_emo, scores

    def _score_intention(self, user_text, user_history=None):
        norm_text = normalize_text(user_text)
        scores = {}
        user_emb = get_sentence_embedding(user_text)

        for intent in LEMA_INTENCIONES:
            keywords = [intent] + LEMA_INTENCIONES[intent]
            keyword_score = keywords_in_text(keywords, norm_text)
            context_score = 0
            if user_history and hasattr(user_history, "recency_weight"):
                context_score = user_history.recency_weight(target_intention=intent, window=self.context_window)
            prototype_emb = INTENTION_PROTOTYPES[intent]
            embedding_score = cosine_similarity(user_emb, prototype_emb)
            total = (
                self.keywords_weight * keyword_score
                + self.context_weight * context_score
                + self.embedding_weight * embedding_score
            )
            scores[intent] = total
        best_intent = max(scores, key=scores.get)
        return best_intent, scores

    def classify(self, user_text, user_history=None):
        """
        Devuelve (emoción, intención) inferidos para el texto, usando scoring robusto
        que combina keywords, contexto y similitud semántica (embeddings).
        Prioriza la detección de crisis: si detecta crisis, retorna ('crisis', 'pedir_ayuda').
        """
        if self.detect_crisis(user_text):
            return ("crisis", "pedir_ayuda")
        best_emo, emo_scores = self._score_emotion(user_text, user_history)
        best_intent, intent_scores = self._score_intention(user_text, user_history)
        return (best_emo, best_intent)