import joblib
import numpy as np
from core import (
    normalize_text, keywords_in_text, fuzzy_similarity, CRITICAL_PHRASES, get_sentence_embedding, cosine_similarity
)

def extract_features(user_text, candidate_response, emotion, intention, context_length,
                    user_input_example=None, search_score=None, user_history=None, input_keywords=None):
    user_norm = normalize_text(user_text)
    cand_norm = normalize_text(candidate_response)
    f = []
    f.append(len(user_norm))
    f.append(len(cand_norm))
    user_words = set(user_norm.split())
    cand_words = set(cand_norm.split())
    shared = user_words & cand_words
    f.append(len(shared))
    f.append(len(shared) / max(1, len(user_words)))
    f.append(hash(emotion) % 1000)
    f.append(hash(intention) % 1000)
    f.append(float(search_score) if search_score is not None else 0.0)
    if user_input_example:
        f.append(fuzzy_similarity(user_norm, normalize_text(user_input_example)))
    else:
        f.append(0.0)
    if input_keywords and isinstance(input_keywords, list) and input_keywords:
        f.append(keywords_in_text(input_keywords, cand_norm))
    elif user_input_example:
        fallback_keywords = normalize_text(user_input_example).split()
        f.append(keywords_in_text(fallback_keywords, cand_norm))
    else:
        # Antes: f.append(keywords_in_text(user_norm.split(), cand_norm))
        # Ahora: feature es 0 si no hay keywords útiles ni ejemplo
        f.append(0)
    hist_score = 0
    if user_history:
        for h in user_history:
            hist_score += keywords_in_text(user_norm.split(), normalize_text(h.get("text", "")))
    f.append(hist_score)
    crit_score = 0
    for phrase in CRITICAL_PHRASES:
        if phrase in cand_norm:
            crit_score += 1
    f.append(crit_score)
    # Feature de similitud semántica embeddings
    user_emb = get_sentence_embedding(user_text)
    cand_emb = get_sentence_embedding(candidate_response)
    sem_sim = cosine_similarity(user_emb, cand_emb)
    f.append(sem_sim)
    return np.array(f)

class ResponseRanker:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=20)
            self.is_trained = False
        else:
            self.is_trained = True

    def fit_and_save(self, feedback):
        X, y = [], []
        for entry in feedback:
            X.append(extract_features(
                entry["user_text"], entry["candidate_response"],
                entry.get("emotion", "neutro"), entry.get("intention", "expresar_emocion"),
                entry.get("context_length", 1),
                user_input_example=entry.get("user_input_example"),
                search_score=entry.get("search_score"),
                user_history=entry.get("user_history"),
                input_keywords=entry.get("input_keywords")
            ))
            y.append(1 if entry.get("was_helpful") else 0)
        if X:
            self.model.fit(X, y)
            joblib.dump(self.model, "ranking_model.joblib")
            self.is_trained = True

    def predict_score(self, user_text, candidate_response, emotion, intention, context_length,
                      user_input_example=None, search_score=None, user_history=None, input_keywords=None):
        feats = extract_features(
            user_text, candidate_response, emotion, intention, context_length,
            user_input_example=user_input_example, search_score=search_score,
            user_history=user_history, input_keywords=input_keywords
        ).reshape(1, -1)
        try:
            return float(self.model.predict(feats)[0])
        except Exception:
            sem_sim = cosine_similarity(get_sentence_embedding(user_text), get_sentence_embedding(candidate_response))
            return (
                0.3 * fuzzy_similarity(normalize_text(user_text), normalize_text(candidate_response))
                + 0.3 * (float(search_score) if search_score is not None else 0.0)
                + 0.4 * sem_sim
            )