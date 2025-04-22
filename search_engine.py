import json
import os
import difflib
import numpy as np
from core import get_sentence_embedding, cosine_similarity

class SearchEngine:
    def __init__(self, db_path):
        self.db_path = db_path
        self.data = self.load_data(db_path)

    def load_data(self, path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_candidates(self, emotion, intent):
        return self.data.get(emotion, {}).get(intent, [])

    def _to_np_embedding(self, emb):
        """
        Convierte cualquier embedding (list, tuple, np.ndarray, None) a un np.ndarray.
        Si emb es None o no convertible, retorna un vector de ceros.
        """
        if emb is None:
            return np.zeros(384)
        if isinstance(emb, np.ndarray):
            return emb
        try:
            arr = np.array(emb, dtype=float)
            # Si el modelo cambia el tamaño, ajusta aquí.
            if arr.shape[0] < 384:
                arr = np.pad(arr, (0, 384 - arr.shape[0]))
            return arr
        except Exception:
            return np.zeros(384)

    def search(self, user_input, user_emotion, user_intent, top_k=10):
        """
        Busca una respuesta en la base, usando bucket de emoción/intención,
        similitud textual y similitud semántica con embeddings.
        """
        user_input_emb = get_sentence_embedding(user_input)
        user_input_emb = self._to_np_embedding(user_input_emb)

        candidates = self._get_candidates(user_emotion, user_intent)

        if not candidates:
            related_buckets = {
                "cansado": ["neutro", "triste", "estresado"],
                "estresado": ["neutro", "cansado", "triste"],
                "triste": ["neutro", "cansado"],
                "neutro": ["triste", "cansado"],
                "feliz": ["neutro"],
            }
            alternates = related_buckets.get(user_emotion, ["neutro"])
            for alt_emotion in alternates:
                candidates = self._get_candidates(alt_emotion, user_intent)
                if candidates:
                    break

        if not candidates:
            candidates = []
            for emo in self.data:
                candidates.extend(self._get_candidates(emo, user_intent))

        if not candidates:
            for emo in self.data:
                for intent in self.data[emo]:
                    candidates.extend(self.data[emo][intent])

        scored = []
        for entry in candidates:
            example = entry.get("user_input_example", "")
            saved_embedding = entry.get("user_input_embedding", None)

            text_sim = difflib.SequenceMatcher(None, user_input.lower(), example.lower()).ratio()

            # Similitud semántica robusta
            sem_sim = 0.0
            np_saved_emb = self._to_np_embedding(saved_embedding)
            if np_saved_emb is not None and np.linalg.norm(np_saved_emb) > 0:
                sem_sim = cosine_similarity(user_input_emb, np_saved_emb)

            emotion_penalty = 0.0
            if entry.get("emotion") and entry.get("emotion") != user_emotion:
                emotion_penalty = -0.15

            score = (0.4 * text_sim) + (0.5 * sem_sim) + emotion_penalty

            scored.append({"entry": entry, "score": score})

        scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        return scored

    def get_all_by_emotion(self, emotion):
        return self.data.get(emotion, {})

    def get_all_by_intent(self, intent):
        result = []
        for emo in self.data:
            result.extend(self.data[emo].get(intent, []))
        return result

    def get_all_examples(self):
        examples = []
        for emo in self.data:
            for intent in self.data[emo]:
                for block in self.data[emo][intent]:
                    examples.append(block.get("user_input_example", ""))
        return examples

    def reload(self):
        self.data = self.load_data(self.db_path)