from collections import deque
import json
import os

class UserHistory:
    """
    Maneja el historial de interacciones del usuario con carga/guardado.
    Permite recency_weight para contexto en clasificación.
    """
    def __init__(self, filename="user_history.json", maxlen=20):
        self.filename = filename
        self.maxlen = maxlen
        self.history = deque(maxlen=maxlen)
        self.load()

    def add(self, emotion=None, intention=None, text=None, meta=None):
        """
        Agrega una nueva entrada de conversación.
        meta puede ser un dict con info extra.
        """
        self.history.append({
            "emotion": emotion,
            "intention": intention,
            "text": text,
            "meta": meta or {}
        })

    def load(self):
        """
        Carga el historial desde el archivo JSON.
        """
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                    self.history = deque(loaded_history, maxlen=self.maxlen)
            except (json.JSONDecodeError, IOError):
                print(f"[ERROR] No se pudo cargar el historial desde {self.filename}. Iniciando vacío.")
                self.history = deque(maxlen=self.maxlen)

    def save(self):
        """
        Guarda el historial actual en un archivo JSON.
        """
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(list(self.history), f, ensure_ascii=False, indent=2)
        except Exception:
            print(f"[ERROR] No se pudo guardar el historial en {self.filename}.")

    def recency_weight(self, target_emotion=None, target_intention=None, window=5):
        """
        Devuelve un score (0-1) basado en cuán frecuente/relevante ha sido la emoción o intención objetivo
        en las últimas 'window' interacciones del historial.
        """
        if not self.history or (target_emotion is None and target_intention is None):
            return 0.0
        recent = list(self.history)[-window:]
        score = 0
        count = 0
        for h in recent:
            if target_emotion and h.get("emotion") == target_emotion:
                score += 1
            if target_intention and h.get("intention") == target_intention:
                score += 1
            count += 1
        if count == 0:
            return 0.0
        normalizer = window if (target_emotion is None or target_intention is None) else 2*window
        return score / normalizer

    def get_last(self, n=1):
        """
        Devuelve las últimas n entradas del historial.
        """
        if n <= 0:
            return []
        return list(self.history)[-n:]

    def __len__(self):
        return len(self.history)

    def __iter__(self):
        return iter(self.history)

    def clear(self):
        self.history.clear()

    