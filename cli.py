import argparse
from utils import get_generic_fallback_response, call_azure_api
from classifier import EmotionIntentionClassifier
from search_engine import SearchEngine
from history import UserHistory
from ranking import ResponseRanker
from integrator import feed_from_pasted_conversation, merge_db

# Configuración global
DB_PATH = "safary_emopet_db.json"
EMERGENCY_RESPONSE = (
    "Entiendo que te sientes muy mal. Estoy aquí para escucharte. "
    "Si necesitas ayuda inmediata, por favor contacta a un profesional o línea de crisis. "
    "Chile: *Fono Salud Responde* 600 360 7777 | *Fono *4141 | "
    
)

def parse_args():
    parser = argparse.ArgumentParser(description="Safary Emopet CLI")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Ruta del archivo de base de datos.")
    parser.add_argument("--azure", action="store_true", help="Usar API externa para fallback.")
    parser.add_argument("--verbose", action="store_true", help="Mostrar información de depuración.")
    return parser.parse_args()

def alimentar_command(search_engine, db_path):
    print("Pega la conversación a alimentar (doble enter para terminar):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    conv_text = "\n".join(lines)
    feed_from_pasted_conversation(conv_text, db_path)
    search_engine.reload()
    print("¡Conversación alimentada y base recargada!")

def curar_command(search_engine, db_path):
    print("Funcionalidad de curación aún no implementada completamente.")
    # Aquí podrías implementar tu lógica de curación

def emopet_chat(user_input, user_history, search_engine, classifier, ranker, args):
    # 1. Clasificación de emoción e intención (incluye detección de crisis interna)
    emotion, intention = classifier.classify(user_input, user_history=user_history)

    # 2. Manejo de crisis (basado en el resultado de la clasificación)
    if emotion == "crisis" and intention == "pedir_ayuda":
        print(EMERGENCY_RESPONSE)
        if args.verbose:
            print(f"[DEBUG] Detección de Crisis activada por clasificador.")
        user_history.add(emotion, intention, user_input, meta={"crisis": True})
        user_history.save()
        return

    if args.verbose:
        print(f"[DEBUG] Emoción: {emotion} | Intención: {intention}")

    # 3. Búsqueda de candidatos (solo si no fue crisis)
    candidates = search_engine.search(
        user_input, user_emotion=emotion, user_intent=intention, top_k=10
    )

    if args.verbose:
        print(f"[DEBUG] {len(candidates)} candidatos encontrados.")

    # 4. Ranking de candidatos (si hay)
    if candidates:
        ranked = []
        for cand in candidates:
            entry = cand["entry"]
            score = cand["score"]
            block = {
                "user_input_example": entry.get("user_input_example", ""),
                "candidate_response": entry.get("response_options", [""])[0],
                "emotion": emotion,
                "intention": intention,
                "context_length": len(user_history),
                "search_score": score,
                "user_history": user_history,
                "input_keywords": entry.get("input_keywords", [])
            }
            # Usa el ranker para calcular el score final
            final_score = ranker.predict_score(
                user_input,
                block["candidate_response"],
                emotion,
                intention,
                block["context_length"],
                block["user_input_example"],
                score,
                user_history,
                block["input_keywords"]
            )
            ranked.append((final_score, entry))
        ranked.sort(reverse=True, key=lambda x: x[0])
        best = ranked[0][1]
        response = best["response_options"][0] if best.get("response_options") else get_generic_fallback_response()
        if args.verbose:
            print(f"[DEBUG] Score rankeado: {ranked[0][0]:.2f} | Ejemplo base: '{best.get('user_input_example','')}'")
    else:
        # Fallback: API externa o respuesta genérica
        if args.azure:
            response = call_azure_api(user_input)
        else:
            response = get_generic_fallback_response()
        if args.verbose:
            print("[DEBUG] Fallback activado.")

    print(response)
    user_history.add(emotion, intention, user_input)
    user_history.save()

def main():
    args = parse_args()
    print("Safary Emopet v2.0 - ¡Bienvenido!")
    search_engine = SearchEngine(args.db)
    classifier = EmotionIntentionClassifier()
    ranker = ResponseRanker(model_path="ranking_model.joblib")
    user_history = UserHistory(filename="user_history.json", maxlen=20)

    print("Escribe tu mensaje (Ctrl+C para salir):")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower().startswith("/alimentar"):
                alimentar_command(search_engine, args.db)
                continue
            if user_input.lower().startswith("/curar"):
                curar_command(search_engine, args.db)
                continue
            emopet_chat(user_input, user_history, search_engine, classifier, ranker, args)
    except (KeyboardInterrupt, EOFError):
        print("\n¡Hasta luego!")
        user_history.save()

if __name__ == "__main__":
    main()