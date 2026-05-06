from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_from_directory

from smart_parking.nlp_model import EntityExtractor, IntentClassifier
from smart_parking.parking_engine import ParkingEngine, VEHICLE_LABELS


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder="static", static_url_path="")
engine = ParkingEngine(BASE_DIR / "data" / "parking_lots.json")
classifier = IntentClassifier()
extractor = EntityExtractor(engine.destinations, engine.lots)

sessions: dict[str, dict] = defaultdict(
    lambda: {
        "vehicle_type": None,
        "location": None,
        "last_results": [],
        "last_selected_lot": None,
        "last_preferences": [],
        "last_budget": None,
    }
)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "LPU Parking Assistant API",
            "frontend": "React",
            "backend": "Flask",
            "model": "TF-IDF + Linear SVM intents, Random Forest availability",
        }
    )


@app.get("/api/lots")
def lots():
    vehicle_type = request.args.get("vehicle", "car")
    return jsonify({"lots": engine.list_lots(vehicle_type)})


@app.get("/api/model")
def model_info():
    return jsonify(
        {
            "name": "LPU Parking Assistant",
            "intent_model": "TF-IDF n-grams + Linear SVM",
            "availability_model": "Random Forest demand estimator",
            "ranking": "availability, walking distance, price, facilities, and area match",
            "vehicles": VEHICLE_LABELS,
            "locations": list(engine.destinations.keys()),
        }
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    session_id = str(payload.get("session_id") or uuid4())

    if not message:
        return jsonify(
            _response(
                session_id,
                "Hlo, I am ready. Tell me where in LPU you are heading and I will find available parking.",
                quick_replies=_location_replies(),
            )
        )

    state = sessions[session_id]
    intent = classifier.predict(message)
    entities = extractor.extract(message)
    intent = _override_intent(message, intent, entities)
    result = _handle_message(message, state, intent, entities)
    result.update(
        {
            "session_id": session_id,
            "intent": intent.name,
            "confidence": intent.confidence,
            "top_intents": intent.top,
            "entities": entities,
            "memory": {
                "vehicle_type": state.get("vehicle_type"),
                "location": state.get("location"),
            },
        }
    )
    return jsonify(result)


@app.post("/api/reserve")
def reserve_from_card():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or uuid4())
    state = sessions[session_id]
    lot_id = payload.get("lot_id") or state.get("last_selected_lot")
    vehicle_type = payload.get("vehicle_type") or state.get("vehicle_type")
    duration = int(payload.get("duration_minutes") or 45)

    if not vehicle_type:
        return jsonify({"ok": False, "reply": "Tell me the vehicle type first."}), 400

    try:
        reservation = engine.reserve(lot_id, vehicle_type, duration)
    except (TypeError, ValueError) as error:
        return jsonify({"ok": False, "reply": str(error)}), 400

    state["last_selected_lot"] = reservation["lot_id"]
    return jsonify(
        {
            "ok": True,
            "reply": _reservation_reply(reservation),
            "reservation": reservation,
            "cards": [],
            "quick_replies": [],
        }
    )


@app.post("/api/refresh")
def refresh_cards():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or "")
    state = sessions.get(session_id)
    if not state or not state.get("location") or not state.get("vehicle_type") or not state.get("last_results"):
        return jsonify({"cards": [], "updated": False})

    cards = engine.recommend(
        location=state.get("location"),
        vehicle_type=state.get("vehicle_type"),
        preferences=state.get("last_preferences") or [],
        budget_per_hour=state.get("last_budget"),
        limit=3,
    )
    state["last_results"] = cards
    state["last_selected_lot"] = cards[0]["id"] if cards else state.get("last_selected_lot")
    return jsonify(
        {
            "cards": cards,
            "updated": True,
            "memory": {
                "vehicle_type": state.get("vehicle_type"),
                "location": state.get("location"),
            },
        }
    )


def _handle_message(message: str, state: dict, intent, entities: dict) -> dict:
    special = _special_choice_flow(message, state)
    if special:
        return special

    knowledge = _knowledge_flow(message, state)
    if knowledge:
        return knowledge

    _remember_entities(state, entities)

    if _is_ambiguous_hostel(message, entities):
        return _response(
            None,
            "Which hostel area do you mean?",
            quick_replies=["Boys Hostel", "Girls Hostel"],
        )

    if intent.name == "greeting":
        if state.get("location") and state.get("vehicle_type"):
            return _search_flow(state, entities, [])
        return _response(
            None,
            "Hlo, I am ready. Tell me where in LPU you are heading and I will find the best live parking option.",
            quick_replies=_location_replies(),
        )

    if intent.name == "thanks":
        return _response(None, "Anytime. I can hold one of these options if you need it.", cards=state.get("last_results", []), quick_replies=_after_results_replies())

    if intent.name == "cancel_reservation":
        return _cancel_flow(entities)

    if intent.name == "reserve_spot":
        return _reserve_flow(state, entities)

    if intent.name == "lot_details":
        return _details_flow(state, entities)

    if intent.name == "navigation":
        return _navigation_flow(state, entities)

    if not entities and not state.get("location") and intent.confidence < 0.28:
        return _response(
            None,
            "I did not catch that clearly. You can ask for LPU parking or ask a short AI concept question for the project demo.",
            quick_replies=["Find parking", "What is ensemble learning?", "How this model works", "More areas"],
        )

    preferences = list(entities.get("preferences", []))
    if intent.name == "price_query" and "cheapest" not in preferences:
        preferences.append("cheapest")

    return _parking_flow(state, entities, preferences)


def _special_choice_flow(message: str, state: dict) -> dict | None:
    lowered = message.lower().strip()
    if lowered in {"more areas", "show more areas", "more places", "other areas", "more locations", "all areas", "show all areas"}:
        return _response(
            None,
            "Here are more LPU areas I can check. You can also type any campus place directly.",
            quick_replies=_more_location_replies(),
        )
    if lowered in {"find parking", "parking", "available parking", "find available parking", "search parking"}:
        return _response(None, "Sure. Which LPU area should I check?", quick_replies=_location_replies())
    if lowered in {"hostels", "hostel"}:
        return _response(None, "Which hostel area should I check?", quick_replies=["Boys Hostel", "Girls Hostel"])
    if lowered in {"popular areas", "back", "start over", "change area"}:
        state["location"] = None
        state["vehicle_type"] = None
        state["last_results"] = []
        return _response(None, "Where in LPU are you heading?", quick_replies=_location_replies())
    if lowered in {"change vehicle", "vehicle", "another vehicle"}:
        state["vehicle_type"] = None
        if state.get("location"):
            return _response(None, f"What are you parking near {_display_location(state['location'])}?", quick_replies=_vehicle_replies())
        return _response(None, "Which vehicle are you parking?", quick_replies=_vehicle_replies())
    if lowered in {"change priority", "priority", "filter", "filters"}:
        return _response(None, "What should I prioritize?", quick_replies=_priority_replies())

    priority = _priority_from_message(lowered)
    if priority is not None and state.get("location") and state.get("vehicle_type"):
        if "ev_charging" in priority:
            state["vehicle_type"] = "ev"
        return _search_flow(state, {"budget_per_hour": None}, priority)
    return None


def _knowledge_flow(message: str, state: dict) -> dict | None:
    lowered = re.sub(r"\s+", " ", message.lower()).strip()
    if not lowered:
        return None

    answers = [
        (
            ["ensemble learning", "ensemble model", "ensemble"],
            "Ensemble learning means combining multiple models so the final prediction is usually more stable than a single model. Random Forest is an ensemble because it uses many decision trees and averages their results.",
        ),
        (
            ["random forest"],
            "Random Forest is an ensemble machine learning model made from many decision trees. In this project it estimates parking demand from time, day, vehicle type, price, safety, peak hours, and current free-space data.",
        ),
        (
            ["tf-idf", "tfidf", "term frequency"],
            "TF-IDF converts a sentence into useful word-weight features. Words that are important to a message get stronger weight, which helps the chatbot classify intents like finding parking, reserving, checking price, or asking directions.",
        ),
        (
            ["svm", "support vector", "linear svc"],
            "Linear SVM is the intent classifier here. It learns boundaries between different chatbot intents, so short messages like 'bike near hostel' or 'reserve first option' can be routed to the right action.",
        ),
        (
            ["entity extraction", "entities", "extract entity"],
            "Entity extraction pulls useful details from the message: destination, vehicle type, budget, reservation duration, selected option, and preferences such as cheapest, nearest, covered, secure, or EV charging.",
        ),
        (
            ["machine learning", "ml"],
            "Machine learning is when a program learns patterns from data instead of using only fixed rules. This parking assistant uses ML for intent classification and availability prediction.",
        ),
        (
            ["supervised learning"],
            "Supervised learning trains a model using examples with known answers. Here, the intent classifier is trained with example user messages and their correct intent labels.",
        ),
        (
            ["classification", "classifier"],
            "Classification means predicting a category. In this chatbot, classification decides whether a message is asking to find parking, reserve a spot, check price, get details, navigate, cancel, or greet the bot.",
        ),
        (
            ["regression", "regressor"],
            "Regression predicts a number instead of a category. The Random Forest Regressor predicts the expected free-space ratio for each parking lot.",
        ),
        (
            ["dataset", "data set", "training data"],
            "The project uses an LPU parking dataset with lot capacity, current availability, vehicle support, location coordinates, price, features, safety score, and peak-hour patterns.",
        ),
        (
            ["chatbot"],
            "A chatbot is a conversational interface. This one keeps session memory, asks missing follow-up questions, extracts parking details, and returns ranked LPU parking recommendations.",
        ),
        (
            ["neural network", "deep learning"],
            "Deep learning uses neural networks with many layers. This project does not need a deep model because the data is structured and small, so TF-IDF, SVM, and Random Forest are faster and easier to explain.",
        ),
        (
            ["artificial intelligence", "what is ai", " ai "],
            "Artificial Intelligence is software that performs tasks needing human-like reasoning, prediction, or language understanding. This project uses AI to understand parking requests and rank live parking options.",
        ),
        (
            ["which model", "what model", "model using", "model are you using", "how this model works", "how does this model work", "project work"],
            "This project uses a TF-IDF plus Linear SVM chatbot intent model, a rule-based entity extractor, and a Random Forest availability estimator. The final ranking considers live free spaces, walking distance, price, safety, facilities, and the selected LPU area.",
        ),
        (
            ["real time", "realtime", "live data", "live parking"],
            "This demo simulates real-time parking sensor changes every few seconds, then refreshes visible recommendations without repeating the same chat response. It is ready to connect to real sensor or database data later.",
        ),
        (
            ["what can you do", "help", "commands"],
            "I can find parking near LPU areas, compare cheapest or nearest options, show details, give directions, and hold a spot with a reservation code.",
        ),
    ]

    if lowered in {"hlo", "helo", "hii", "hy", "hello", "hi", "hey"}:
        return _response(
            None,
            "Hlo, I am ready. Tell me where in LPU you are heading and I will find available parking.",
            quick_replies=_location_replies(),
        )

    question_like = lowered.startswith(
        (
            "what is",
            "what are",
            "explain",
            "which model",
            "what model",
            "how does",
            "how this",
            "is this",
            "does this",
            "do you",
            "can you",
        )
    )
    direct_help = lowered in {"help", "commands", "what can you do"}
    topic_like = any(
        word in lowered
        for word in [
            "model",
            "learning",
            "forest",
            "tf-idf",
            "tfidf",
            "svm",
            "entity",
            "machine",
            "artificial",
            "ai",
            "classification",
            "classifier",
            "regression",
            "dataset",
            "data set",
            "chatbot",
            "neural",
            "deep",
            "help",
            "command",
        ]
    )
    for keys, answer in answers:
        if any(key in f" {lowered} " for key in keys) and (question_like or direct_help or topic_like):
            suffix = ""
            if state.get("location") and state.get("vehicle_type"):
                suffix = " I can continue with your current parking search too."
                quick_replies = _after_results_replies()
            else:
                suffix = " Now tell me the LPU area where you need parking."
                quick_replies = _location_replies()
            return _response(None, f"{answer}{suffix}", cards=state.get("last_results", []), quick_replies=quick_replies)

    return None


def _parking_flow(state: dict, entities: dict, preferences: list[str]) -> dict:
    if not state.get("location"):
        return _response(None, "Which LPU area should I check near?", quick_replies=_location_replies())

    if not state.get("vehicle_type"):
        return _response(
            None,
            f"What are you parking near {_display_location(state['location'])}?",
            quick_replies=_vehicle_replies(),
        )

    return _search_flow(state, entities, preferences)


def _search_flow(state: dict, entities: dict, preferences: list[str]) -> dict:
    vehicle_type = state["vehicle_type"]
    location = state["location"]
    budget = entities.get("budget_per_hour")

    if vehicle_type == "ev" and "ev_charging" not in preferences:
        preferences.append("ev_charging")
    if vehicle_type == "accessible" and "accessible" not in preferences:
        preferences.append("accessible")

    results = engine.recommend(
        location=location,
        vehicle_type=vehicle_type,
        preferences=preferences,
        budget_per_hour=budget,
        limit=3,
    )
    state["last_results"] = results
    state["last_selected_lot"] = results[0]["id"] if results else None
    state["last_preferences"] = list(preferences)
    state["last_budget"] = budget

    if not results:
        return _response(
            None,
            f"I do not see a suitable {_vehicle_phrase(vehicle_type)} option near {_display_location(location)} right now.",
            quick_replies=_location_replies(),
        )

    best = results[0]
    backup = "" if len(results) == 1 else f" I also found {len(results) - 1} backup option{'s' if len(results) > 2 else ''}."
    reply = (
        f"Near {_display_location(location)}, I would use {best['name']} for {_vehicle_phrase(vehicle_type)} parking. "
        f"It has {best['spaces_available']} of {best['capacity']} spaces free, about {best['walking_minutes']} min away, "
        f"and {_rate_phrase(best)}.{backup}"
    )
    return _response(None, reply, cards=results, quick_replies=_after_results_replies())


def _reserve_flow(state: dict, entities: dict) -> dict:
    vehicle_type = entities.get("vehicle_type") or state.get("vehicle_type")
    if not vehicle_type:
        return _response(None, "Which vehicle type should I hold a space for?", quick_replies=_vehicle_replies())

    lot_id = entities.get("lot_id")
    if not lot_id:
        last_results = state.get("last_results", [])
        ordinal = int(entities.get("ordinal", 0))
        if last_results and ordinal < len(last_results):
            lot_id = last_results[ordinal]["id"]
            vehicle_type = last_results[ordinal].get("vehicle_type", vehicle_type)
        else:
            lot_id = state.get("last_selected_lot")

    if not lot_id:
        return _response(None, "Ask me for parking first, then I can hold one of the options.", quick_replies=_location_replies())

    try:
        reservation = engine.reserve(lot_id, vehicle_type, int(entities.get("duration_minutes") or 45))
    except ValueError as error:
        return _response(None, str(error), cards=state.get("last_results", []), quick_replies=_after_results_replies())

    state["vehicle_type"] = vehicle_type
    state["last_selected_lot"] = lot_id
    return _response(None, _reservation_reply(reservation), quick_replies=[])


def _cancel_flow(entities: dict) -> dict:
    code = entities.get("reservation_code")
    if not code:
        return _response(None, "Send the reservation code, for example: cancel LPU-A1B2C3.")
    try:
        reservation = engine.cancel(code)
    except ValueError as error:
        return _response(None, str(error))
    return _response(None, f"Cancelled {reservation['code']} for {reservation['lot_name']}.")


def _details_flow(state: dict, entities: dict) -> dict:
    lot_id = entities.get("lot_id") or state.get("last_selected_lot")
    vehicle_type = entities.get("vehicle_type") or state.get("vehicle_type") or "car"
    if not lot_id:
        return _response(None, "Choose a parking option first and I will show its current details.", quick_replies=_location_replies())

    try:
        details = engine.details(lot_id, vehicle_type, state.get("location"))
    except ValueError as error:
        return _response(None, str(error))

    reply = (
        f"{details['name']} is showing {details['spaces_available']} of {details['capacity']} spaces free. "
        f"It is around {details['walking_minutes']} min away, {_rate_phrase(details)}, "
        f"and was last updated at {details['last_updated']}."
    )
    return _response(None, reply, cards=[details], quick_replies=_after_results_replies())


def _navigation_flow(state: dict, entities: dict) -> dict:
    cards = state.get("last_results", [])
    ordinal = int(entities.get("ordinal", 0))
    selected = cards[ordinal] if cards and ordinal < len(cards) else None
    if not selected and state.get("last_selected_lot"):
        selected = engine.details(state["last_selected_lot"], state.get("vehicle_type") or "car", state.get("location"))
    if not selected:
        return _response(None, "Ask me for parking first, then I can guide you to that option.", quick_replies=_location_replies())

    state["last_selected_lot"] = selected["id"]
    reply = (
        f"Go toward {selected['area']} and use {selected['name']}. "
        f"From your selected destination it is roughly {selected['distance_m']} m, about {selected['walking_minutes']} min on foot."
    )
    return _response(None, reply, cards=[selected], quick_replies=_after_results_replies())


def _remember_entities(state: dict, entities: dict) -> None:
    if entities.get("vehicle_type"):
        state["vehicle_type"] = entities["vehicle_type"]
    if entities.get("location"):
        state["location"] = entities["location"]
    if entities.get("lot_id"):
        state["last_selected_lot"] = entities["lot_id"]


def _override_intent(message: str, intent, entities: dict):
    lowered = message.lower()
    preferences = set(entities.get("preferences", []))
    if any(word in lowered for word in ["cancel", "remove booking", "free my reserved"]):
        return replace(intent, name="cancel_reservation", confidence=max(intent.confidence, 0.7))
    if any(word in lowered for word in ["book", "reserve", "confirm", "hold"]):
        return replace(intent, name="reserve_spot", confidence=max(intent.confidence, 0.72))
    if any(phrase in lowered for phrase in ["detail", "facility", "features", "tell me about", "information"]):
        return replace(intent, name="lot_details", confidence=max(intent.confidence, 0.65))
    if any(word in lowered for word in ["direction", "navigate", "route", "reach"]):
        return replace(intent, name="navigation", confidence=max(intent.confidence, 0.65))
    if entities.get("budget_per_hour") is not None or {"cheapest", "free"} & preferences or any(word in lowered for word in ["price", "cost", "rate"]):
        return replace(intent, name="price_query", confidence=max(intent.confidence, 0.68))
    if any(key in entities for key in ["vehicle_type", "location", "preferences", "budget_per_hour"]):
        return replace(intent, name="find_parking", confidence=max(intent.confidence, 0.64))
    return intent


def _is_ambiguous_hostel(message: str, entities: dict) -> bool:
    lowered = message.lower()
    return (
        bool(re.search(r"\bhostel\b", lowered))
        and "boys" not in lowered
        and "girls" not in lowered
        and "bh" not in lowered
        and "gh" not in lowered
        and entities.get("location") in {None, "boys hostel"}
    )


def _reservation_reply(reservation: dict) -> str:
    vehicle = _vehicle_phrase(reservation["vehicle_type"])
    article = "an" if vehicle[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return (
        f"Done. I held {article} {vehicle} space at {reservation['lot_name']} until "
        f"{reservation['hold_until']}. Code: {reservation['code']}."
    )


def _response(session_id: str | None, reply: str, cards: list[dict] | None = None, quick_replies: list[str] | None = None) -> dict:
    payload = {
        "reply": reply,
        "cards": cards or [],
        "quick_replies": quick_replies or [],
    }
    if session_id:
        payload["session_id"] = session_id
    return payload


def _location_replies() -> list[str]:
    return [
        "LPU Main Gate",
        "Central Library",
        "Block 34",
        "Uni Mall",
        "Hostels",
        "More areas",
    ]


def _more_location_replies() -> list[str]:
    return [
        "Boys Hostel",
        "Girls Hostel",
        "LPU Auditorium",
        "Sports Complex",
        "Admin Block",
        "LPU Campus",
        "Popular areas",
    ]


def _vehicle_replies() -> list[str]:
    return ["Car", "Bike", "EV", "Accessible"]


def _after_results_replies() -> list[str]:
    return ["Hold first option", "Show details", "Directions", "Change priority", "Change area"]


def _priority_replies() -> list[str]:
    return ["Best available", "Nearest", "Cheapest", "Secure", "Covered", "EV charging"]


def _priority_from_message(lowered: str) -> list[str] | None:
    mapping = {
        "best available": [],
        "best": [],
        "nearest": ["nearest"],
        "closest": ["nearest"],
        "cheapest": ["cheapest"],
        "cheap": ["cheapest"],
        "secure": ["security"],
        "security": ["security"],
        "covered": ["covered"],
        "ev charging": ["ev_charging"],
        "charging": ["ev_charging"],
    }
    return mapping.get(lowered)


def _display_location(location: str) -> str:
    replacements = {
        "lpu": "LPU",
        "lovely professional university": "Lovely Professional University",
        "lpu main gate": "LPU Main Gate",
        "academic block 34": "Academic Block 34",
    }
    if location in replacements:
        return replacements[location]
    return " ".join(part.upper() if part == "lpu" else part.capitalize() for part in location.split())


def _vehicle_phrase(vehicle_type: str) -> str:
    if vehicle_type == "ev":
        return "EV"
    return VEHICLE_LABELS.get(vehicle_type, vehicle_type).lower()


def _rate_phrase(lot: dict) -> str:
    return "is free" if lot.get("rate", 0) == 0 else f"costs {lot['rate_label']}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
