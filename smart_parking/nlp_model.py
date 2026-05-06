from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class IntentResult:
    name: str
    confidence: float
    top: list[dict[str, float]]


class IntentClassifier:
    """Small supervised NLP model for parking-chatbot intents."""

    def __init__(self) -> None:
        examples = _training_examples()
        texts: list[str] = []
        labels: list[str] = []
        for label, phrases in examples.items():
            texts.extend(phrases)
            labels.extend([label] * len(phrases))

        self.model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=2.8,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.model.fit(texts, labels)

    def predict(self, text: str) -> IntentResult:
        scores = self.model.decision_function([text])[0]
        probabilities = _softmax(scores, temperature=0.72)
        classes = list(self.model.classes_)
        ranked = sorted(
            zip(classes, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        top = [{"intent": name, "confidence": round(float(score), 3)} for name, score in ranked[:3]]
        return IntentResult(
            name=ranked[0][0],
            confidence=round(float(ranked[0][1]), 3),
            top=top,
        )


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / temperature
    shifted = scaled - np.max(scaled)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


class EntityExtractor:
    """Hybrid entity extractor for locations, vehicles, budgets, and commands."""

    VEHICLE_SYNONYMS = {
        "car": ["car", "cars", "four wheeler", "sedan", "suv", "hatchback"],
        "bike": ["bike", "bikes", "motorbike", "motorcycle", "two wheeler", "scooty", "scooter"],
        "ev": ["ev", "electric", "electric car", "electric vehicle", "charging"],
        "accessible": ["accessible", "disabled", "handicap", "wheelchair"],
    }

    PREFERENCE_SYNONYMS = {
        "covered": ["covered", "shade", "shaded", "basement", "roof"],
        "security": ["secure", "security", "guard", "safe", "cctv"],
        "ev_charging": ["charging", "charger", "ev"],
        "accessible": ["accessible", "disabled", "wheelchair", "handicap"],
        "free": ["free", "no charge"],
        "cheapest": ["cheap", "cheapest", "low cost", "minimum price", "budget"],
        "nearest": ["near", "nearest", "closest", "short walk"],
    }

    ORDINALS = {
        "1": 0,
        "first": 0,
        "one": 0,
        "top": 0,
        "2": 1,
        "second": 1,
        "two": 1,
        "3": 2,
        "third": 2,
        "three": 2,
    }

    def __init__(self, locations: dict[str, Any], lots: list[dict[str, Any]]) -> None:
        self.alias_to_location: dict[str, str] = {}
        for location, meta in locations.items():
            self.alias_to_location[_clean(location)] = location
            for alias in meta.get("aliases", []):
                self.alias_to_location[_clean(alias)] = location

        self.alias_to_lot: dict[str, str] = {}
        for lot in lots:
            aliases = [lot["name"], lot["id"], lot.get("area", "")]
            aliases.extend(lot.get("aliases", []))
            for alias in aliases:
                cleaned = _clean(alias)
                if cleaned:
                    self.alias_to_lot[cleaned] = lot["id"]

        self.location_aliases = sorted(self.alias_to_location, key=len, reverse=True)
        self.lot_aliases = sorted(self.alias_to_lot, key=len, reverse=True)

    def extract(self, text: str) -> dict[str, Any]:
        cleaned = _clean(text)
        entities: dict[str, Any] = {
            "vehicle_type": self._vehicle(cleaned),
            "location": self._location(cleaned),
            "lot_id": self._lot(cleaned),
            "duration_minutes": self._duration(cleaned),
            "budget_per_hour": self._budget(cleaned),
            "preferences": self._preferences(cleaned),
            "ordinal": self._ordinal(cleaned),
            "reservation_code": self._reservation_code(text),
        }
        return {key: value for key, value in entities.items() if value not in (None, [], {})}

    def _vehicle(self, cleaned: str) -> str | None:
        for vehicle, synonyms in self.VEHICLE_SYNONYMS.items():
            for synonym in synonyms:
                if re.search(rf"\b{re.escape(synonym)}\b", cleaned):
                    return vehicle
        return None

    def _location(self, cleaned: str) -> str | None:
        for alias in self.location_aliases:
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                return self.alias_to_location[alias]

        match = re.search(r"\b(?:near|at|around|beside|to|in)\s+([a-z0-9\s.-]{3,40})", cleaned)
        if match:
            candidate = match.group(1).strip()
            close = get_close_matches(candidate, self.location_aliases, n=1, cutoff=0.62)
            if close:
                return self.alias_to_location[close[0]]
        if 3 <= len(cleaned) <= 42:
            close = get_close_matches(cleaned, self.location_aliases, n=1, cutoff=0.72)
            if close:
                return self.alias_to_location[close[0]]
        return None

    def _lot(self, cleaned: str) -> str | None:
        for alias in self.lot_aliases:
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                return self.alias_to_lot[alias]
        return None

    def _duration(self, cleaned: str) -> int | None:
        match = re.search(r"\b(\d{1,2})\s*(hours?|hrs?|h)\b", cleaned)
        if match:
            return int(match.group(1)) * 60
        match = re.search(r"\b(\d{1,3})\s*(minutes?|mins?|m)\b", cleaned)
        if match:
            return int(match.group(1))
        return None

    def _budget(self, cleaned: str) -> int | None:
        match = re.search(r"\b(?:under|below|less than|max|maximum|upto|up to)\s*(?:rs\.?\s*)?(\d{1,4})\b", cleaned)
        if match:
            return int(match.group(1))
        return 0 if re.search(r"\bfree\b", cleaned) else None

    def _preferences(self, cleaned: str) -> list[str]:
        preferences: list[str] = []
        for preference, synonyms in self.PREFERENCE_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(synonym)}\b", cleaned) for synonym in synonyms):
                preferences.append(preference)
        return preferences

    def _ordinal(self, cleaned: str) -> int | None:
        for token, index in self.ORDINALS.items():
            if re.search(rf"\b{re.escape(token)}\b", cleaned):
                return index
        return None

    def _reservation_code(self, text: str) -> str | None:
        match = re.search(r"\b((?:PW-[A-Z0-9]{5})|(?:LPU-[A-Z0-9]{6}))\b", text.upper())
        return match.group(1) if match else None


def _training_examples() -> dict[str, list[str]]:
    return {
        "greeting": [
            "hello",
            "hlo",
            "helo",
            "hii",
            "hy",
            "hi",
            "hey bot",
            "good morning",
            "good evening",
            "start",
            "what can you do",
        ],
        "find_parking": [
            "find parking near library",
            "find available parking spots",
            "where can i park my car",
            "parking for bike near hostel",
            "bike parking near boys hostel",
            "show empty parking",
            "i need a car spot",
            "available car parking near cafeteria",
            "spot near main gate",
            "car near lpu main gate",
            "parking near uni mall",
            "nearest parking for scooter",
            "find me a free parking place",
            "best parking near academic block",
            "parking near block 34",
            "i am going to auditorium find parking",
            "need covered parking for electric vehicle",
            "where should i park",
            "parking around sports complex",
            "look for parking at admin block",
            "find available parking at lovely professional university",
        ],
        "check_availability": [
            "is parking available",
            "check availability",
            "how many spots are free",
            "any slot available in library basement",
            "availability near hostel",
            "is there space for bikes",
            "are car slots open",
            "parking status",
            "show current available slots",
            "is the main gate parking full",
            "is uni mall parking available",
            "availability near block 34",
        ],
        "reserve_spot": [
            "reserve a spot",
            "book parking",
            "book the first one",
            "reserve the nearest parking",
            "hold this spot for me",
            "reserve car parking for one hour",
            "book library basement",
            "confirm the second option",
            "take the top result",
            "reserve for 30 minutes",
        ],
        "cancel_reservation": [
            "cancel reservation",
            "cancel booking",
            "remove my booking",
            "cancel PW-ABCDE",
            "cancel LPU-A1B2C3",
            "i do not need the spot",
            "free my reserved spot",
        ],
        "price_query": [
            "show cheapest parking",
            "parking under rs 20",
            "free parking near library",
            "cheapest parking near uni mall",
            "what is the price",
            "how much does parking cost",
            "lowest price parking",
            "budget parking for car",
            "parking rate near gate",
        ],
        "lot_details": [
            "details of library basement",
            "tell me about gate 2 parking",
            "does this lot have security",
            "is ev charging available",
            "show facilities",
            "parking lot information",
            "what features are there",
        ],
        "navigation": [
            "how do i reach there",
            "give directions",
            "navigate to parking",
            "walking time from library",
            "route to the selected lot",
            "where is the first parking",
        ],
        "thanks": [
            "thanks",
            "thank you",
            "ok thanks",
            "great",
            "nice",
            "that helps",
        ],
    }
