from __future__ import annotations

import hashlib
import json
import math
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


DESTINATIONS: dict[str, dict[str, Any]] = {
    "lovely professional university": {
        "lat": 31.2533614,
        "lng": 75.7035391,
        "aliases": ["lpu", "campus", "college", "university", "lovely campus", "lpu campus"],
    },
    "central library": {
        "lat": 31.25415,
        "lng": 75.70405,
        "aliases": ["library", "main library", "central libray", "lpu library"],
    },
    "academic block 34": {
        "lat": 31.25355,
        "lng": 75.70465,
        "aliases": ["academic", "academic block", "academic building", "classes", "lecture hall", "block 34", "block thirty four"],
    },
    "lpu main gate": {
        "lat": 31.2529,
        "lng": 75.703,
        "aliases": ["main gate", "gate", "entry gate", "front gate", "gate 1", "lpu gate"],
    },
    "boys hostel": {
        "lat": 31.2556,
        "lng": 75.7021,
        "aliases": ["bh", "boys hostel", "boys hostels", "boys residence"],
    },
    "girls hostel": {
        "lat": 31.2549,
        "lng": 75.70515,
        "aliases": ["gh", "girls hostel", "girls hostels", "girls residence"],
    },
    "uni mall": {
        "lat": 31.2527,
        "lng": 75.70425,
        "aliases": ["unimall", "uni mall", "mall", "food court", "cafe", "cafeteria", "canteen"],
    },
    "lpu auditorium": {
        "lat": 31.2532,
        "lng": 75.7054,
        "aliases": ["auditorium", "audi", "seminar hall", "main hall", "event hall"],
    },
    "sports complex": {
        "lat": 31.25625,
        "lng": 75.7032,
        "aliases": ["sports", "ground", "gym", "stadium"],
    },
    "admin block": {
        "lat": 31.2524,
        "lng": 75.70365,
        "aliases": ["admin", "office", "accounts", "admission block", "admissions"],
    },
}


VEHICLE_LABELS = {
    "car": "Car",
    "bike": "Bike",
    "ev": "EV",
    "accessible": "Accessible",
}


FEATURE_LABELS = {
    "covered": "Covered",
    "security": "Security",
    "cctv": "CCTV",
    "ev_charging": "EV charging",
    "accessible": "Accessible",
    "guarded": "Guarded",
    "overnight": "Overnight",
}


class ParkingEngine:
    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.lots: list[dict[str, Any]] = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.lot_by_id = {lot["id"]: lot for lot in self.lots}
        self.live_free = {
            lot["id"]: {vehicle: int(count) for vehicle, count in lot["available"].items()}
            for lot in self.lots
        }
        now = datetime.now()
        self.last_updated = {
            lot["id"]: {
                vehicle: now - timedelta(minutes=int(abs(_stable_noise(f"start:{lot['id']}:{vehicle}")) * 6) + 1)
                for vehicle in lot["available"]
            }
            for lot in self.lots
        }
        self.last_simulated_at = now
        self.live_sequence = 0
        self.reservations: dict[str, dict[str, Any]] = {}
        self.predictor = DemandPredictor(self.lots)

    @property
    def destinations(self) -> dict[str, dict[str, Any]]:
        return DESTINATIONS

    def list_lots(self, vehicle_type: str | None = None) -> list[dict[str, Any]]:
        self.advance_live_state()
        vehicle = vehicle_type or "car"
        destination = DESTINATIONS["lovely professional university"]
        return [
            self._public_lot(lot, vehicle, destination)
            for lot in self.lots
            if vehicle in lot["vehicle_types"]
        ]

    def recommend(
        self,
        *,
        location: str | None,
        vehicle_type: str,
        preferences: list[str] | None = None,
        budget_per_hour: int | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        self.advance_live_state()
        destination_name = location or "lovely professional university"
        destination = DESTINATIONS.get(destination_name, DESTINATIONS["lovely professional university"])
        preferences = preferences or []
        ranked: list[dict[str, Any]] = []

        for lot in self.lots:
            if vehicle_type not in lot["vehicle_types"]:
                continue

            rate = self._rate(lot, vehicle_type)
            if budget_per_hour is not None and rate > budget_per_hour:
                continue

            distance_m = _distance_m(destination["lat"], destination["lng"], lot["lat"], lot["lng"])
            availability = self.estimate_availability(lot, vehicle_type)
            public = self._public_lot(lot, vehicle_type, destination)
            score = self._score_lot(
                lot=lot,
                vehicle_type=vehicle_type,
                distance_m=distance_m,
                availability=availability,
                preferences=preferences,
                budget_per_hour=budget_per_hour,
                area_match=_area_matches(lot, destination_name),
            )
            public["score"] = round(score, 3)
            public["recommendation"] = _recommendation_label(score, public["status"])
            public["reasons"] = self._reasons(public, preferences, budget_per_hour)
            public["reason"] = ", ".join(public["reasons"][:3])
            ranked.append(public)

        if "cheapest" in preferences or "free" in preferences:
            ranked.sort(key=lambda item: (item["rate"], item["distance_m"], -item["spaces_available"], -item["score"]))
        elif "nearest" in preferences:
            ranked.sort(key=lambda item: (item["distance_m"], -item["spaces_available"], -item["score"]))
        else:
            ranked.sort(key=lambda item: (item["score"], item["spaces_available"], -item["distance_m"]), reverse=True)
        return ranked[:limit]

    def estimate_availability(self, lot: dict[str, Any], vehicle_type: str) -> dict[str, Any]:
        now = datetime.now()
        capacity = int(lot["capacity"].get(vehicle_type, 0))
        if capacity <= 0:
            return _availability_payload(0, 0, 0, "full", "now")

        measured_free = int(self.live_free[lot["id"]].get(vehicle_type, 0))
        predicted_free = self.predictor.predict(lot, vehicle_type, now, measured_free, capacity)
        blended_free = round(measured_free * 0.72 + predicted_free * 0.28)
        free_spaces = max(0, min(capacity, blended_free))
        occupancy_percent = int(round((1 - free_spaces / capacity) * 100))

        if free_spaces <= 0:
            status = "full"
        elif free_spaces <= max(2, capacity * 0.1):
            status = "limited"
        elif occupancy_percent >= 80:
            status = "busy"
        else:
            status = "available"

        confidence = 0.86
        if self.predictor.pressure_label(lot, now) == "high":
            confidence -= 0.06
        if status in {"limited", "full"}:
            confidence -= 0.08

        return {
            "free_spaces": free_spaces,
            "capacity": capacity,
            "occupancy_percent": occupancy_percent,
            "status": status,
            "confidence": round(max(0.62, confidence), 2),
            "pressure": self.predictor.pressure_label(lot, now),
            "updated_at": self.last_updated[lot["id"]][vehicle_type].strftime("%I:%M %p").lstrip("0"),
        }

    def reserve(self, lot_id: str, vehicle_type: str, duration_minutes: int = 45) -> dict[str, Any]:
        self.advance_live_state()
        lot = self.lot_by_id.get(lot_id)
        if not lot:
            raise ValueError("That parking area was not found.")
        if vehicle_type not in lot["vehicle_types"]:
            raise ValueError(f"{lot['name']} does not support {vehicle_type} parking.")

        availability = self.estimate_availability(lot, vehicle_type)
        measured_free = int(self.live_free[lot_id].get(vehicle_type, 0))
        if measured_free <= 0 or availability["free_spaces"] <= 0:
            raise ValueError(f"{lot['name']} is currently full for {VEHICLE_LABELS.get(vehicle_type, vehicle_type).lower()} parking.")

        self.live_free[lot_id][vehicle_type] = measured_free - 1
        self.last_updated[lot_id][vehicle_type] = datetime.now()
        code = f"LPU-{secrets.token_hex(3).upper()}"
        hold_until = datetime.now() + timedelta(minutes=duration_minutes)
        reservation = {
            "code": code,
            "lot_id": lot_id,
            "lot_name": lot["name"],
            "vehicle_type": vehicle_type,
            "duration_minutes": duration_minutes,
            "hold_until": hold_until.strftime("%I:%M %p").lstrip("0"),
        }
        self.reservations[code] = reservation
        return reservation

    def cancel(self, code: str) -> dict[str, Any]:
        reservation = self.reservations.pop(code.upper(), None)
        if not reservation:
            raise ValueError("I could not find that reservation code.")

        lot = self.lot_by_id[reservation["lot_id"]]
        vehicle_type = reservation["vehicle_type"]
        capacity = int(lot["capacity"].get(vehicle_type, 0))
        self.live_free[lot["id"]][vehicle_type] = min(
            capacity,
            int(self.live_free[lot["id"]].get(vehicle_type, 0)) + 1,
        )
        self.last_updated[lot["id"]][vehicle_type] = datetime.now()
        return reservation

    def details(self, lot_id: str, vehicle_type: str = "car", location: str | None = None) -> dict[str, Any]:
        self.advance_live_state()
        lot = self.lot_by_id.get(lot_id)
        if not lot:
            raise ValueError("That parking area was not found.")
        destination = DESTINATIONS.get(location or "lovely professional university", DESTINATIONS["lovely professional university"])
        vehicle = vehicle_type if vehicle_type in lot["vehicle_types"] else lot["vehicle_types"][0]
        public = self._public_lot(lot, vehicle, destination)
        public["recommendation"] = "Selected"
        public["reasons"] = self._reasons(public, [], None)
        public["reason"] = ", ".join(public["reasons"][:3])
        return public

    def advance_live_state(self) -> None:
        now = datetime.now()
        elapsed = int((now - self.last_simulated_at).total_seconds())
        if elapsed < 7:
            return

        steps = min(5, elapsed // 7)
        for _ in range(steps):
            self.live_sequence += 1
            for lot in self.lots:
                pressure = self.predictor.pressure_label(lot, now)
                for vehicle_type, capacity in lot["capacity"].items():
                    current = int(self.live_free[lot["id"]].get(vehicle_type, 0))
                    if capacity <= 0:
                        continue
                    noise = _stable_noise(f"live:{self.live_sequence}:{lot['id']}:{vehicle_type}")
                    if pressure == "high":
                        delta = -max(0, round(capacity * (0.006 + max(0, noise) * 0.018)))
                    elif pressure == "medium":
                        delta = round(noise * capacity * 0.014)
                    elif pressure == "low":
                        delta = max(0, round(capacity * (0.004 + max(0, noise) * 0.014)))
                    else:
                        delta = round(noise * capacity * 0.01)

                    if delta == 0 and abs(noise) > 0.42:
                        delta = 1 if noise > 0 else -1

                    updated = max(0, min(int(capacity), current + delta))
                    if updated != current:
                        self.live_free[lot["id"]][vehicle_type] = updated
                        self.last_updated[lot["id"]][vehicle_type] = now

        self.last_simulated_at = now

    def _public_lot(self, lot: dict[str, Any], vehicle_type: str, destination: dict[str, Any]) -> dict[str, Any]:
        distance_m = round(_distance_m(destination["lat"], destination["lng"], lot["lat"], lot["lng"]))
        availability = self.estimate_availability(lot, vehicle_type)
        return {
            "id": lot["id"],
            "name": lot["name"],
            "area": lot["area"],
            "vehicle_type": vehicle_type,
            "vehicle_label": VEHICLE_LABELS.get(vehicle_type, vehicle_type.title()),
            "vehicle_types": lot["vehicle_types"],
            "lat": lot["lat"],
            "lng": lot["lng"],
            "map_position": lot.get("map_position", [50, 50]),
            "capacity": availability["capacity"],
            "spaces_available": availability["free_spaces"],
            "predicted_available": availability["free_spaces"],
            "status": availability["status"],
            "availability_risk": availability["status"],
            "prediction_confidence": availability["confidence"],
            "occupancy_percent": availability["occupancy_percent"],
            "pressure": availability["pressure"],
            "rate": self._rate(lot, vehicle_type),
            "rate_label": self._rate_label(lot, vehicle_type),
            "distance_m": distance_m,
            "walking_minutes": max(1, round(distance_m / 78)),
            "features": lot.get("features", []),
            "feature_labels": [FEATURE_LABELS.get(feature, feature.title()) for feature in lot.get("features", [])],
            "safety_score": lot.get("safety_score", 0),
            "last_updated": availability["updated_at"],
        }

    def _score_lot(
        self,
        *,
        lot: dict[str, Any],
        vehicle_type: str,
        distance_m: float,
        availability: dict[str, Any],
        preferences: list[str],
        budget_per_hour: int | None,
        area_match: bool,
    ) -> float:
        capacity = max(1, availability["capacity"])
        free_ratio = availability["free_spaces"] / capacity
        free_score = min(1.0, free_ratio / 0.22)
        distance_score = 1 / (1 + distance_m / 260)
        price_score = 1 - min(self._rate(lot, vehicle_type), 40) / 40
        facility_score = self._facility_score(lot, preferences)
        safety_score = float(lot.get("safety_score", 4)) / 5

        if availability["status"] == "full":
            return 0.0

        score = (
            free_score * 0.34
            + distance_score * 0.32
            + price_score * 0.12
            + facility_score * 0.12
            + safety_score * 0.1
        )
        if "cheapest" in preferences or "free" in preferences or budget_per_hour is not None:
            score += price_score * 0.12
        if "nearest" in preferences:
            score += distance_score * 0.1
        if area_match:
            score += 0.16
        if availability["status"] == "limited":
            score -= 0.12
        if availability["status"] == "busy":
            score -= 0.05
        return max(0.01, min(0.99, score))

    def _facility_score(self, lot: dict[str, Any], preferences: list[str]) -> float:
        required = [preference for preference in preferences if preference not in {"nearest", "cheapest", "free"}]
        if not required:
            return 0.5
        matches = sum(1 for preference in required if preference in lot.get("features", []))
        return matches / len(required)

    def _rate(self, lot: dict[str, Any], vehicle_type: str) -> int:
        rates = lot.get("hourly_rate", {})
        if vehicle_type == "accessible":
            return int(rates.get("car", 0))
        return int(rates.get(vehicle_type, rates.get("car", 0)))

    def _rate_label(self, lot: dict[str, Any], vehicle_type: str) -> str:
        rate = self._rate(lot, vehicle_type)
        return "Free" if rate == 0 else f"Rs {rate}/hr"

    def _reasons(self, lot: dict[str, Any], preferences: list[str], budget_per_hour: int | None) -> list[str]:
        reasons: list[str] = []
        if lot["distance_m"] <= 120:
            reasons.append("closest to destination")
        elif lot["distance_m"] <= 360:
            reasons.append("short walk")
        if lot["spaces_available"] >= max(3, lot["capacity"] * 0.2):
            reasons.append("good chance of space")
        elif lot["spaces_available"] > 0:
            reasons.append("spaces filling fast")
        if lot["rate"] == 0:
            reasons.append("free parking")
        elif budget_per_hour is not None:
            reasons.append("within budget")
        if "ev_charging" in preferences and "ev_charging" in lot["features"]:
            reasons.append("EV charging")
        if "security" in preferences and ("security" in lot["features"] or "guarded" in lot["features"]):
            reasons.append("guarded area")
        if "covered" in preferences and "covered" in lot["features"]:
            reasons.append("covered parking")
        return reasons or ["balanced option"]


class DemandPredictor:
    VEHICLE_CODES = {"car": 0, "bike": 1, "ev": 2, "accessible": 3}

    def __init__(self, lots: list[dict[str, Any]]) -> None:
        self.model = RandomForestRegressor(
            n_estimators=110,
            max_depth=8,
            min_samples_leaf=3,
            random_state=274,
        )
        x_train, y_train = self._make_training_data(lots)
        self.model.fit(x_train, y_train)

    def predict(self, lot: dict[str, Any], vehicle_type: str, now: datetime, measured_free: int, capacity: int) -> int:
        features = np.array([self._features(lot, vehicle_type, now, measured_free, capacity)], dtype=float)
        free_ratio = float(self.model.predict(features)[0])
        small_variation = _stable_noise(f"{lot['id']}:{vehicle_type}:{now:%Y%m%d%H}") * 0.035
        free_ratio = max(0.0, min(1.0, free_ratio + small_variation))
        return round(capacity * free_ratio)

    def pressure_label(self, lot: dict[str, Any], now: datetime) -> str:
        if now.hour in set(lot.get("peak_hours", [])):
            return "high"
        if now.hour in {8, 9, 13, 14, 16, 17, 18, 19}:
            return "medium"
        if now.hour in {21, 22, 23, 0, 1, 2, 3, 4, 5, 6}:
            return "low"
        return "normal"

    def _make_training_data(self, lots: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        targets: list[float] = []
        start = datetime(2026, 1, 5)
        for lot_index, lot in enumerate(lots):
            for vehicle_type, capacity in lot["capacity"].items():
                capacity = int(capacity)
                if capacity <= 0:
                    continue
                measured_free = int(lot["available"].get(vehicle_type, max(1, capacity // 5)))
                for day in range(35):
                    for hour in range(6, 23):
                        moment = start + timedelta(days=day, hours=hour)
                        rows.append(self._features(lot, vehicle_type, moment, measured_free, capacity))
                        targets.append(self._target_ratio(lot, vehicle_type, moment, measured_free, capacity, lot_index))
        return np.array(rows, dtype=float), np.array(targets, dtype=float)

    def _features(
        self,
        lot: dict[str, Any],
        vehicle_type: str,
        moment: datetime,
        measured_free: int,
        capacity: int,
    ) -> list[float]:
        rates = lot.get("hourly_rate", {})
        return [
            moment.hour / 23,
            moment.weekday() / 6,
            1.0 if moment.weekday() >= 5 else 0.0,
            1.0 if moment.hour in set(lot.get("peak_hours", [])) else 0.0,
            measured_free / capacity,
            self.VEHICLE_CODES.get(vehicle_type, 0) / 3,
            min(40, int(rates.get(vehicle_type, rates.get("car", 0)))) / 40,
            float(lot.get("safety_score", 4)) / 5,
            1.0 if "ev_charging" in lot.get("features", []) else 0.0,
            1.0 if "overnight" in lot.get("features", []) else 0.0,
        ]

    def _target_ratio(
        self,
        lot: dict[str, Any],
        vehicle_type: str,
        moment: datetime,
        measured_free: int,
        capacity: int,
        lot_index: int,
    ) -> float:
        ratio = measured_free / capacity
        if moment.hour in set(lot.get("peak_hours", [])):
            ratio -= 0.18
        if moment.hour in {9, 10, 11, 12, 16, 17, 18}:
            ratio -= 0.07
        if vehicle_type == "bike":
            ratio += 0.06
        if vehicle_type == "ev":
            ratio -= 0.04
        if "overnight" in lot.get("features", []) and moment.hour >= 20:
            ratio -= 0.1
        if moment.weekday() >= 5:
            ratio += 0.1
        wave = math.sin((moment.timetuple().tm_yday + lot_index * 5) / 6) * 0.035
        return max(0.02, min(0.94, ratio + wave))


def _availability_payload(
    free_spaces: int,
    capacity: int,
    occupancy_percent: int,
    status: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "free_spaces": free_spaces,
        "capacity": capacity,
        "occupancy_percent": occupancy_percent,
        "status": status,
        "confidence": 0.92,
        "pressure": "normal",
        "updated_at": updated_at,
    }


def _distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    hav = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))


def _stable_noise(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value - 0.5


def _updated_at(lot_id: str, vehicle_type: str, now: datetime) -> str:
    offset_minutes = int(abs(_stable_noise(f"updated:{lot_id}:{vehicle_type}:{now:%Y%m%d%H}")) * 5) + 1
    updated = now - timedelta(minutes=offset_minutes)
    return updated.strftime("%I:%M %p").lstrip("0")


def _area_matches(lot: dict[str, Any], location: str) -> bool:
    return lot.get("area", "").lower() == location.lower()


def _recommendation_label(score: float, status: str) -> str:
    if status == "full":
        return "Full"
    if score >= 0.82:
        return "Best choice"
    if status in {"busy", "limited"}:
        return "Filling fast"
    return "Backup option"
