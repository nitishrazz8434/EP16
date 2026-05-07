import re
import unittest

from app import app


class ParkingChatbotFlowTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def ask(self, session_id, message):
        response = self.client.post("/api/chat", json={"session_id": session_id, "message": message})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["frontend"], "React")

    def test_slot_filling_then_recommendation(self):
        first = self.ask("slot-flow", "LPU Main Gate")
        self.assertIn("What are you parking", first["reply"])
        self.assertEqual(first["cards"], [])

        second = self.ask("slot-flow", "Car")
        self.assertIn("LPU Main Gate Parking", second["reply"])
        self.assertGreaterEqual(len(second["cards"]), 1)
        self.assertEqual(second["cards"][0]["vehicle_type"], "car")

    def test_ambiguous_hostel_question(self):
        response = self.ask("hostel-flow", "Bike near Hostel")
        self.assertIn("Which hostel", response["reply"])
        self.assertIn("Boys Hostel", response["quick_replies"])
        self.assertIn("Girls Hostel", response["quick_replies"])

    def test_ev_charging_near_block_34(self):
        response = self.ask("ev-flow", "EV charging near Block 34")
        self.assertIn("Block 34", response["reply"])
        self.assertEqual(response["cards"][0]["vehicle_type"], "ev")
        self.assertIn("ev_charging", response["cards"][0]["features"])

    def test_typos_and_ai_concept_questions(self):
        greeting = self.ask("concept-flow", "Hlo")
        self.assertIn("Hlo", greeting["reply"])
        self.assertIn("LPU Main Gate", greeting["quick_replies"])

        concept = self.ask("concept-flow", "what is ensemble learning?")
        self.assertIn("Random Forest", concept["reply"])
        self.assertEqual(concept["cards"], [])

        model = self.ask("concept-flow", "which model are you using")
        self.assertIn("TF-IDF", model["reply"])
        self.assertIn("Random Forest", model["reply"])

    def test_varied_project_questions_do_not_fall_into_same_parking_prompt(self):
        cases = {
            "what is api": "React UI calls Flask API",
            "what is flask": "Python web framework",
            "what is react": "frontend library",
            "project features": "conversational slot filling",
            "which algorithm is used": "TF-IDF",
            "who are you": "LPU Parking Assistant",
            "how are you": "running fine",
            "what is photosynthesis": "focused on this parking-assistant project",
        }
        for index, (message, expected) in enumerate(cases.items()):
            with self.subTest(message=message):
                response = self.ask(f"varied-flow-{index}", message)
                self.assertIn(expected, response["reply"])
                self.assertNotEqual(response["reply"], "Which LPU area should I check near?")
                if message != "what is photosynthesis":
                    self.assertEqual(response["intent"], "project_question")
                else:
                    self.assertEqual(response["intent"], "clarification")

    def test_fuzzy_location_typo(self):
        response = self.ask("typo-flow", "libary")
        self.assertIn("Central Library", response["reply"])
        self.assertIn("Car", response["quick_replies"])

    def test_more_areas_and_priority_chat_flow(self):
        more = self.ask("expanded-flow", "More areas")
        self.assertIn("more LPU areas", more["reply"])
        self.assertIn("LPU Auditorium", more["quick_replies"])
        self.assertIn("Sports Complex", more["quick_replies"])

        location = self.ask("expanded-flow", "LPU Auditorium")
        self.assertIn("What are you parking", location["reply"])

        vehicle = self.ask("expanded-flow", "Car")
        self.assertGreaterEqual(len(vehicle["cards"]), 1)
        self.assertIn("Change priority", vehicle["quick_replies"])

        priority = self.ask("expanded-flow", "Change priority")
        self.assertIn("prioritize", priority["reply"])
        self.assertIn("Cheapest", priority["quick_replies"])

        cheapest = self.ask("expanded-flow", "Cheapest")
        self.assertGreaterEqual(len(cheapest["cards"]), 1)
        self.assertIn("LPU Auditorium", cheapest["reply"])

    def test_live_refresh_uses_session_context(self):
        self.ask("refresh-flow", "Car near LPU Main Gate")
        response = self.client.post("/api/refresh", json={"session_id": "refresh-flow"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["updated"])
        self.assertGreaterEqual(len(data["cards"]), 1)
        self.assertEqual(data["memory"]["vehicle_type"], "car")

    def test_reservation_and_cancel(self):
        self.ask("reserve-flow", "Car near LPU Main Gate")
        reserve = self.ask("reserve-flow", "Hold first option")
        self.assertRegex(reserve["reply"], r"LPU-[A-Z0-9]{6}")
        self.assertEqual(reserve["cards"], [])
        code = re.search(r"LPU-[A-Z0-9]{6}", reserve["reply"]).group(0)

        cancel = self.ask("reserve-flow", f"Cancel {code}")
        self.assertIn("Cancelled", cancel["reply"])


if __name__ == "__main__":
    unittest.main()
