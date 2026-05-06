# LPU Parking Assistant

A Flask + scikit-learn chatbot that finds available parking spots around Lovely Professional University, Phagwara.

## What It Does

- Understands parking chat requests using a trained TF-IDF + Linear SVM intent classifier.
- Asks follow-up questions when the destination or vehicle type is missing.
- Extracts vehicle type, destination, budget, duration, and preferences.
- Estimates availability with a Random Forest model trained on campus demand patterns.
- Simulates live sensor changes and refreshes visible recommendations without repeating chat messages.
- Ranks parking lots by availability, walking distance, price, safety, and features.
- Reserves a suggested spot and returns a reservation code.
- Handles casual greetings and typo-style inputs such as `Hlo` and `libary`.
- Answers short AI/demo questions such as ensemble learning, Random Forest, TF-IDF, SVM, dataset, and model workflow.
- Shows a focused, mobile-friendly chat UI with practical LPU parking recommendations.

## Run

```powershell
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Run smoke tests:

```powershell
python -m unittest discover -s tests
```

## Example Prompts

```text
Hlo
What is ensemble learning?
Which model are you using?
Find available parking spots near LPU Main Gate for car
Bike near Boys Hostel
Cheapest parking near Uni Mall
EV charging near Block 34
Live parking near library
Reserve first option
Show details
Cancel LPU-A1B2C3
```

## Main Files

- `app.py` - Flask routes and chatbot session flow
- `smart_parking/nlp_model.py` - intent classifier and entity extraction
- `smart_parking/parking_engine.py` - availability prediction, scoring, reservation logic
- `data/parking_lots.json` - campus parking dataset
- `static/` - web app UI
