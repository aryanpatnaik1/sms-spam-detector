from pathlib import Path
import joblib

model_path = Path(__file__).parent / "models" / "spam_classifier.joblib"
model = joblib.load(model_path)

print("Spam detector ready. Type 'quit' to exit.")

while True:
    message = input("\nEnter a message: ").strip()

    if message.lower() == "quit":
        break

    if not message:
        print("Please enter a message.")
        continue

    probabilities = model.predict_proba([message])[0]
    scores = dict(zip(model.classes_, probabilities))
    spam_score = scores["spam"]

    if spam_score >= 0.60:
        print("Prediction: spam")
    elif spam_score <= 0.40:
        print("Prediction: ham")
    else:
        print("Prediction: uncertain — review this message")

    print(f"Spam score: {spam_score:.1%}")