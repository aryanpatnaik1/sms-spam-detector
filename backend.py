from flask import Flask, jsonify, request
from pathlib import Path
from joblib import load

app = Flask(__name__, static_folder="frontend", static_url_path="/frontend")
modelpath = Path(__file__).parent / "models" / "spam_classifier.joblib"
model = load(modelpath)
@app.get("/")
def home():
    return app.send_static_file("index.html")

@app.get("/health")
def health():
    return "Backend is running"

@app.post("/predict")
def predict():
    data = request.get_json()
    if(not isinstance(data, dict)):
        return jsonify(error="Expected a JSON object"), 400
    message = data.get("message")
    if(not isinstance(message, str)):
        return jsonify(error="Expected a string for 'message'"), 400
    message = message.strip()
    if(not message):
        return jsonify(error="Please enter a message"), 400
    probabilities = model.predict_proba([message])[0]
    scores = dict(zip(model.classes_, probabilities))
    spam_score = scores["spam"]
    if( spam_score >= 0.60):
        prediction = "SPAM"
    elif( spam_score<= 0.40):
        prediction = "NOT SPAM"
    else:
        prediction = "UNCERTAIN"
    return jsonify(prediction=prediction, spam_score = float(spam_score))
