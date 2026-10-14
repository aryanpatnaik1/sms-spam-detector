# SMS Spam Detector

I built this project to learn how to train a machine learning model and connect it to a website. You can enter a text message and get a result of spam, not spam, or uncertain, along with a spam score.

## What I used

- Python and scikit-learn for training the model
- TF-IDF to turn message text into numerical features
- Logistic regression to classify messages
- Flask for the backend
- HTML, CSS, and JavaScript for the frontend

## How it works

The frontend sends the message to the Flask backend as JSON. The backend loads the saved model, gets its scores, and sends the result back to the page. It does not retrain the model for each message.

I removed exact duplicate messages and split the 5,171 unique examples into training, validation, and test groups. I compared a regular logistic regression model with one using balanced class weights and selected the balanced model using the validation results.

## Results

On the separate test set of 1,035 messages, the selected model achieved:

- Accuracy: 97.4%
- Spam precision: 88.8%
- Spam recall: 90.8%
- Spam F1-score: 0.898

These results are for the model's original spam/ham predictions. The app separately marks spam scores above 40% and below 60% as uncertain. That review range is experimental and is not included in these metrics. A score is a model estimate, and the app can still make mistakes.

## Run locally on macOS

From the project folder, create and activate a virtual environment, then install the recorded dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The app needs the saved model at `models/spam_classifier.joblib`. To recreate it, place the UCI `SMSSpamCollection` file in `data/` and run `python train_model.py`.

Start the app:

```bash
python -m flask --app backend run --port 8000
```

Open http://127.0.0.1:8000/ in a browser. You can also run `python predict.py` to try messages in the terminal.

## Dataset

I used the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) by Almeida and Hidalgo (2011), licensed under CC BY 4.0. It contains older SMS messages, so the test results may not reflect performance on newer messages.
