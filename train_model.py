from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

data_path = Path(__file__).parent / "data" / "SMSSpamCollection"
unique_messages = {}

with open(data_path, encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        label, message = line.rstrip("\n").split("\t", 1)

        if label not in ("spam", "ham"):
            raise ValueError(f"Unknown label on line {line_number}")

        if not message.strip():
            raise ValueError(f"Empty message on line {line_number}")

        if message in unique_messages:
            if unique_messages[message] != label:
                raise ValueError(
                    f"Conflicting labels on line {line_number}"
                )
        else:
            unique_messages[message] = label

messages = list(unique_messages.keys())
labels = list(unique_messages.values())

print("Unique messages loaded:", len(messages))
print("First message:", messages[0])
print("Its label:", labels[0])
# Reserve 20% of the examples for final testing.
remaining_messages, test_messages, remaining_labels, test_labels = (
    train_test_split(
        messages,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
)

# Split the remaining 80% into training and validation groups.
train_messages, validation_messages, train_labels, validation_labels = (
    train_test_split(
        remaining_messages,
        remaining_labels,
        test_size=0.25,
        random_state=42,
        stratify=remaining_labels,
    )
)
print("Training examples:", len(train_messages))
print("Validation examples:", len(validation_messages))
print("Test examples:", len(test_messages))
model = Pipeline([
    ("text", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(train_messages, train_labels)

print("Model training complete.")

sample_messages = [
    "Are we still meeting at the library?",
    "Congratulations! You won a cash prize. Call now to claim!",
]

sample_predictions = model.predict(sample_messages)

for message, prediction in zip(sample_messages, sample_predictions):
    print("Message:", message)
    print("Prediction:", prediction)
    print()
# Predict labels for messages the model did not train on.
validation_predictions = model.predict(validation_messages)

# Compare against a simple strategy: always predict ham.
baseline_predictions = ["ham"] * len(validation_labels)

baseline_accuracy = accuracy_score(
    validation_labels, baseline_predictions
)

model_accuracy = accuracy_score(
    validation_labels, validation_predictions
)

print(f"\nAlways-ham baseline accuracy: {baseline_accuracy:.1%}")
print(f"Model validation accuracy: {model_accuracy:.1%}")

print("\nValidation report:")
print(classification_report(
    validation_labels,
    validation_predictions,
    digits=3,
    zero_division=0,
))
missed_spam = 0
false_alarms = 0

print("\nExamples of missed spam:")

for message, actual, predicted in zip(
    validation_messages,
    validation_labels,
    validation_predictions,
):
    if actual == "spam" and predicted == "ham":
        missed_spam += 1

        if missed_spam <= 5:
            print("\nMessage:", message)
            print("Actual:", actual, "| Predicted:", predicted)

    elif actual == "ham" and predicted == "spam":
        false_alarms += 1

print("\nTotal missed spam:", missed_spam)
print("Total false alarms:", false_alarms)
balanced_model = Pipeline([
    ("text", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )),
])

balanced_model.fit(train_messages, train_labels)

balanced_predictions = balanced_model.predict(validation_messages)

print("\nBalanced model validation report:")
print(classification_report(
    validation_labels,
    balanced_predictions,
    digits=3,
    zero_division=0,
))
# Select the model using the validation results.
selected_model = balanced_model

# Evaluate it on the reserved test messages.
test_predictions = selected_model.predict(test_messages)

test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\nFinal test accuracy: {test_accuracy:.1%}")

print("\nFinal test report:")
print(classification_report(
    test_labels,
    test_predictions,
    digits=3,
    zero_division=0,
))
# Create a folder for the saved model.
model_directory = Path(__file__).parent / "models"
model_directory.mkdir(exist_ok=True)

# Save the selected model and its text-processing step together.
model_path = model_directory / "spam_classifier.joblib"
joblib.dump(selected_model, model_path)

print("\nModel saved to:", model_path)