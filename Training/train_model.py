import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load datasets
drowsy_df = pd.read_csv(r"Dataset\drowsy.csv")
not_drowsy_df = pd.read_csv(r"Dataset\not_drowsy.csv")

# Combine and shuffle
data = pd.concat([drowsy_df, not_drowsy_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = data.drop('label', axis=1)
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, r"Saved_Models\drowsiness_model.pkl")
