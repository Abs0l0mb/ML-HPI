import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = '../data/substances.csv'
data = pd.read_csv(file_path)

target_column = 'substance'
X = data.drop(columns=[target_column])
y = data[target_column]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost classifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

unique_labels = sorted(set(y_test))
#print(classification_report(y_test, y_pred, target_names=label_encoder.classes_[unique_labels], labels=unique_labels))


# Save the model
model.save_model('../models/xgboost_classifier_model.json')
print("Model saved to ../models/xgboost_classifier_model.json")