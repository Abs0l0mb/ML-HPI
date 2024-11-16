import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer
import numpy as np

# Load data
data = pd.read_csv("./data/train.csv")

# Select spectral features and target (purity)
X = data.iloc[:, 6:]  # Spectral data columns
y = data['PURITY']    # Target variable (purity)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom scoring function
def purity_score(y_true, y_pred):
    # Calculate the fraction of predictions within ±5% of actual purity values
    within_5_percent = np.abs(y_true - y_pred) <= 5
    return np.mean(within_5_percent)

# Make scorer for cross-validation
custom_scorer = make_scorer(purity_score, greater_is_better=True)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance on test data using custom scoring
y_pred = model.predict(X_test)
score = purity_score(y_test, y_pred)

print(f"Custom Purity Score (within ±5% range): {score:.4f}")

# Optionally, cross-validate the model using custom scorer
cv_score = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
print(f"Cross-Validation Custom Purity Score (±5% range): {cv_score.mean():.4f}")
