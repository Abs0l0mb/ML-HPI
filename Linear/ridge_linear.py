import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer
from scipy.stats import zscore
from scipy.signal import savgol_filter

# Load and preprocess the dataset
data = pd.read_csv("../data/train.csv")

# Extract spectrum data (columns from index 6 onward)
spectrum = data.iloc[:, 6:]

# Apply Savitzky-Golay filter for preprocessing
spectrum_filtered_array = savgol_filter(spectrum, 7, 3, deriv=2, axis=1)

# Ensure the shape matches the original spectrum
spectrum_filtered = pd.DataFrame(spectrum_filtered_array, columns=spectrum.columns[:spectrum_filtered_array.shape[1]])

# Standardize the filtered spectrum
spectrum_standardized = pd.DataFrame(zscore(spectrum_filtered, axis=1), columns=spectrum_filtered.columns)

# Feature selection: Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.01)
spectrum_selected = var_thresh.fit_transform(spectrum_standardized)
spectrum_selected = pd.DataFrame(spectrum_selected)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
spectrum_poly = poly.fit_transform(spectrum_selected)
spectrum_poly = pd.DataFrame(spectrum_poly)

# Define the target variable
y = data['PURITY']

# Define a custom reliability scoring function for ±5% tolerance
def reliability_score(y_true, y_pred, tolerance=5):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance)

reliability_scorer = make_scorer(reliability_score, greater_is_better=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(spectrum_poly, y, test_size=0.2, random_state=42)

# Define the model pipeline with Ridge Regression
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 20)}

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features
    ('ridge', Ridge())
])

# Perform grid search with cross-validation to find the best alpha
grid_search = GridSearchCV(pipeline, {'ridge__alpha': param_grid['alpha']}, cv=5, scoring=reliability_scorer)
grid_search.fit(X_train, y_train)

# Display best parameters and best score
print("Best alpha found:", grid_search.best_params_['ridge__alpha'])
print("Best cross-validation reliability score:", grid_search.best_score_)

# Test the model on the test set
final_model = grid_search.best_estimator_
y_test_pred = final_model.predict(X_test)

# Calculate reliability score on the test set for ±5% tolerance
test_score_5 = reliability_score(y_test, y_test_pred, tolerance=5)

print("Test Reliability Score (Fraction within ±5% tolerance):", test_score_5)

# Load and preprocess the test dataset for predictions
test_data = pd.read_csv("../data/test.csv")
test_spectrum = test_data.iloc[:, 5:]
test_spectrum_filtered_array = savgol_filter(test_spectrum, 7, 3, deriv=2, axis=1)

# Ensure the shape matches the original spectrum
test_spectrum_filtered = pd.DataFrame(test_spectrum_filtered_array, columns=spectrum.columns[:test_spectrum_filtered_array.shape[1]])

# Standardize and transform the test spectrum
test_spectrum_standardized = pd.DataFrame(zscore(test_spectrum_filtered, axis=1))
test_spectrum_selected = var_thresh.transform(test_spectrum_standardized)
test_spectrum_poly = poly.transform(test_spectrum_selected)

# Make predictions using the trained model
test_predictions = final_model.predict(test_spectrum_poly)

# Create an output DataFrame in the required format
output = pd.DataFrame({
    "ID": test_data.index + 1,  # Assuming IDs are sequential starting from 1
    "PURITY": test_predictions
})

# Save predictions to a CSV file
output.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")
