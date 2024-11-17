import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from category_encoders import TargetEncoder
import pywt
from umap import UMAP
from scipy.stats import zscore

# Load and preprocess the dataset
data = pd.read_csv('./data/train.csv')
data.columns = data.columns.astype(str)

# Separate categorical label columns and spectral data
categorical_columns = data.columns[:6]
spectrum = data.iloc[:, 6:]

# Apply wavelet transform to reduce noise
def wavelet_denoise(data, wavelet="db1", level=1):
    denoised_data = []
    for i in range(data.shape[1]):
        coeffs = pywt.wavedec(data.iloc[:, i], wavelet, level=level)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # Zero out high-frequency components
        denoised_signal = pywt.waverec(coeffs, wavelet)
        denoised_data.append(denoised_signal)
    return pd.DataFrame(np.array(denoised_data).T, columns=data.columns)

spectrum_denoised = wavelet_denoise(spectrum, wavelet="db1", level=1)
spectrum_standardized = pd.DataFrame(zscore(spectrum_denoised, axis=1))
spectrum_standardized.columns = spectrum_standardized.columns.astype(str)

# Combine preprocessed spectral data with categorical labels
X = pd.concat([data[categorical_columns], spectrum_standardized], axis=1)
y = data['PURITY']

# Apply Target Encoding for categorical columns
target_encoder = TargetEncoder(cols=categorical_columns)
X_encoded = target_encoder.fit_transform(X, y)

# Use UMAP for dimensionality reduction on the spectral data only
umap = UMAP(n_components=50, random_state=42)
X_umap = umap.fit_transform(spectrum_standardized)
X_umap = pd.DataFrame(X_umap, columns=[f'UMAP_{i}' for i in range(X_umap.shape[1])])

# Combine UMAP-reduced spectral data with encoded categorical features
X_final = pd.concat([X_encoded[categorical_columns].reset_index(drop=True), X_umap], axis=1)

# Define a custom reliability scoring function for ±5% tolerance
def reliability_score(y_true, y_pred, tolerance=5):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance)

reliability_scorer = make_scorer(reliability_score, greater_is_better=True)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Define the model pipeline with Ridge Regression
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

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
