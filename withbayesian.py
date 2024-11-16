import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter
from scipy.stats import zscore

def pre_process_data(file_path):
    """
    Preprocess the dataset:
    - Drop 'sample_name' and 'prod_substance' columns.
    - Convert string keys in 'device_serial', 'substance_form_display', and 'measure_type_display' to numeric values.
    
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    columns_to_drop = ['sample_name', 'prod_substance']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Encode string columns to numeric values
    string_columns = ['device_serial', 'substance_form_display', 'measure_type_display']
    for col in string_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    spectrum = df.iloc[:, 4:]
    spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
    spectrum_filtered_standardized = pd.DataFrame(zscore(spectrum_filtered, axis = 1))

    combined_df = pd.concat([df.iloc[:, :4], spectrum_filtered_standardized], axis=1)
    return df#combined_df

# Load data
data = pre_process_data("./data/train.csv")
print(data)
X = data.drop(columns=['PURITY'])
y = data['PURITY']    # Target variable (purity)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the custom scoring function
def purity_score(y_true, y_pred):
    within_5_percent = np.abs(y_true - y_pred) <= 5
    return np.mean(within_5_percent)

# Custom scorer
custom_scorer = make_scorer(purity_score, greater_is_better=True)

'''
# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)  # Example number; adjust based on prior tuning. TWEAK 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
'''

# Optuna objective function
def objective(trial):
    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    # Define and train Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Cross-validate and return custom purity score
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer).mean()
    return cv_score

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Get best hyperparameters and train final model
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the best model
best_rf_model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42
)
best_rf_model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = best_rf_model.predict(X_test)
print(y_pred)
final_score = purity_score(y_test, y_pred)
print(f"Final Optimized Random Forest Custom Purity Score (within ±5% range): {final_score:.4f}")
