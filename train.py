import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy.signal import savgol_filter
import numpy as np
import joblib

def purity_score(y_true, y_pred):
    within_5_percent = np.abs(y_true - y_pred) <= 5
    return np.mean(within_5_percent)

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
    spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0), columns=spectrum.columns)
    spectrum_filtered_standardized = pd.DataFrame(zscore(spectrum_filtered, axis = 1), columns=spectrum.columns)
    combined_df = pd.concat([df.iloc[:, :4], spectrum_filtered_standardized], axis=1)
    return df#combined_df


# Load data
data = pre_process_data("./data/train.csv")
print(data)
X = data.drop(columns=['PURITY'])
y = data['PURITY']    # Target variable (purity)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model with the optimal hyperparameters
best_rf_model = RandomForestRegressor(
    n_estimators=394,
    max_depth=25,
    min_samples_split=2,
    random_state=42
)
best_rf_model.fit(X_train, y_train)

y_pred = best_rf_model.predict(X_test)

# Save the trained PCA and model for future use
joblib.dump(best_rf_model, "./random_forest_model.joblib")
final_score = purity_score(y_test, y_pred)
print(f"Final Optimized Random Forest Custom Purity Score (within ±5% range): {final_score:.4f}")
