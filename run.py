import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from scipy.signal import savgol_filter

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

# Load the PCA and trained model
best_rf_model = joblib.load("./random_forest_model.joblib")

# Load data
test_data = pd.read_csv("./data/test.csv")
X_test_data = pre_process_data("./data/test.csv")
print(X_test_data)

# Predict purity for the test data
predicted_purity = best_rf_model.predict(X_test_data)

# Create the submission DataFrame with the required structure
submission = pd.DataFrame({
    "ID": test_data.index + 1,  # Assuming IDs are 1-indexed; adjust if necessary
    "PURITY": predicted_purity
})

# Save the predictions to CSV in the desired format
submission.to_csv("./results/predicted_purity_submission.csv", index=False)
print("Predictions saved to predicted_purity_submission.csv")
