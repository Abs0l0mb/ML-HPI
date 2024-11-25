import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

def pre_process_data(file_path: str, transform_spectrum: bool, add_substance_classif: bool):
    """
    Preprocess the dataset:
    - Drop 'sample_name' and 'prod_substance' columns.
    - Convert string keys in 'device_serial', 'substance_form_display', and 'measure_type_display' to numeric values.
    - (Optionnal) transforms the spectrum using savgol filter and zscore 

    Args:
        file_path (str): Path to the CSV file.
        transform_spectrum (bool): Indicates whether or not the spectrum should be transformed using procedure given in the subject
        add_substance_classif (bool): Add the substance classification feature 
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """

    df = pd.read_csv(file_path)

    # Drop columns
    columns_to_drop = ['sample_name', 'prod_substance']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Label encoder
    string_columns = ['device_serial', 'substance_form_display', 'measure_type_display']
    for col in string_columns:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    # Tranform spectrum
    if transform_spectrum:
        spectrum = df.iloc[:, 4:]
        spectrum_filtered_standardized = savgol(spectrum)
        df = pd.concat([df.iloc[:, :4], spectrum_filtered_standardized], axis=1)

    if add_substance_classif:
        substance_model = XGBClassifier()
        substance_model.load_model('../models/xgboost_classifier_model.json')
        spectrum_data = df.iloc[:, 4:] 
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(spectrum_data)
        predictions = substance_model.predict(scaled_data)
        df['substance_class'] = predictions

    return df

def savgol(spectrum):
    spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0), columns=spectrum.columns)
    spectrum_filtered_standardized = pd.DataFrame(zscore(spectrum_filtered, axis = 1), columns=spectrum.columns)
    return spectrum_filtered_standardized

def purity_score(y_true, y_pred):
    within_5_percent = np.abs(y_true - y_pred) <= 5
    return np.mean(within_5_percent)