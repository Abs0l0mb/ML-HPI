import pickle
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IRCNNModel import IRClassificationCNN

def pre_process_data(file_path: str, transform_spectrum: bool, add_substance_classif: bool, save_encoders: bool, encoders: None | object = None):
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
    if(encoders == None):    
        encoders = {
            'device_serial': LabelEncoder(),
            'substance_form_display': LabelEncoder(),
            'measure_type_display': LabelEncoder()
        }

        encoders['device_serial'].fit(df['device_serial'])
        encoders['substance_form_display'].fit(df['substance_form_display'])
        encoders['measure_type_display'].fit(df['measure_type_display'])

    df['device_serial'] = safe_transform(df['device_serial'], encoders['device_serial'])
    df['substance_form_display'] = safe_transform(df['substance_form_display'], encoders['substance_form_display'])
    df['measure_type_display'] = safe_transform(df['measure_type_display'], encoders['measure_type_display'])

    # Save the encoders to a file
    if(save_encoders):
        with open('encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)

    print("Encoders saved to encoders.pkl")

    # Tranform spectrum
    if transform_spectrum:
        spectrum = df.iloc[:, 4:]
        spectrum_filtered_standardized = savgol(spectrum)
        df = pd.concat([df.iloc[:, :4], spectrum_filtered_standardized], axis=1)

    if add_substance_classif:

        df_substances = pd.read_csv('../data/substances.csv')
        num_substance_classes = df_substances.iloc[:, 0].nunique()

        substance_model = IRClassificationCNN(num_substance_classes)
        substance_model.load_state_dict(torch.load("ir_classification_cnn.pth"))
        substance_model.eval()

        scaler = StandardScaler()
        ir_data_scaled = scaler.fit_transform(df.iloc[:, 4:].astype(np.float32))
        ir_data_scaled = torch.tensor(ir_data_scaled)

        with torch.no_grad():
            predictions = pd.read_csv("../data/predictions_substances.csv")
            #predictions = pd.read_csv("../data/test_predictions_substances.csv")
            predictions = predictions.drop(columns=['index'])
            df = pd.concat([df, predictions], axis=1)

    return df

def safe_transform(column, encoder, default_value=0):
    """Safely transform a column using a LabelEncoder, mapping unseen values to a default."""
    known_classes = set(encoder.classes_)  # Get the known classes from the encoder
    column_mapped = column.apply(lambda x: x if x in known_classes else encoder.classes_[0])  # Map unseen labels
    return encoder.transform(column_mapped)

def savgol(spectrum):
    spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0), columns=spectrum.columns)
    spectrum_filtered_standardized = pd.DataFrame(zscore(spectrum_filtered, axis = 1), columns=spectrum.columns)
    return spectrum_filtered_standardized

def purity_score(y_true, y_pred):
    within_5_percent = np.abs(y_true - y_pred) <= 5
    return np.mean(within_5_percent)