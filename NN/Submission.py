import pickle
import pandas as pd
import torch
import Utils as utils
from FCCNNModel import FCCNNModel 

# Load the model structure
model = FCCNNModel(
    48,  # Replace with the number of unique device_serial values in your train set
    3,  # Replace with the number of unique substance_form values in your train set
    2,  # Replace with the number of unique measure_type values in your train set
    87,
    125  # Replace with the number of features in your spectrum
)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

print("Encoders loaded from encoders.pkl")

# Load the saved weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess test data
file_path = '../data/test.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, True, False, encoders)

# Split metadata and spectrum
metadata = pd.concat([data.iloc[:, :3], data.iloc[:, -1]], axis=1)  # Assuming first three columns are metadata and last is predicted substance
spectrum = data.iloc[:, 3:]  # All columns except target

# Convert data to tensors
device_serial_test_tensor = torch.tensor(metadata['device_serial'].values, dtype=torch.long)
substance_form_test_tensor = torch.tensor(metadata['substance_form_display'].values, dtype=torch.long)
measure_type_test_tensor = torch.tensor(metadata['measure_type_display'].values, dtype=torch.long)
predicted_substance_test_tensor = torch.tensor(metadata['predicted_substance'].values, dtype=torch.long)
spec_test_tensor = torch.tensor(spectrum.values, dtype=torch.float32)

# Print debug information
print(f"Max index in test measure_type: {metadata['measure_type_display'].max()}")
print(f"Embedding size for measure_type: {model.measure_type_embedding.num_embeddings}")
print(f"Max index in test device_serial: {metadata['device_serial'].max()}")
print(f"Embedding size for device_serial: {model.device_embedding.num_embeddings}")
print(f"Max index in test substance_form_display: {metadata['substance_form_display'].max()}")
print(f"Embedding size for substance_form_display: {model.substance_form_embedding.num_embeddings}")

max_train_index_device = model.device_embedding.num_embeddings - 1
max_train_index_form = model.substance_form_embedding.num_embeddings - 1
max_train_index_measure = model.measure_type_embedding.num_embeddings - 1

# Assign unseen classes to default value
metadata['device_serial'] = metadata['device_serial'].apply(
    lambda x: x if x <= max_train_index_device else 0
)
metadata['substance_form_display'] = metadata['substance_form_display'].apply(
    lambda x: x if x <= max_train_index_form else 0
)
metadata['measure_type_display'] = metadata['measure_type_display'].apply(
    lambda x: x if x <= max_train_index_measure else 0
)

# Run the entire test data through the model
with torch.no_grad():
    predictions = model((spec_test_tensor, device_serial_test_tensor, substance_form_test_tensor, measure_type_test_tensor, predicted_substance_test_tensor))
    predictions = predictions.view(-1).cpu().numpy()  # Convert to a NumPy array

# Load the sample submission file (if provided by the competition)
submission = pd.DataFrame({'ID': data.index+1, 'PURITY': predictions*100})  # Adjust 'Id' to match your test file

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")