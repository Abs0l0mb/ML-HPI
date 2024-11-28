import pickle
import pandas as pd
import torch
import Utils as utils
from NN.CNNModel import SpectrumPredictionModel

# Load the model structure
model = SpectrumPredictionModel(
    125  # Replace with the number of features in your spectrum
)

# Load the saved weights
model.load_state_dict(torch.load('best_spectrum_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess test data
file_path = '../data/test.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, False, False)

spectrum = data.iloc[:, 3:]  # All columns except target

# Convert data to tensors
spec_test_tensor = torch.tensor(spectrum.values, dtype=torch.float32)

# Run the entire test data through the model
with torch.no_grad():
    predictions = model((spec_test_tensor))
    predictions = predictions.view(-1).cpu().numpy()  # Convert to a NumPy array

# Load the sample submission file (if provided by the competition)
submission = pd.DataFrame({'ID': data.index+1, 'PURITY': predictions*100})  # Adjust 'Id' to match your test file

# Save to CSV
submission.to_csv('spectrum_submission.csv', index=False)
print("Submission saved to submission.csv")
