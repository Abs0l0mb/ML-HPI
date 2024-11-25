import pandas as pd
import torch
import Utils as utils
from PurityPredictionModel import PurityPredictionModel 

# Load the model structure
model = PurityPredictionModel(
    48,  # Replace with the number of unique device_serial values in your train set
    3,  # Replace with the number of unique substance_form values in your train set
    2,  # Replace with the number of unique measure_type values in your train set
    125  # Replace with the number of features in your spectrum
)

# Load the saved weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set the model to evaluation mode

file_path = '../data/test.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, False)

metadata = data.iloc[:, :3]  # Assuming first three columns are metadata
spectrum = data.iloc[:, 3:]  # All columns except target

device_serial_test_tensor = torch.tensor(metadata['device_serial'].values, dtype=torch.long)
substance_form_test_tensor = torch.tensor(metadata['substance_form_display'].values, dtype=torch.long)
measure_type_test_tensor = torch.tensor(metadata['measure_type_display'].values, dtype=torch.long)
spec_test_tensor = torch.tensor(spectrum.values, dtype=torch.float32)

# Create a DataLoader for batching (if needed)
test_dataset = torch.utils.data.TensorDataset(
    spec_test_tensor, 
    device_serial_test_tensor, 
    substance_form_test_tensor, 
    measure_type_test_tensor
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Max index in test substance_form: {metadata['measure_type_display'].max()}")
print(f"Embedding size for substance_form: {model.measure_type_embedding.num_embeddings}")


# Generate predictions
predictions = []
with torch.no_grad():
    for batch_spec, batch_device, batch_form, batch_type in test_loader:
        outputs = model((batch_spec, batch_device, batch_form, batch_type))
        predictions.extend(outputs.view(-1).cpu().numpy())

print(predictions)

# Load the sample submission file (if provided by the competition)
submission = pd.DataFrame({'ID': test_data['Id'], 'PURITY': predictions})  # Adjust 'Id' to match your test file

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

submission.to_csv("./results/predicted_purity_submission.csv", index=False)
print("Predictions saved to predicted_purity_submission.csv")
