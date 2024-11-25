import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import Utils as utils
from PurityPredictionModel import PurityPredictionModel

# Load and Preprocess Data
file_path = '../data/train.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, False)

metadata = data.iloc[:, :3]  # Assuming first three columns are metadata
spectrum = data.iloc[:, 4:]  # All columns except target
target = data.iloc[:, 4] 


# Split data into train and test sets
meta_train, meta_test, spec_train, spec_test, y_train, y_test = train_test_split(
    metadata, spectrum, target, test_size=0.2#, random_state=42
)

# Split training set into training and validation
meta_train, meta_val, spec_train, spec_val, y_train, y_val = train_test_split(
    meta_train, spec_train, y_train, test_size=0.2#, random_state=42
)

print(meta_train, meta_val, meta_test)

# Convert data to PyTorch tensors
device_serial_tensor = torch.tensor(meta_train.iloc[:, 0].values, dtype=torch.long)
substance_form_tensor = torch.tensor(meta_train.iloc[:, 1].values, dtype=torch.long)
measure_type_tensor = torch.tensor(meta_train.iloc[:, 2].values, dtype=torch.long)
device_serial_val_tensor = torch.tensor(meta_val.iloc[:, 0].values, dtype=torch.long)
substance_form_val_tensor = torch.tensor(meta_val.iloc[:, 1].values, dtype=torch.long)
measure_type_val_tensor = torch.tensor(meta_val.iloc[:, 2].values, dtype=torch.long)
spec_val_tensor = torch.tensor(spec_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

spec_train_tensor = torch.tensor(spec_train.values, dtype=torch.float32)
spec_test_tensor = torch.tensor(spec_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Test tensors for metadata
device_serial_test_tensor = torch.tensor(meta_test.iloc[:, 0].values, dtype=torch.long)
substance_form_test_tensor = torch.tensor(meta_test.iloc[:, 1].values, dtype=torch.long)
measure_type_test_tensor = torch.tensor(meta_test.iloc[:, 2].values, dtype=torch.long)

# Create DataLoaders for batching
train_dataset = TensorDataset(device_serial_tensor, substance_form_tensor, measure_type_tensor, spec_train_tensor, y_train_tensor)
val_dataset = TensorDataset(device_serial_val_tensor, substance_form_val_tensor, measure_type_val_tensor, spec_val_tensor, y_val_tensor)
test_dataset = TensorDataset(device_serial_test_tensor, substance_form_test_tensor, measure_type_test_tensor, spec_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after the last time validation loss improved.
            delta (float): Minimum change in validation loss to qualify as an improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # Save the best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Initialize Model
num_devices = metadata.iloc[:, 0].nunique()
num_substance_forms = metadata.iloc[:, 1].nunique()
num_measure_types = metadata.iloc[:, 2].nunique()
spectrum_input_size = spec_train.shape[1]

print(num_devices, num_substance_forms, num_measure_types, spectrum_input_size)

model = PurityPredictionModel(num_devices, num_substance_forms, num_measure_types, spectrum_input_size)

# Define Loss and Optimizer
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Train Model
early_stopping = EarlyStopping(patience=1000, delta=0, path='best_model.pth')

num_epochs = 1000
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_loss = 0
    for batch_device, batch_form, batch_type, batch_spec, batch_y in train_loader:
        outputs = model((batch_spec, batch_device, batch_form, batch_type))
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_device, batch_form, batch_type, batch_spec, batch_y in val_loader:
            outputs = model((batch_spec, batch_device, batch_form, batch_type))
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    # Average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early Stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))




def shap():
    # Prepare data for SHAP
    model.eval()
    def model_predict(inputs):
        spectrum, device_serial, substance_form, measure_type = inputs
        with torch.no_grad():
            return model((spectrum, device_serial, substance_form, measure_type)).cpu().numpy()

    test_input = (
        spec_test_tensor,
        device_serial_test_tensor, 
        substance_form_test_tensor, 
        measure_type_test_tensor
    )

    combined_test_input = torch.cat([
        spec_test_tensor,
        device_serial_test_tensor.unsqueeze(1).float(),
        substance_form_test_tensor.unsqueeze(1).float(),
        measure_type_test_tensor.unsqueeze(1).float()
    ], dim=1)

    explainer = shap.GradientExplainer(model, test_input)
    shap_values = explainer.shap_values(test_input)
    shap.summary_plot(shap_values, test_input)







# Evaluate Model
model.eval()
test_loss = 0
correct_guesses = 0
total_samples = 0
y_pred = []
y_true = []
with torch.no_grad():
    for batch_device, batch_form, batch_type, batch_spec, batch_y in test_loader:
        outputs = model((batch_spec, batch_device, batch_form, batch_type))
        y_pred.extend(outputs.view(-1).cpu().numpy())
        y_true.extend(batch_y.view(-1).cpu().numpy())

        # Calculate correct guesses within ±5%
        tolerance = 0.05
        lower_bound = batch_y * (1 - tolerance)
        upper_bound = batch_y * (1 + tolerance)
        correct = ((outputs >= lower_bound) & (outputs <= upper_bound)).sum().item()
        
        correct_guesses += correct
        total_samples += batch_y.size(0)

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}")

accuracy = (correct_guesses / total_samples) * 100
print(f"Accuracy (% within ±5%): {accuracy:.2f}%")