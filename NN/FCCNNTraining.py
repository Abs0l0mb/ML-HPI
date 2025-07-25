import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import Utils as utils
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from FCCNNModel import FCCNNModel
import numpy as np

# Set random seeds for reproducibility
seed = 42  # Choose any fixed number for the seed

torch.manual_seed(seed)  # For PyTorch CPU
torch.cuda.manual_seed(seed)  # For PyTorch GPU
torch.cuda.manual_seed_all(seed)  # For all GPUs if using multiple
torch.backends.cudnn.deterministic = True  # Ensure deterministic operations in CUDA
torch.backends.cudnn.benchmark = False  # Avoid non-deterministic algorithms in CUDA

# Define worker_init_fn to ensure reproducibility with DataLoader
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)  # You can use the same seed or modify it slightly for each worker

# Load and Preprocess Data
file_path = '../data/train.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, True, True)
metadata = pd.concat([data.iloc[:, :3], data.iloc[:, -87:]], axis=1) # Assuming first three columns are metadata and last is predicted substance
spectrum = data.iloc[:, 4:].iloc[:, :-87] # All columns except target
target = data.iloc[:, 3]/100 # Get purity percentage as float
#print(metadata, target)

# Split data into train and test sets
meta_train, meta_test, spec_train, spec_test, y_train, y_test = train_test_split(
    metadata, spectrum, target, test_size=0.2, random_state=42
)

# Split training set into training and validation
meta_train, meta_val, spec_train, spec_val, y_train, y_val = train_test_split(
    meta_train, spec_train, y_train, test_size=0.2, random_state=42
)

#print(meta_train, meta_val, meta_test)

# Convert training data to PyTorch tensors
device_serial_tensor = torch.tensor(meta_train.iloc[:, 0].values, dtype=torch.long)
substance_form_tensor = torch.tensor(meta_train.iloc[:, 1].values, dtype=torch.long)
measure_type_tensor = torch.tensor(meta_train.iloc[:, 2].values, dtype=torch.long)
predicted_substance_tensor = torch.tensor(meta_train.iloc[:, -87:].values, dtype=torch.float32)
spec_train_tensor = torch.tensor(spec_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Convert validation data to PyTorch tensors
device_serial_val_tensor = torch.tensor(meta_val.iloc[:, 0].values, dtype=torch.long)
substance_form_val_tensor = torch.tensor(meta_val.iloc[:, 1].values, dtype=torch.long)
measure_type_val_tensor = torch.tensor(meta_val.iloc[:, 2].values, dtype=torch.long)
predicted_substance_val_tensor = torch.tensor(meta_val.iloc[:, -87:].values, dtype=torch.float32)
spec_val_tensor = torch.tensor(spec_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# Convert test data to PyTorch tensors
device_serial_test_tensor = torch.tensor(meta_test.iloc[:, 0].values, dtype=torch.long)
substance_form_test_tensor = torch.tensor(meta_test.iloc[:, 1].values, dtype=torch.long)
measure_type_test_tensor = torch.tensor(meta_test.iloc[:, 2].values, dtype=torch.long)
predicted_substance_test_tensor = torch.tensor(meta_test.iloc[:, -87:].values, dtype=torch.float32)
spec_test_tensor = torch.tensor(spec_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoaders for batching
train_dataset = TensorDataset(device_serial_tensor, substance_form_tensor, measure_type_tensor, predicted_substance_tensor, spec_train_tensor, y_train_tensor)
val_dataset = TensorDataset(device_serial_val_tensor, substance_form_val_tensor, measure_type_val_tensor, predicted_substance_val_tensor, spec_val_tensor, y_val_tensor)
test_dataset = TensorDataset(device_serial_test_tensor, substance_form_test_tensor, measure_type_test_tensor, predicted_substance_test_tensor, spec_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=worker_init_fn)

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

model = FCCNNModel(num_devices, num_substance_forms, num_measure_types)

# Define Loss and Optimizer
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Train Model
early_stopping = EarlyStopping(patience=50, delta=0, path='best_model.pth')

num_epochs = 1000
for epoch in range(num_epochs):
    
    # Training Phase
    train_correct_guesses = 0
    train_total_samples = 0
    correct_guesses = 0
    total_samples = 0

    model.train()
    train_loss = 0
    for batch_device, batch_form, batch_type, batch_ir_predictions, batch_spec, batch_y in train_loader:

        outputs = model((batch_spec, batch_device, batch_form, batch_type, batch_ir_predictions))
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        train_lower_bound = batch_y - 0.05
        train_upper_bound = batch_y + 0.05
        train_correct = ((outputs >= train_lower_bound) & (outputs <= train_upper_bound)).sum().item()
        train_correct_guesses += train_correct
        train_total_samples += batch_y.size(0)
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_device, batch_form, batch_type, batch_ir_predictions, batch_spec, batch_y in val_loader:
            outputs = model((batch_spec, batch_device, batch_form, batch_type, batch_ir_predictions))
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            lower_bound = batch_y - 0.05
            upper_bound = batch_y + 0.05
            correct = ((outputs >= lower_bound) & (outputs <= upper_bound)).sum().item()
            correct_guesses += correct
            total_samples += batch_y.size(0)

    # Average losses
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_accuracy = (train_correct_guesses / train_total_samples) * 100
    val_accuracy = (correct_guesses / total_samples) * 100

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy : {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
    
    # Early Stopping
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate Model
model.eval()
test_loss = 0
correct_guesses = 0
total_samples = 0
y_pred = []
y_true = []
with torch.no_grad():
    for batch_device, batch_form, batch_type, batch_ir_predictions, batch_spec, batch_y in test_loader:
        outputs = model((batch_spec, batch_device, batch_form, batch_type, batch_ir_predictions))
        y_pred.extend(outputs.view(-1).cpu().numpy())
        y_true.extend(batch_y.view(-1).cpu().numpy())

        # Calculate correct guesses within ±5%
        lower_bound = batch_y - 0.05
        upper_bound = batch_y + 0.05

        correct = ((outputs >= lower_bound) & (outputs <= upper_bound)).sum().item()
        
        correct_guesses += correct
        total_samples += batch_y.size(0)

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}")

accuracy = (correct_guesses / total_samples) * 100
print(f"Accuracy (% within ±5%): {accuracy:.2f}%")