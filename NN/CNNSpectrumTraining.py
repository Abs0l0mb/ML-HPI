import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import Utils as utils
from NN.CNNModel import SpectrumPredictionModel

# Load and Preprocess Data
file_path = '../data/train.csv'  # Adjust path if necessary
data = utils.pre_process_data(file_path, False, False, False)

print(data)

spectrum = data.iloc[:, 4:]  # All columns except target
target = data.iloc[:, 3] / 100

print(target, spectrum)

# Split data into train and test sets
spec_train, spec_test, y_train, y_test = train_test_split(
    spectrum, target, test_size=0.2, random_state=42
)

# Split training set into training and validation
spec_train, spec_val, y_train, y_val = train_test_split(
    spec_train, y_train, test_size=0.2#, random_state=42
)

# Convert data to PyTorch tensors
spec_train_tensor = torch.tensor(spec_train.values, dtype=torch.float32)
spec_val_tensor = torch.tensor(spec_val.values, dtype=torch.float32)
spec_test_tensor = torch.tensor(spec_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoaders for batching
train_dataset = TensorDataset(spec_train_tensor, y_train_tensor)
val_dataset = TensorDataset(spec_val_tensor, y_val_tensor)
test_dataset = TensorDataset(spec_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='best_spectrum_model.pth'):
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
spectrum_input_size = spec_train.shape[1]

model = SpectrumPredictionModel(spectrum_input_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train Model
early_stopping = EarlyStopping(patience=50, delta=0, path='best_spectrum_model.pth')

num_epochs = 1000
for epoch in range(num_epochs):
    # Training Phase
    model.train()
    train_loss = 0
    for batch_spec, batch_y in train_loader:
        outputs = model((batch_spec))
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_spec, batch_y in val_loader:
            outputs = model((batch_spec))
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
model.load_state_dict(torch.load('best_spectrum_model.pth'))

# Evaluate Model
model.eval()
test_loss = 0
correct_guesses = 0
total_samples = 0
y_pred = []
y_true = []
with torch.no_grad():
    for batch_spec, batch_y in test_loader:
        outputs = model((batch_spec))
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