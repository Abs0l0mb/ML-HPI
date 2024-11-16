import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from scipy.stats import zscore
import torch.nn as nn
import torch.optim as optim

# Preprocessing function
def pre_process_data(file_path):
    """
    Preprocess the dataset:
    - Perform spectrum analysis using Savitzky-Golay filter and standardize it.
    - Use only the spectrum for the neural network input.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed spectral data.
        pd.Series: Target values (PURITY).
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    print(df)
    # Extract target column
    target = df['PURITY']

    # Perform spectrum analysis
    spectrum = df.iloc[:, 6:].values  # Extract spectrum data
    spectrum_filtered = savgol_filter(spectrum, window_length=7, polyorder=3, deriv=2, axis=0)  # Apply Savitzky-Golay filter
    spectrum_filtered_standardized = zscore(spectrum_filtered, axis=1)  # Standardize spectrum data
    
    return pd.DataFrame(spectrum_filtered_standardized), target

# Load and preprocess the dataset
file_path = './data/train.csv'  # Adjust path if needed
X, y = pre_process_data(file_path)

# Convert target to NumPy array
y = y.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Neural Network
class SpectrumNN(nn.Module):
    def __init__(self, input_size):
        super(SpectrumNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
input_size = X_train.shape[1]
model = SpectrumNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
test_loss = 0
correct_guesses = 0
total_samples = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()
        
        # Calculate correct guesses within ±5%
        tolerance = 0.05
        lower_bound = batch_y * (1 - tolerance)
        upper_bound = batch_y * (1 + tolerance)
        correct = ((outputs >= lower_bound) & (outputs <= upper_bound)).sum().item()
        
        correct_guesses += correct
        total_samples += batch_y.size(0)

# Calculate accuracy
accuracy = (correct_guesses / total_samples) * 100
print(f"Test Loss (MSE): {test_loss / len(test_loader):.4f}")
print(f"Accuracy (% within ±5%): {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), '/mnt/data/spectral_nn_model.pth')
print("Model saved to /mnt/data/spectral_nn_model.pth")
