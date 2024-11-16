import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import zscore

def pre_process_data(file_path):
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

    # Extract spectrum and perform processing
    spectrum = df.iloc[:, 4:]
    spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv=2, axis=0))
    spectrum_filtered_standardized = pd.DataFrame(zscore(spectrum_filtered, axis=1))

    # Combine metadata and spectrum data
    metadata = df.iloc[:, :3]
    print(metadata)
    return metadata, spectrum, spectrum_filtered_standardized, df['PURITY']

class CustomLoss(nn.Module):
    def __init__(self, tolerance=0.05):
        super(CustomLoss, self).__init__()
        self.tolerance = tolerance  # Define the tolerance for ±5%

    def forward(self, predictions, targets):
        """
        Forward pass of the loss function.
        Computes the penalty for predictions outside the ±tolerance range.
        """
        lower_bound = targets * (1 - self.tolerance)
        upper_bound = targets * (1 + self.tolerance)
        
        # Calculate penalties for being out of bounds
        below_bound_penalty = abs(lower_bound - predictions)  # Predictions too low
        above_bound_penalty = abs(predictions - upper_bound)  # Predictions too high
        
        # Total penalty is the sum of both
        total_penalty = below_bound_penalty + above_bound_penalty
        
        # Mean penalty across the batch
        return total_penalty.mean()

# Load the dataset
file_path = './data/train.csv'  # Adjust path if needed
# Preprocess data
metadata, spectrum, spectrum_filtered, target = pre_process_data(file_path)

# Convert to NumPy arrays
metadata = metadata.values
spectrum = spectrum.values
spectrum_filtered = spectrum_filtered.values
target = target.values

# Split data into training and testing sets
meta_train, meta_test, spec_train, spec_test, spec_fil_train, spec_fil_test, y_train, y_test = train_test_split(
    metadata, spectrum, spectrum_filtered, target, test_size=0.2, random_state=42
)

# Standardize metadata
scaler = StandardScaler()
meta_train = scaler.fit_transform(meta_train)
meta_test = scaler.transform(meta_test)

# Convert to PyTorch tensors
meta_train_tensor = torch.tensor(meta_train, dtype=torch.float32)
meta_test_tensor = torch.tensor(meta_test, dtype=torch.float32)
spec_train_tensor = torch.tensor(spec_train, dtype=torch.float32)
spec_test_tensor = torch.tensor(spec_test, dtype=torch.float32)
spec_fil_train_tensor = torch.tensor(spec_fil_train, dtype=torch.float32)
spec_fil_test_tensor = torch.tensor(spec_fil_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batching
train_dataset = TensorDataset(meta_train_tensor, spec_train_tensor, spec_fil_train_tensor, y_train_tensor)
test_dataset = TensorDataset(meta_test_tensor, spec_test_tensor, spec_fil_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ComplexNN(nn.Module):
    def __init__(self, spectrum_input_size, spectrum_fil_size, metadata_input_size):
        super(ComplexNN, self).__init__()
        
        # Spectrum branch (using convolutional layers)
        self.spectrum_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate output size after convolution and pooling
        conv_output_size = spectrum_input_size // 4  # Adjust based on the number of pooling layers
        
        # Spectrum branch (using convolutional layers)
        self.spectrum_fil_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # Calculate output size after convolution and pooling
        conv_output_size = spectrum_fil_size // 4  # Adjust based on the number of pooling layers

        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(metadata_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(conv_output_size * 32 + conv_output_size * 32 + 32, 128),  # Combining spectrum and metadata features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for predicting PURITY
        )

    def forward(self, spectrum_data, spectrum_fil_data, metadata_data):
        # Forward pass through spectrum branch
        spectrum_data = spectrum_data.unsqueeze(1)  # Add channel dimension for Conv1d
        spectrum_features = self.spectrum_branch(spectrum_data)
        
        spectrum_fil_data = spectrum_fil_data.unsqueeze(1)  # Add channel dimension for Conv1d
        spectrum_fil_features = self.spectrum_fil_branch(spectrum_data)

        # Forward pass through metadata branch
        metadata_features = self.metadata_branch(metadata_data)
        
        # Concatenate features from both branches
        combined_features = torch.cat((spectrum_features, spectrum_fil_features, metadata_features), dim=1)
        
        # Final prediction
        output = self.fusion_layer(combined_features)
        return output

# Initialize model, loss, and optimizer

# Sizes of metadata and spectrum data
meta_data_size = meta_train_tensor.shape[1]  # Number of features in metadata
spectrum_size = spec_train_tensor.shape[1]  # Number of features in spectrum data
spectrum_fil_size = spec_fil_train_tensor.shape[1]  # Number of features in spectrum data

# Initialize the ComplexNN model
model = ComplexNN(spectrum_size, spectrum_fil_size, meta_data_size)

# Custom loss function
criterion = CustomLoss(tolerance=0.05)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_meta, batch_spec, batch_spec_fil, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_spec, batch_spec_fil, batch_meta)
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
    for batch_meta, batch_spec, batch_spec_fil, batch_y in test_loader:
        # Forward pass
        outputs = model(batch_spec, batch_spec_fil, batch_meta)
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
print(f"Test Loss (Fraction of Failures): {test_loss / len(test_loader):.4f}")
print(f"Accuracy (% within ±5%): {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), './purity_prediction_model.pth')
print("Model saved to ./purity_prediction_model.pth")
