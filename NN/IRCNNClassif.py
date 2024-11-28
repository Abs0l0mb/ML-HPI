import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from IRCNNModel import IRClassificationCNN

# Load the dataset
file_path = '../data/substances.csv'
data = pd.read_csv(file_path)

# Preprocess the data
# Separate features and target
target = data.iloc[:, 0]  # First column is the target (substance name)
ir_data = data.iloc[:, 1:]  # Remaining columns are IR data

# Encode the target labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Standardize the IR data
scaler = StandardScaler()
ir_data_scaled = scaler.fit_transform(ir_data)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    ir_data_scaled, target_encoded, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
num_classes = len(label_encoder.classes_)
model = IRClassificationCNN(input_size=125, num_classes=num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), 'ir_classification_cnn.pth')
print("Model saved to 'ir_classification_cnn.pth'")
