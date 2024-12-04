import torch
import torch.nn as nn

class IRClassificationCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IRClassificationCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        cnn_output_size = 960  # Calculate the flattened output size of the CNN
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        return x