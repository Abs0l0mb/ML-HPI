import torch.nn as nn

class SpectrumPredictionModel(nn.Module):
    def __init__(self, spectrum_input_size):
        super(SpectrumPredictionModel, self).__init__()
        
        # Spectrum Module
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),

            nn.Linear(992, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


        self.fc = nn.Sequential(
            nn.Linear(125, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, inputs):

        spectrum = inputs
        
        # Spectrum through CNN
        spectrum = spectrum.unsqueeze(1)  # Add channel dimension
        output = self.cnn(spectrum)
        return output
