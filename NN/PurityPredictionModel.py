import torch
import torch.nn as nn

class PurityPredictionModel(nn.Module):
    def __init__(self, num_devices, num_substance_forms, num_measure_types, spectrum_input_size):
        super(PurityPredictionModel, self).__init__()
        
        # Spectrum Module
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        # Metadata Embedding
        self.device_embedding = nn.Embedding(num_devices, 8)
        self.substance_form_embedding = nn.Embedding(num_substance_forms, 8)
        self.measure_type_embedding = nn.Embedding(num_measure_types, 8)
        
        # Metadata Fully Connected
        self.metadata_fc = nn.Sequential(
            nn.Linear(8 * 3, 16),  # 8 embedding size * 3 metadata features
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fc = nn.Sequential(
            nn.Linear(32 * (spectrum_input_size // 4) + 16, 64),  # Adjust dynamically
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Regression output
        )
    
    def forward(self, inputs):

        spectrum, device_serial, substance_form, measure_type = inputs
        
        # Spectrum through CNN
        spectrum = spectrum.unsqueeze(1)  # Add channel dimension
        spectrum_features = self.cnn(spectrum)
        
        # Metadata through embeddings
        device_embed = self.device_embedding(device_serial)
        substance_form_embed = self.substance_form_embedding(substance_form)
        measure_type_embed = self.measure_type_embedding(measure_type)
        
        # Combine metadata embeddings
        metadata = torch.cat([device_embed, substance_form_embed, measure_type_embed], dim=1)
        metadata_features = self.metadata_fc(metadata)
        
        # Fusion
        combined_features = torch.cat([spectrum_features, metadata_features], dim=1)
        output = self.fc(combined_features)
        return output
