import torch
import torch.nn as nn

class FCCNNModel(nn.Module):
    def __init__(self, num_devices, num_substance_forms, num_measure_types):
        super(FCCNNModel, self).__init__()
        
        # Spectrum Module
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
        )
        
        # Metadata Embedding
        self.device_embedding = nn.Embedding(num_devices, 8)
        self.substance_form_embedding = nn.Embedding(num_substance_forms, 1)
        self.measure_type_embedding = nn.Embedding(num_measure_types, 1)
        self.ir_dim_reduc = nn.Sequential(
            nn.Linear(87, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Metadata Fully Connected
        self.metadata_fc = nn.Sequential(
            nn.Linear(8 + 1 + 1 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Metadata Projection to Match Spectrum Features
        self.metadata_projection = nn.Linear(128, 256)

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(1504, 128),  # Adjust for concatenated input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, inputs):
        spectrum, device_serial, substance_form, measure_type, ir_predictions = inputs
        
        # Spectrum through CNN
        spectrum = spectrum.unsqueeze(1)  # Add channel dimension
        spectrum_features = self.cnn(spectrum)  # Output size: (B, 992)
        
        # Metadata through embeddings
        device_embed = self.device_embedding(device_serial)
        substance_form_embed = self.substance_form_embedding(substance_form)
        measure_type_embed = self.measure_type_embedding(measure_type)
        ir_reduced = self.ir_dim_reduc(ir_predictions)

        # Combine metadata embeddings
        metadata = torch.cat([device_embed, substance_form_embed, measure_type_embed, ir_reduced], dim=1)
        metadata_features = self.metadata_fc(metadata)  # Output size: (B, 16)
        #metadata_features = self.metadata_projection(metadata_features)  # Output size: (B, 960)
        
        # Concatenate features
        combined_features = torch.cat([spectrum_features, metadata_features], dim=1)  # Output size: (B, 1920)
        output = self.fc(combined_features)
        return output