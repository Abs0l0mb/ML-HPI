import numpy as np
import pandas as pd
import torch
from IRCNNModel import IRClassificationCNN
from sklearn.preprocessing import StandardScaler

model = IRClassificationCNN(87)
model.load_state_dict(torch.load('ir_classification_cnn.pth'))

model.eval()

data = pd.read_csv('../data/test.csv')
data = data.iloc[:, -125:]

scaler = StandardScaler()
ir_data_scaled = scaler.fit_transform(data.astype(np.float32))
ir_data_scaled = torch.tensor(ir_data_scaled)


with torch.no_grad():
    predictions = model(ir_data_scaled)
    result = pd.DataFrame(predictions)

result.to_csv('../data/test_predictions_substances.csv')
