import pandas as pd
import joblib

# Load the PCA and trained model
pca = joblib.load("./results/pca_model.joblib")
best_rf_model = joblib.load("./results/random_forest_model.joblib")

# Load the test data
test_data = pd.read_csv("./data/test.csv")
X_test_data = test_data.iloc[:, 6:]  # Adjust column indices as needed

# Align columns with training data (assuming `training_columns` are saved from the training process)
training_columns = joblib.load("./results/training_columns.joblib")
X_test_data = X_test_data.reindex(columns=training_columns, fill_value=0)

# Apply PCA transformation
X_test_data_pca = pca.transform(X_test_data)

# Predict purity for the test data
predicted_purity = best_rf_model.predict(X_test_data_pca)

# Create the submission DataFrame with the required structure
submission = pd.DataFrame({
    "ID": test_data.index + 1,  # Assuming IDs are 1-indexed; adjust if necessary
    "PURITY": predicted_purity
})

# Save the predictions to CSV in the desired format
submission.to_csv("./results/predicted_purity_submission.csv", index=False)
print("Predictions saved to predicted_purity_submission.csv")
