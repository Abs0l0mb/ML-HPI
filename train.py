import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

# Load the training data
data = pd.read_csv("./data/train.csv")
X = data.iloc[:, 6:]  # Spectral data columns
y = data['PURITY']    # Target variable (purity)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA with a fixed number of components (e.g., 20)
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the Random Forest model with the optimal hyperparameters
best_rf_model = RandomForestRegressor(
    n_estimators=332,
    max_depth=22,
    min_samples_split=2,
    random_state=42
)
best_rf_model.fit(X_train_pca, y_train)

# Save the trained PCA and model for future use
joblib.dump(pca, "./results/pca_model.joblib")
joblib.dump(best_rf_model, "./results/random_forest_model.joblib")
joblib.dump(X_train.columns, "./results/training_columns.joblib")
