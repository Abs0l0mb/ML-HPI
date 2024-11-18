import optuna
import pandas as pd
import Utils as Utils
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# Optuna objective function
def objective(trial):
    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    learning_rate = trial.suggest_int("min_samples_split", 0.05, 0.2)

    # Define and train Random Forest model
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

    model.fit(X_train, y_train)

    # Cross-validate and return custom purity score
    cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring=custom_scorer).mean()
    return cv_score

data = pd.read_csv('../data/substances.csv')
target_column = 'substance'
X = data.drop(columns=[target_column])
y = data[target_column]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

custom_scorer = make_scorer(accuracy_score, greater_is_better=True)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Parameters:", best_params)

best_xgb_model = XGBClassifier(n_estimators=best_params["n_estimators"], learning_rate=best_params["learning_rate"], max_depth=best_params["max_depth"], random_state=42)
best_xgb_model.fit(X_train_scaled, y_train)

y_pred = best_xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final model accuracy: {accuracy:.4f}")
