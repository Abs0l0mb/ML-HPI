import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
import RF.Utils as Utils

# Optuna objective function
def objective(trial):
    # Suggest values for hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 400)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

    # Define and train Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Cross-validate and return custom purity score
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer).mean()
    return cv_score

data = Utils.pre_process_data("../data/train.csv", False)
X = data.drop(columns=['PURITY'])
y = data['PURITY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_scorer = make_scorer(Utils.purity_score, greater_is_better=True)

'''
# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)  # Example number; adjust based on prior tuning. TWEAK 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
'''

study = optuna.create_study(direction="maximize")
study.optimize(Utils.objective, n_trials=30)

best_params = study.best_params
print("Best Parameters:", best_params)

best_rf_model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42
)
best_rf_model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = best_rf_model.predict(X_test)
final_score = Utils.purity_score(y_test, y_pred)
print(f"Final Optimized Random Forest Custom Purity Score (within ±5% range): {final_score:.4f}")
