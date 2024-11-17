from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import Utils as Utils

data = Utils.pre_process_data("../data/train.csv", False, True)

X = data.drop(columns=['PURITY'])
y = data['PURITY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

best_rf_model = RandomForestRegressor(
    n_estimators=394,
    max_depth=25,
    min_samples_split=2,
    random_state=42
)
best_rf_model.fit(X_train, y_train)

y_pred = best_rf_model.predict(X_test)

#joblib.dump(best_rf_model, "./random_forest_model.joblib")
final_score = Utils.purity_score(y_test, y_pred)
print(f"Final Optimized Random Forest Custom Purity Score (within ±5% range): {final_score:.4f}")
