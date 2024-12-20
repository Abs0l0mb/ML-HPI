import pandas as pd
import joblib
import Utils as Utils

best_rf_model = joblib.load("./random_forest_model.joblib")

test_data = pd.read_csv("../data/test.csv")
X_test_data = Utils.pre_process_data("../data/test.csv", True, False)
print(X_test_data)

predicted_purity = best_rf_model.predict(X_test_data)

submission = pd.DataFrame({
    "ID": test_data.index + 1, 
    "PURITY": predicted_purity
})

submission.to_csv("../results/predicted_purity_submission.csv", index=False)
print("Predictions saved to predicted_purity_submission.csv")
