import json
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "GreatLearningSayli/tourism-model"

# 1) Load model from Hugging Face Model Hub
model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.pkl")
model = joblib.load(model_path)

# 2) Load feature columns from Hugging Face Model Hub
cols_path = hf_hub_download(repo_id=REPO_ID, filename="feature_columns.json")
with open(cols_path, "r") as f:
    feature_cols = json.load(f)

def predict():
    # ✅ Use feature_cols (NOT X_train.columns)
    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[0] = 0

    # Example values (update as you want)
    if "Age" in input_data.columns:
        input_data.at[0, "Age"] = 35
    if "CityTier" in input_data.columns:
        input_data.at[0, "CityTier"] = 2
    if "TypeofContact_Self" in input_data.columns:
        input_data.at[0, "TypeofContact_Self"] = 1

    pred = model.predict(input_data)
    return pred

if __name__ == "__main__":
    print("Prediction:", predict())
