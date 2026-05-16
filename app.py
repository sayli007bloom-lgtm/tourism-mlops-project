import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ✅ Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="GreatLearningSayli/tourism-model",
    filename="best_model.pkl"
)
model = joblib.load(model_path)

# Load training columns
X_train = pd.read_csv("tourism_project/data/X_train.csv")

def predict():
    input_data = pd.DataFrame(columns=X_train.columns)
    input_data.loc[0] = 0

    input_data.at[0, 'Age'] = 35
    input_data.at[0, 'CityTier'] = 2

    if 'TypeofContact_Self' in input_data.columns:
        input_data.at[0, 'TypeofContact_Self'] = 1

    prediction = model.predict(input_data)
    return prediction

print("Prediction:", predict())