import joblib
import os

def predict_price(sample_features):
    model = joblib.load("models/car_price_model.pkl")
    prediction = model.predict([sample_features])[0]

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_prediction.txt", "w") as f:
        f.write(f"Input Features: {sample_features}\n")
        f.write(f"Predicted Car Price: {prediction}\n")

    return prediction
