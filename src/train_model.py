from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/car_price_model.pkl")

    return model
