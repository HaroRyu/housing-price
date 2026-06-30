from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Création de l'application FastAPI
app = FastAPI(title="Housing Price Prediction API")

# Chargement du modèle
model = joblib.load("model.joblib")


# Schéma d'entrée
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


# Endpoint de test
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint de prédiction
@app.post("/predict")
def predict(data: HousingInput):
    input_data = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    prediction = model.predict(input_data)

    return {
        "predicted_house_value": round(float(prediction[0]), 2)
    }