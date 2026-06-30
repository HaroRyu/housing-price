from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Chargement du dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Création du modèle
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Entraînement
model.fit(X_train, y_train)

# Évaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"MSE : {mse:.4f}")

# Sauvegarde
joblib.dump(model, "model.joblib")

print("Modèle sauvegardé dans model.joblib")