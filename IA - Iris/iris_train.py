import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar el archivo csv
data = pd.read_csv("iris2.csv")

# Convertir las etiquetas de especies en valores numéricos
data["Species"] = data["Species"].map(
    {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
)

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop("Species", axis=1).values
y = data["Species"].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear gráfico del conjunto de datos
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Anchura del sépalo (cm)")
plt.title("Conjunto de Datos Iris")
plt.show()

# Solicitar al usuario que ingrese los pesos
pesos = []
for i in range(1, X.shape[1]):
    peso = float(input(f"Ingrese el peso {i}: "))
    pesos.append(peso)
pesos = np.array(pesos)


# Definir la función de activación (función escalón)
def funcion_activacion(x):
    return np.where(x >= 0, 1, 0)


# Realizar predicciones en el conjunto de prueba utilizando los pesos ingresados
y_pred = [funcion_activacion(np.dot(xi, pesos)) for xi in X_test]

# Convertir los valores de y_pred a enteros
y_pred = [int(prediccion) for prediccion in y_pred]

# Mapear los valores numéricos de las predicciones a etiquetas de especies
mapeo_especies = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
especies_predichas = [mapeo_especies[prediccion] for prediccion in y_pred]

# Crear gráfico de entrenamiento
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", label="Verdadero")
plt.scatter(
    X_test[:, 0], X_test[:, 1], c=y_pred, cmap="viridis", marker="x", label="Predicho"
)
plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Anchura del sépalo (cm)")
plt.title("Resultado del Entrenamiento del Perceptrón")
plt.legend()
plt.show()
