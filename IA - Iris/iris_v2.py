import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Importamos el dataset
iris = pd.read_csv("Iris2.csv")


# Eliminamos la primera columna ID
iris = iris.drop("Id", axis=1)
# Visualizamos los primeros 5 datos del dataset
print(iris.head())

# Analizamos los datos que tenemos disponibles
print("Informacion del dataset: ")
print(iris.info())

print("Descripcion del dataset: ")
print(iris.describe())

# Distribucion de las especies
print(iris.groupby("Species").size())

# Generamos los graficos correspondientes al largo y ancho del sepalo
"""
fig = iris[iris.Species == "Iris-setosa"].plot(
    kind="scatter",
    x="SepalLengthCm",
    y="SepalWidthCm",
    color="red",
    label="Setosa",
)
iris[iris.Species == "Iris-versicolor"].plot(
    kind="scatter",
    x="SepalLengthCm",
    y="SepalWidthCm",
    color="green",
    ax=fig,
    label="Versicolor",
)
iris[iris.Species == "Iris-virginica"].plot(
    kind="scatter",
    x="SepalLengthCm",
    y="SepalWidthCm",
    color="blue",
    ax=fig,
    label="Virginica",
)
"""

# Generamos los graficos correspondientes al largo y ancho del petalo
fig = iris[iris.Species == "Iris-setosa"].plot(
    kind="scatter",
    x="PetalLengthCm",
    y="PetalWidthCm",
    color="red",
    label="Setosa",
)
iris[iris.Species == "Iris-versicolor"].plot(
    kind="scatter",
    x="PetalLengthCm",
    y="PetalWidthCm",
    color="green",
    ax=fig,
    label="Versicolor",
)
iris[iris.Species == "Iris-virginica"].plot(
    kind="scatter",
    x="PetalLengthCm",
    y="PetalWidthCm",
    color="blue",
    ax=fig,
    label="Virginica",
)

fig.set_xlabel("Petalo - Longitud")
fig.set_ylabel("Petalo - Ancho")
fig.set_xlabel("Petalo - Longitud vs Ancho")
# fig.set_xlabel("Sepalo - Longitud")
# fig.set_ylabel("Sepalo - Ancho")
# fig.set_xlabel("Sepalo - Longitud vs Ancho")
# plt.show()


# Separo todos los datos con las características y las etiquetas o resultados
X = np.array(iris.drop("Species", axis=1))
y = np.array(iris["Species"])


# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(
    "Son {} datos para entrenamiento y {} datos para prueba".format(
        X_train.shape[0], X_test.shape[0]
    )
)

# Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print("Precisión Regresión Logística: {}".format(algoritmo.score(X_train, y_train)))

# Modelo de Máquinas de Vectores de Soporte
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print(
    "Precisión Máquinas de Vectores de Soporte: {}".format(
        algoritmo.score(X_train, y_train)
    )
)

# Modelo de Vecinos más Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print("Precisión Vecinos más Cercanos: {}".format(algoritmo.score(X_train, y_train)))

# Modelo de Árboles de Decisión Clasificación
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print(
    "Precisión Árboles de Decisión Clasificación: {}".format(
        algoritmo.score(X_train, y_train)
    )
)
