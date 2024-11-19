import csv
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron


datos = []
etiquetas = []
etiquetas_mapeo = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 1}

with open("Iris2.csv") as file:
    archivo = csv.reader(file)
    next(archivo)
    for fila in archivo:
        if fila:
            datos.append([float(x) for x in fila[:-1]])
            etiquetas.append(etiquetas_mapeo[fila[-1]])

pr = Perceptron(3)
pesos = []
errores = []

for i in range(100):
    for planta in datos:
        if planta:
            salida = planta[-1]  # salida esperada
            entrada = [1] + planta[0:-1]  # se le agrega una entrada que es 1
            pesos.append(pr._w)
            error = pr.train(entrada, salida)
            errores.append(error)

x1 = float(input("Ingrese largo del sepalo (cm): "))
x2 = float(input("Ingrese ancho del sepalo (cm): "))
x3 = float(input("Ingrese largo del petalo (cm): "))
x4 = float(input("Ingrese ancho del petalo (cm): "))

X_test = [
    [x1, x2],
    [x3, x4],
]
y_pred = [pr.predict([1] + x) for x in X_test]

tipos = list(etiquetas_mapeo.keys())

print(f"Valor de entrada: {X_test},\nValor de salida: {y_pred}")
print(tipos[1]) if y_pred[1] == 0 else print(tipos[0])


# fase de graficación
plt.scatter(datos[0], datos[1])
plt.scatter(X_test[0], X_test[1], c=y_pred, marker="o")
plt.xlabel("Alto de sépalo")
plt.ylabel("Ancho de sépalo")
plt.title("Clasificación de Iris")
plt.show()
