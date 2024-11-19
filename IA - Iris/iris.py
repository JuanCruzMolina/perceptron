import matplotlib.pyplot as plt
import csv
import numpy as np

with open("iris.csv", newline="") as File:
    reader = csv.reader(File)
    data = list(reader)
    print(data)
    caracteristicas = []
    etiquetas = []
    for row in reader:
        caracteristicas.append([float(row[0]), float(row[1])])
        etiquetas.append(row[3])
        print(row)


# Definimos un diccionario con los colores correspondientes
colores_iris = {
    "Iris-setosa": "red",
    "Iris-versicolor": "green",
    "Iris-virginica": "blue",
}


# Graficamos los datos con colores correspondientes al tipo de iris
for i in range(len(caracteristicas)):
    plt.scatter(
        caracteristicas[i][0],
        caracteristicas[i][1],
        color=colores_iris[etiquetas[i]],
    )


# Grafico
#    plt.legend(colores_iris.keys())
#    plt.xlabel("Longitud del sépalo (cm)")
#    plt.ylabel("Anchura del sépalo (cm)")
#    plt.title("Distribución de tipos de iris con perceptrón")
#    plt.show()
