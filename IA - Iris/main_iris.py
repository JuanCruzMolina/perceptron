import csv, matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np
from perceptron import Perceptron


data = []
tipos = []
eje_x = []
eje_y = []
eje_x2 = []
eje_y2 = []
punto = []
punto2 = []


# Set de colores para los tipos
color_iris = {
    "Iris-setosa": "red",
    "Iris-versicolor": "green",
    "Iris-virginica": "blue",
}

# Apertura del csv y cargamos los datos en una lista.
with open("iris.csv", newline="") as File:
    reader = csv.reader(File)
    for row in reader:
        if row:
            data.append([float(x) for x in row[:-1]])
            tipos.append(color_iris[row[-1]])


# Creamos los conjuntos
for i in range(0, len(data)):
    print(data[i])
    eje_x.append(float(data[i][0]))
    eje_y.append(float(data[i][1]))
    eje_x2.append(float(data[i][2]))
    eje_y2.append(float(data[i][3]))
    punto.append([eje_x[i], eje_y[i]])
    punto2.append([eje_x2[i], eje_y2[i]])

tipos = np.array(tipos)

# Entrenamiento


# Graficamos los puntos del csv.
leyendas = [
    mpatches.Patch(color=color, label=especies)
    for especies, color in color_iris.items()
]

# Grafico 1
fig, ax = plt.subplots()
ax.legend(handles=leyendas)
ax.scatter(eje_x[:], eje_y[:], c=tipos)
ax.set_xlabel("Longitud del Sépalo")
ax.set_ylabel("Ancho del Sépalo")
ax.plot()
ax.set_title("Iris")

# Grafico 2
fig, ay = plt.subplots()
ay.legend(handles=leyendas)
ay.scatter(eje_x2[:], eje_y2[:], c=tipos)
ay.set_xlabel("Longitud del Pétalo")
ay.set_ylabel("Ancho del Pétalo")
ay.plot()
ay.set_title("Iris")

plt.show()
