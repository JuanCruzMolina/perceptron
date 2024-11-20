import numpy as np
import csv

class Perceptron:
    def __init__(self, n):
        self.pesos = np.zeros(n)
        self.n = n

    def propagacion(self, entradas):
        suma = np.dot(self.pesos, entradas)
        self.salida = 1 if suma > 9.5 else 0
        self.entradas = entradas

    def actualizacion_coef(self, alfa, salidad):
        error = salidad - self.salida
        for i in range(self.n):
            self.pesos[i] += alfa * error * self.entradas[i]

# Carga de datos
caracteristicas = []
etiquetas = []

with open("iris.csv", newline="") as File:
    reader = csv.reader(File)
    for row in reader:
        # Asumiendo que las columnas están en el orden correcto
        petal_length = float(row[2])
        petal_width = float(row[3])
        clase = row[4]

        # Convertir etiquetas a valores numéricos
        if clase == 'Iris-setosa':
            etiqueta = 0
        else:
            etiqueta = 1  # Puedes ajustar esto según tu necesidad

        # Añadir el sesgo (x = 1)
        caracteristicas.append([1, petal_length, petal_width])
        etiquetas.append(etiqueta)

# Convertir a arrays de NumPy
caracteristicas = np.array(caracteristicas)
etiquetas = np.array(etiquetas)

# Entrenamiento del perceptrón
perceptron = Perceptron(3)
alfa = 0.1  # Tasa de aprendizaje

for epoch in range(100):  # Número de iteraciones
    for entradas, salida_real in zip(caracteristicas, etiquetas):
        perceptron.propagacion(entradas)
        perceptron.actualizacion_coef(alfa, salida_real)

print("Pesos entrenados:", perceptron.pesos)
