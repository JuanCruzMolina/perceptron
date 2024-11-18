import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n):
        self.pesos = np.random.randn(n)
        self.n = n

    def propagacion(self, entradas):
        self.salida = 1 * (self.pesos.dot(entradas) > 0)
        self.entradas = entradas

    def actualizacion_coef(self, alfa, salidad):
        for i in range(0, self.n):
            self.pesos[i] = (
                self.pesos[i] + alfa * (salidad - self.salida) * self.entradas[i]
            )


Perceptron_tres_entradas = Perceptron(3)

print(Perceptron_tres_entradas.pesos)

Perceptron_tres_entradas.propagacion([1, 0, 1])
print(Perceptron_tres_entradas.salida)

Perceptron_tres_entradas.actualizacion_coef(0.5, 1)
print(Perceptron_tres_entradas.pesos)
