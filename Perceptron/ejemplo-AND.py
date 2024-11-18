from main import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# La variable ejemplo contiene la tabla del AND
ejemplos = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
perceptron_and = Perceptron(3)


grad_pesos = [perceptron_and.pesos]

for epoca in range(0, 100):
    for i in range(0, 4):
        perceptron_and.propagacion(ejemplos[i, 0:3])
        perceptron_and.actualizacion_coef(0.5, ejemplos[i, 3])
        grad_pesos = np.concatenate((grad_pesos, [perceptron_and.pesos]), axis=0)

plt.plot(grad_pesos[:, 0], "k")
plt.plot(grad_pesos[:, 1], "r")
plt.plot(grad_pesos[:, 2], "b")

plt.show()
