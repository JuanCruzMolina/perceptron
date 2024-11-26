import numpy as np
import csv
import matplotlib.pyplot as plt

# Paso 1: Leer y preparar los datos
features = []
labels = []

with open('iris.csv', 'r') as file:
    reader = csv.reader(file)
    # next(reader)  # Descomenta esta línea si el CSV tiene un encabezado
    for row in reader:
        if not row:
            continue  # Saltar líneas vacías
        petal_length = float(row[2])
        petal_width = float(row[3])
        species = row[4]

        feature_vector = [1.0, petal_length, petal_width]
        features.append(feature_vector)

        if species == 'Iris-setosa':
            labels.append(1)
        else:
            labels.append(0)

features = np.array(features)
labels = np.array(labels)

# Paso 2: Barajar los datos
dataset = list(zip(features, labels))
np.random.shuffle(dataset)
features, labels = zip(*dataset)
features = np.array(features)
labels = np.array(labels)

# Paso 3: Inicializar los pesos del perceptrón aleatoriamente
np.random.seed(42)  # Fijar la semilla para reproducibilidad
weights = np.random.uniform(-0.5, 0.5, 3)

# Paso 4: Definir la función de activación
def activation_function(f):
    if f > 0.6:
        return 1
    else:
        return 0

# Paso 5: Implementar el algoritmo de entrenamiento del perceptrón
learning_rate = 0.1
epochs = 50
errors = []

for epoch in range(epochs):
    total_error = 0
    for i in range(len(features)):
        input_vector = features[i]
        target = labels[i]
        f = np.dot(weights, input_vector)
        output = activation_function(f)
        error = target - output
        total_error += abs(error)
        weights += learning_rate * error * input_vector
    errors.append(total_error)
    # Opcional: Detener el entrenamiento si no hay errores
    # if total_error == 0:
    #     break

# Paso 6: Mostrar los pesos resultantes
print("Pesos entrenados:")
print(f"w0 (sesgo): {weights[0]:.2f}")
print(f"w1: {weights[1]:.2f}")
print(f"w2: {weights[2]:.2f}")

# Paso 7: Mostrar el número de épocas ejecutadas
print(f"\nNúmero de épocas ejecutadas: {epoch + 1}")

# Paso 8: Graficar los errores durante el entrenamiento
# plt.plot(errors)
# plt.xlabel('Épocas')
# plt.ylabel('Errores')
# plt.title('Errores durante el entrenamiento')
# plt.show()

# Paso 9: Opciones dadas
print("\nOpciones dadas:")
print("a. w0 = 1.2, w1 = -0.15, w2 = -0.47")
print("b. w0 = 9.5, w1 = -1.21, w2 = -2.48")
print("c. w0 = 6.2, w1 = -1.54, w2 = -2.18")
print("d. w0 = 12, w1 = -1.5, w2 = -4.7")
