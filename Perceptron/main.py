import numpy as np
import csv
import matplotlib.pyplot as plt

# Paso 1: Leer y preparar los datos
features = []
labels = []

with open('irisFinal.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  #  no usa el encabezado
    for row in reader:
        if not row:
            continue  
        petal_length = float(row[2])
        sepal_width = float(row[1])
        species = row[4]

        feature_vector = [1.0, sepal_width, petal_length]  # sesgo 1.0
        features.append(feature_vector)

        if species == 'setosa':
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
    return 1 if f > 0.6 else 0

# Paso 5: Implementar el algoritmo de entrenamiento del perceptrón
learning_rate = 0.1
epochs = 1000
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
    # Detener si no hay errores
    if total_error == 0:
        break

# Mostrar los pesos resultantes
print("Pesos entrenados:")
print(f"w0 (sesgo): {weights[0]:.2f}")
print(f"w1: {weights[1]:.2f}")
print(f"w2: {weights[2]:.2f}")

#Evaluar los pesos entrenados
def evaluate_weights(weights, features, labels):
    correct_predictions = 0
    for i in range(len(features)):
        input_vector = features[i]
        target = labels[i]
        f = np.dot(weights, input_vector)
        output = activation_function(f)
        if output == target:
            correct_predictions += 1
    accuracy = correct_predictions / len(features)
    return accuracy

# Evaluación de los pesos entrenados
accuracy_trained = evaluate_weights(weights, features, labels)
print(f"\nPrecisión con los pesos entrenados: {accuracy_trained * 100:.2f}%")

# Evaluar pesos arbitrarios
test_weights = [-3.1, 2.56,-1.17]  # Pesos de prueba
accuracy_test = evaluate_weights(test_weights, features, labels)
print(f"\nPrecisión con los pesos de prueba {test_weights}: {accuracy_test * 100:.2f}%")

# Paso 7: Graficar los errores durante el entrenamiento
plt.plot(errors)
plt.xlabel('Épocas')
plt.ylabel('Errores')
plt.title('Errores durante el entrenamiento')
plt.show()
