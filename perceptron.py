import pandas as pd
import numpy as np

# Cargar el conjunto de datos
data = pd.read_csv('iris.csv')

# Supongamos que la columna 'Clase' tiene las etiquetas (0 y 1)
# Asegúrate de ajustar esto según tu archivo CSV

X = data[['PetalLengthCM', 'PetalWidthCM']].values
y = data['Clase'].values

# Agregar el término de bias
X = np.insert(X, 0, 1, axis=1)

# Inicializar pesos
w = np.zeros(X.shape[1])

# Parámetros de entrenamiento
eta = 0.01
max_iter = 1000
threshold = 0.6

for epoch in range(max_iter):
    error_count = 0
    for xi, target in zip(X, y):
        f = np.dot(w, xi)
        output = 1 if f > threshold else 0
        error = target - output
        if error != 0:
            w += eta * error * xi
            error_count += 1
    if error_count == 0:
        print(f'Convergió en {epoch} épocas')
        break

print(f'Pesos finales: w0={w[0]}, w1={w[1]}, w2={w[2]}')
