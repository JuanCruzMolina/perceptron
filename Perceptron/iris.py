from main import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import csv

# Lectura del CSV
caracteristicas = []
etiquetas = []

with open("iris.csv", newline="") as File:
    reader = csv.reader(File)
    data = list(reader)
    caract = list()
    for row in reader:
        print("A")
        etiquetas.append(row[3])


# Perceptron
