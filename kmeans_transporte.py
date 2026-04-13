import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#DATASET
data = {
    "hora": [7, 8, 12, 18, 9, 14, 20],
    "pasajeros": [120, 200, 80, 250, 150, 100, 220],
    "clima": [0, 1, 0, 1, 2, 0, 1],  # 0=Soleado,1=Lluvia,2=Nublado
    "trafico": [2, 2, 1, 2, 1, 0, 2]  # 0=Bajo,1=Medio,2=Alto
}

df = pd.DataFrame(data)

# Convertir a matriz
X = df.values


# FUNCIONES K-MEANS
# Distancia euclidiana
def distancia(a, b):
    return np.sqrt(np.sum((a - b)**2))
# Número de clusters
k = 3
# Inicializar centroides
np.random.seed(1)
centroides = X[np.random.choice(len(X), k, replace=False)]


#ENTRENAMIENTO
for _ in range(100):
    clusters = [[] for _ in range(k)]

    # Asignar cada punto al cluster más cercano
    for punto in X:
        distancias = [distancia(punto, c) for c in centroides]
        cluster = np.argmin(distancias)
        clusters[cluster].append(punto)

    # Calcular nuevos centroides
    nuevos_centroides = []
    for cluster in clusters:
        nuevos_centroides.append(np.mean(cluster, axis=0))

    nuevos_centroides = np.array(nuevos_centroides)

    # Verificar convergencia
    if np.all(centroides == nuevos_centroides):
        break

    centroides = nuevos_centroides


#RESULTADOS
print("Centroides finales:")
print(centroides)

# Mostrar cluster de cada dato
print("\nAsignación de datos a clusters:")
for i, punto in enumerate(X):
    distancias = [distancia(punto, c) for c in centroides]
    cluster = np.argmin(distancias)
    print(f"Dato {i} -> Cluster {cluster}")


#VISUALIZACIÓN
colores = ["red", "blue", "green"]

plt.figure(figsize=(8,6))

# Graficar clusters
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(
        cluster[:, 0],  # hora
        cluster[:, 1],  # pasajeros
        color=colores[i],
        label=f"Cluster {i}",
        s=100
    )

# Graficar centroides
centroides = np.array(centroides)
plt.scatter(
    centroides[:, 0],
    centroides[:, 1],
    color="black",
    marker="X",
    s=200,
    label="Centroides"
)

# Etiquetas para cada punto
for i, punto in enumerate(X):
    plt.text(punto[0]+0.2, punto[1]+2, f"{i}", fontsize=9)

# Diseño
plt.title("Clustering del Sistema de Transporte", fontsize=14)
plt.xlabel("Hora del día", fontsize=12)
plt.ylabel("Número de pasajeros", fontsize=12)
plt.legend()
plt.grid(True)

plt.show()
