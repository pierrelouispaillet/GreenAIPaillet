import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vos données
data = np.array([
    [14719.72538, 10951.68322, 7808.65548, 4008.92116, 2416.72592],
    [14079.73732, 10475.52308, 7469.14872, 3834.62024, 2311.65088],
    [16319.69553, 12142.08357, 8657.42238, 4444.67346, 2679.41352],
    [16959.68359, 12618.24371, 8996.92914, 4618.97438, 2784.48856],
    [17279.67762, 12856.32378, 9166.68252, 4706.12484, 2837.02608],
    [16959.68359, 12618.24371, 8996.92914, 4618.97438, 2784.48856],
    [16639.68956, 12380.16364, 8827.17576, 4531.82392, 2731.95104],
    [15679.70747, 11665.92343, 8317.91562, 4270.37254, 2574.33848],
    [14719.72538, 10951.68322, 7808.65548, 4008.92116, 2416.72592],
    [14399.73135, 10713.60315, 7638.9021, 3921.7707, 2364.1884],
    [13759.74329, 10237.44301, 7299.39534, 3747.46978, 2259.11336],
    [13119.75523, 9761.28287, 6959.88858, 3573.16886, 2154.03832],
    [12799.7612, 9523.2028, 6790.1352, 3486.0184, 2101.5008],
    [12159.77314, 9047.04266, 6450.62844, 3311.71748, 1996.42576],
    [12799.7612, 9523.2028, 6790.1352, 3486.0184, 2101.5008],
    [12799.7612, 9523.2028, 6790.1352, 3486.0184, 2101.5008],
    [13439.74926, 9999.36294, 7129.64196, 3660.31932, 2206.57584],
    [14079.73732, 10475.52308, 7469.14872, 3834.62024, 2311.65088],
    [13759.74329, 10237.44301, 7299.39534, 3747.46978, 2259.11336],
    [14079.73732, 10475.52308, 7469.14872, 3834.62024, 2311.65088],
    [15999.7015, 11904.0035, 8487.669, 4357.523, 2626.876],
    [16319.69553, 12142.08357, 8657.42238, 4444.67346, 2679.41352],
    [15999.7015, 11904.0035, 8487.669, 4357.523, 2626.876],
    [16639.68956, 12380.16364, 8827.17576, 4531.82392, 2731.95104]
])



# Création de la grille
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
x, y = np.meshgrid(x, y)

# Création de la figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dessin du meshgrid 3D
ax.plot_surface(x, y, data, cmap='viridis')

# Réglage des graduations de l'axe X
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels([ 1, 0.8, 0.6, 0.4, 0.2])


# Réglage des graduations de l'axe Y
ax.set_yticks(np.arange(0, 25, 3))
ax.set_yticklabels(["{}h".format(i) for i in range(0, 25, 3)])

# Nommer les axes
ax.set_xlabel('Fraction de la base de données')
ax.set_ylabel('Heure de la journée')
ax.set_zlabel('CO2')

# Affichage du graphique
plt.show()