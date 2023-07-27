import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Vos nouvelles données
data = np.array([
    [3365431.419, 5151402.972, 7031579.854, 8970313.123, 11159695.65],
    [3110127.542, 4760614.1, 6498159.508, 8289819.175, 10313113.68],
    [3075950.317, 4708299.659, 6426751.162, 8198722.261, 10199782.76],
    [2805608.461, 4294492.434, 5861911.143, 7478145.671, 9303335.189],
    [2792621.115, 4274612.946, 5834775.971, 7443528.844, 9260269.44],
    [2815178.084, 4309140.477, 5881905.48, 7503652.807, 9335067.847],
    [3345266.855, 5120537.452, 6989448.93, 8916565.943, 11092830.41],
    [4008646.807, 6135960.745, 8375484.93, 10684757.04, 13292583.56],
    [4101267.089, 6277732.879, 8569001.549, 10931629.68, 13599710.35],
    [3663115.055, 5607061.75, 7653546.55, 9763767.244, 12146807.96],
    [3327836.47, 5093857.087, 6953030.673, 8870106.517, 11035031.64],
    [3151140.213, 4823391.429, 6583849.523, 8399135.472, 10449110.79],
    [3257431.385, 4986089.339, 6805929.48, 8682446.874, 10801569.95],
    [3037330.052, 4649184.341, 6346059.73, 8095782.748, 10071718.82],
    [2689064.121, 4116100.191, 5618408.682, 7167505.194, 8916876.753],
    [2480583.044, 3796982.103, 5182817.77, 6611814.019, 8225558.144],
    [2472038.738, 3783903.493, 5164965.684, 6589039.79, 8197225.414],
    [3032203.468, 4641337.175, 6335348.478, 8082118.211, 10054719.19],
    [3348001.033, 5124722.607, 6995161.598, 8923853.696, 11101896.88],
    [3977887.304, 6088877.748, 8311217.419, 10602769.82, 13190585.73],
    [4031545.548, 6171011.42, 8423328.522, 10745791.98, 13368515.27],
    [4203798.766, 6434676.201, 8783226.588, 11204920.42, 13939703.11],
    [4230798.774, 6476004.609, 8839639.181, 11276886.99, 14029234.54],
    [3903039.18, 5974309.123, 8154833.141, 10403267.58, 12942391.02]
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
ax.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1])

# Réglage des graduations de l'axe Y
ax.set_yticks(np.arange(0, 25, 3))
ax.set_yticklabels(["{}h".format(i) for i in range(0, 25, 3)])

# Nommer les axes
ax.set_xlabel('Fraction de la base de données')
ax.set_ylabel('Heure de la journée')
ax.set_zlabel('Prix')

# Affichage du graphique
plt.show()