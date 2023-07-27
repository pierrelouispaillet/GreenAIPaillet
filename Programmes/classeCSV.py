import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Nouveaux ensembles de données
dataAcc = np.array([0.53933334, 0.57674998, 0.55785716, 0.56849998, 0.44220001,
       0.638999999, 0.610249996, 0.631428599, 0.631166637, 0.662999988,
       0.667777777, 0.632499993, 0.670428574, 0.672999978, 0.729200006,
       0.703555584, 0.674875021, 0.669142842, 0.710666656, 0.691799998,
       0.664444447, 0.686625004, 0.675857127, 0.719166696, 0.683600008])

duration = np.array([
    68.63483428955078, 61.720112562179565, 59.34459567070007, 
    56.79960227012634, 22.822481155395508, 93.24828052520752, 
    87.48148393630981, 81.13729047775269, 75.78829431533813, 
    37.31761407852173, 119.59509062767029, 110.02155661582947, 
    100.2523455619812, 91.79510378837585, 52.094313859939575, 
    145.63746666908264, 133.0175004005432, 120.76134037971497, 
    109.40738296508789, 67.34171104431152, 173.43949151039124, 
    158.7904932498932, 143.6190323829651, 130.45067286491394, 
    88.06346249580383
])

package_0 = np.array([
    2373398587.0, 2219975701.0, 2126302784.0, 2031111500.0, 
    868413609.0, 3634118566.0, 3465666298.0, 3252875096.0, 
    2992682707.0, 1549339295.0, 4798017929.0, 4368254747.0, 
    3881545911.0, 3508561635.0, 2103368090.0, 5892410452.0, 
    5306359768.0, 4793448056.0, 4327710784.0, 2799240454.0, 
    7392337826.0, 6838410289.0, 6098132277.0, 5577304204.0, 
    4324956030.0
])

dram_0 = np.array([
    50806877.0, 44910712.0, 43203441.0, 42554091.0, 18052627.0, 
    75323110.0, 72514096.0, 67716563.0, 62250024.0, 31164349.0, 
    93173895.0, 85450709.0, 76410999.0, 69393255.0, 41725174.0, 
    112612261.0, 99599842.0, 90086927.0, 81661168.0, 54482893.0, 
    139149973.0, 127724648.0, 115879098.0, 112927689.0, 94506655.0
])

core_0 = np.array([
    2308111461.0, 2161539388.0, 2070154246.0, 1977285832.0, 
    846541473.0, 3544270528.0, 3381010815.0, 3174238578.0, 
    2919307224.0, 1513060519.0, 4682938512.0, 4262658399.0, 
    3785637952.0, 3420911469.0, 2053053582.0, 5752046126.0, 
    5178736987.0, 4677667041.0, 4222933097.0, 2733765402.0, 
    7223387453.0, 6683581523.0, 5958146185.0, 5448703153.0, 
    4234463953.0
])

nvidia_gpu_0 = np.array([
    3898753, 3515878, 3419803, 3276461, 1468615, 6066121, 
    5676403, 5026764, 4898895, 2698239, 10007117, 9252351, 
    8619131, 7533524, 4956723, 15979638, 14441871, 12912401, 
    11205426, 7923737, 21477277, 19344881, 17223435, 15316027, 
    11304277
])




dataAcc = dataAcc.reshape(5, 5)
dataPackage0 = package_0.reshape(5, 5)
dataDram0 = dram_0.reshape(5, 5)
dataCore0 = core_0.reshape(5, 5)
dataNvidiaGpu0 = nvidia_gpu_0.reshape(5, 5)
duration = duration.reshape(5, 5)

dataPackage0E = dataPackage0/duration
dataDram0E = dataDram0/duration
dataCore0E  = dataCore0 /duration
dataNvidiaGpu0E = dataNvidiaGpu0/duration

y_values = [0.2, 0.4, 0.6, 0.8, 1]
x_values = [9, 8, 7, 6,5]

x = np.array(x_values)
y = np.array(y_values)

X, Y = np.meshgrid(x, y)

data_arrays = [dataAcc, dataPackage0, dataDram0]
data_arraysE = [dataAcc, dataPackage0E, dataDram0E]
titles = ['Accuracy', 'Processeur', 'RAM']

data_arrays2 = [duration, dataCore0 , dataNvidiaGpu0]
data_arrays2E = [duration, dataCore0E , dataNvidiaGpu0E]
titles2 = ['Duration', 'Coeur du processeur', 'GPU Nvidia']


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), subplot_kw={'projection': '3d'})


for idx, (ax, data, title) in enumerate(zip(axes[0], data_arrays, titles)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre de classes')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

for idx, (ax, data, title) in enumerate(zip(axes[1], data_arrays2, titles2)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre de classes')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

plt.show()