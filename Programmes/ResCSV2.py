import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Nouveaux ensembles de données
dataAcc = np.array([0.5586249828338623, 0.7383750081062317, 0.799875020980835, 0.8641250133514404, 0.7148125171661377, 0.7287499904632568, 0.804937481880188, 0.8334375023841858, 0.7303749918937683, 0.7836666703224182, 0.7985833287239075, 0.8964583277702332, 0.7293750047683716, 0.7985312342643738, 0.797531247138977, 0.8847187757492065, 0.7206000089645386, 0.7311750054359436, 0.8866000175476074, 0.8828999996185303])

dataPackage0 = np.array([1230127771, 1474442454, 1857048407, 2352088583, 1916655860, 2674989368, 3462880234, 4292082762, 2880481457, 4131135664, 5085315473, 6181815290, 3746054131, 5407746044, 7619098903, 7994892191, 4168396983, 6097743362, 7943974109, 9900700484])

dataDram0 = np.array([33501074, 37042385, 44688301, 56997169, 44604317, 61410365, 79485209, 99523182, 69319647, 97159725, 119074585, 142526857, 89093583, 126244672, 173671248, 179772244, 97078915, 139414010, 181219324, 224463719])

dataCore0 = np.array([1189581549, 1431052183, 1802878929, 2285536141, 1861273067, 2598760986, 3364117181, 4170888492, 2799797032, 4017623015, 4940801622, 6005526496, 3642665662, 5260335105, 7424372168, 7766155839, 4046399127, 5921591774, 7714271328, 9616689676])

dataNvidiaGpu0 = np.array([2874410, 3259677, 4363717, 5416153, 4422970, 6102224, 7981921, 9794379, 6449327, 8792597, 11359477, 13911216, 7880077, 11302815, 14664232, 17407965, 9113583, 13277137, 17266528, 21428906])

duration = np.array([39.5441524982452,45.295131444931,56.7420718669891,69.1822185516357,57.9580805301666,79.8963119983673,103.367507457733,126.493623018265,83.3799107074738,116.570511817932,150.553865194321,184.387726783752,106.145223855972,152.175242185593,199.994711875916,241.057812690735,128.383146762848,185.487561702728,242.016945123672,299.333375930786])




dataAcc = dataAcc.reshape(5, 4)
dataPackage0 = dataPackage0.reshape(5, 4)
dataDram0 = dataDram0.reshape(5, 4)
dataCore0 = dataCore0.reshape(5, 4)
dataNvidiaGpu0 = dataNvidiaGpu0.reshape(5, 4)
duration = duration.reshape(5, 4)

dataPackage0E = dataPackage0/duration
dataDram0E = dataDram0/duration
dataCore0E  = dataCore0 /duration
dataNvidiaGpu0E = dataNvidiaGpu0/duration

y_values = [0.2, 0.4, 0.6, 0.8, 1]
x_values = [10, 15, 20, 25]

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
    ax.set_xlabel('Nombre d epochs')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

for idx, (ax, data, title) in enumerate(zip(axes[1], data_arrays2, titles2)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre d epochs')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

plt.show()