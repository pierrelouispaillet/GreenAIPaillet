import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Nouveaux ensembles de données
dataAcc = np.array([0.97772276401519783, 0.9826732873916626, 0.96410888433456425, 0.9752475023269653, 0.9851484894752502, 0.9777, 0.9778, 0.9786790, 0.987689, 0.978364,0.97772276401519783, 0.9826732873916626, 0.96410888433456425, 0.9752475023269653, 0.9851484894752502, 0.98767, 0.9777, 0.9778, 0.9786790, 0.987689, 0.978364, 0.97772276401519783, 0.9826732873916626, 0.96410888433456425, 0.9752475023269653])

timestamp = np.array([1685525788.0697148, 1685525831.9135482, 1685525877.980751, 1685525930.4272766, 1685525988.9149406, 1685526058.5135574, 1685526098.7885785, 1685526144.8310232, 1685526197.0681875, 1685526255.128802, 1685526324.0570421, 1685526364.5927746, 1685526410.610263, 1685526463.1098406, 1685526521.4827418, 1685526591.280038, 1685526631.2277756, 1685526677.489994, 1685526729.5658376, 1685526787.9806561, 1685526857.2499769, 1685526897.356788, 1685526945.7324743, 1685526998.9519324, 1685527056.9871955])

duration = np.array([43.84300446510315, 46.0664918422699, 52.44561958312988, 58.48659801483154, 69.59768891334534, 40.2741322517395, 46.041494369506836, 52.236164808273315, 58.05970740318298, 68.92706990242004, 40.53450036048889, 46.016496419906616, 52.498456954956055, 58.37174987792969, 69.79595518112183, 39.94661521911621, 46.26115322113037, 52.07311177253723, 58.41399121284485, 69.26827669143677, 40.105650424957275, 48.374536991119385, 53.218217611312866, 58.034114837646484, 69.50230693817139])

package_0 = np.array([1665195322.0, 1768523732.0, 1962626140.0, 2145783203.0, 2448675433.0, 1670925532.0, 1892355507.0, 2050720641.0, 2121157351.0, 2436139027.0, 1683460229.0, 1890455366.0, 2033436201.0, 2175148362.0, 2457598994.0, 1683791344.0, 1887326468.0, 2032508957.0, 2179868748.0, 2456727781.0, 1714217596.0, 2325012114.0, 2104604415.0, 2167551336.0, 2469421657.0])

dram_0 = np.array([60371000.0, 61752894.0, 66883557.0, 72462034.0, 80449073.0, 57336340.0, 62146996.0, 67095837.0, 71737976.0, 80164406.0, 57811314.0, 62273217.0, 67689524.0, 72630002.0, 82039890.0, 57027076.0, 62780234.0, 67008068.0, 72621030.0, 80854590.0, 57902867.0, 71584107.0, 69250738.0, 71922850.0, 81187048.0])

core_0 = np.array([1615898817.0, 1718205196.0, 1906205448.0, 2083473731.0, 2375245689.0, 1626439196.0, 1842056075.0, 1994455588.0, 2059242531.0, 2363248100.0, 1638693620.0, 1840010548.0, 1976923283.0, 2112752953.0, 2383809449.0, 1639406143.0, 1836739377.0, 1976366706.0, 2117197278.0, 2383499818.0, 1669369140.0, 2270339769.0, 2047081246.0, 2105530987.0, 2396006439.0])

nvidia_gpu_0 = np.array([4925072, 5825395, 6907811, 7901925, 9952569, 5403826, 6187901, 7047909, 7816006, 9873623, 5446073, 6129353, 7062120, 7922954, 9866373, 5450823, 6222664, 7111083, 7901164, 9798484, 5461699, 6385536, 7143191, 7914866, 9841710])






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
x_values = [140, 110, 80, 50,20]

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
    ax.set_xlabel('Nombre de couches vérouillées')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

for idx, (ax, data, title) in enumerate(zip(axes[1], data_arrays2, titles2)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre de couches vérouillées')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

plt.show()