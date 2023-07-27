import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataAcc = np.array([0.2800999879837036, 0.38530001044273376, 0.5489000082015991, 0.4512999951839447, 0.34619998931884766, 0.36640000343322754, 0.5756999850273132, 0.633899986743927, 0.3871000111103058, 0.5016000270843506, 0.6345000267028809, 0.5968000292778015, 0.44929999113082886, 0.5819000005722046, 0.5539000034332275, 0.6891000270843506, 0.4830000102519989, 0.6801999807357788, 0.6563000082969666, 0.5893999934196472])

# New values for other datasets
new_data = """
1689059542.795455;AlexNet;32.161073446273804;1035497471.0;27027213.0;1003100886.0;1785100
1689059574.9573574;AlexNet;47.98742175102234;1743273065.0;39404929.0;1697298084.0;2667318
1689059622.9455166;AlexNet;69.06946349143982;2614929280.0;56536477.0;2548638861.0;3948480
1689059692.0184476;AlexNet;90.81942105293274;3505074706.0;74447991.0;3417722574.0;5253752
1689059782.8386734;AlexNet;37.40242385864258;1481196098.0;31862101.0;1444925745.0;2462407
1689059820.2418475;AlexNet;67.831547498703;2683187772.0;54466108.0;2617980786.0;4324449
1689059888.0741363;AlexNet;97.65968704223633;3788314460.0;73887445.0;3695075929.0;6122385
1689059985.7346272;AlexNet;131.0570456981659;5358723647.0;110789573.0;5231444066.0;8715046
1689060116.7924228;AlexNet;47.74318766593933;1962035993.0;41912612.0;1915457377.0;4371323
1689060164.5363653;AlexNet;88.91766881942749;3726439984.0;78065901.0;3639677389.0;8567782
1689060253.4571157;AlexNet;130.7572727203369;5528338996.0;115121104.0;5400461760.0;12860429
1689060384.215813;AlexNet;173.11681532859802;7309951379.0;151953896.0;7140834624.0;16975338
1689060557.3333848;AlexNet;57.903000593185425;2477724492.0;51585866.0;2420823454.0;6211521
1689060615.237194;AlexNet;109.90817666053772;4751454267.0;97982660.0;4643290359.0;12099561
1689060725.1476188;AlexNet;159.89832830429077;6721669438.0;133229334.0;6565904883.0;17546794
1689060885.04671;AlexNet;214.31981086730957;9320649845.0;191483397.0;9109691401.0;23808007
1689061099.3692012;AlexNet;68.18834829330444;2969071074.0;60484647.0;2901807879.0;7922901
1689061167.558328;AlexNet;126.67356371879578;5298172899.0;98652396.0;5175201473.0;14914858
1689061294.2326274;AlexNet;193.0527355670929;8541535995.0;170436209.0;8351087630.0;22952160
1689061487.2861035;AlexNet;257.45375990867615;11588960871.0;238384156.0;11333091959.0;31999403
"""

# Parse new data
new_data = new_data.strip().split("\n")
new_data = [line.split(";")[2:] for line in new_data]
new_data = np.array(new_data, dtype=float)

# Split new data into separate arrays
duration, dataPackage0, dataDram0, dataCore0, dataNvidiaGpu0 = np.hsplit(new_data, 5)

# Reshape data arrays
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


for idx, (ax, data, title) in enumerate(zip(axes[0], data_arraysE, titles)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre d epochs')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

for idx, (ax, data, title) in enumerate(zip(axes[1], data_arrays2E, titles2)):
    ax.plot_wireframe(X, Y, data)
    ax.set_xlabel('Nombre d epochs')
    ax.set_ylabel('Pourcentage de la base de donnees')
    ax.set_zlabel('Z')
    ax.set_title(title)

plt.show()