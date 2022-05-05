import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time

script_name = path.basename(__file__).split('.')[0]
output_folder = path.join(get_root_path(), 'output', script_name, get_time())
makedirs(output_folder)
fontsize_1 = 18
fontsize_2 = 14

linewidth = 2
n_exp = [5, 10, 20,  50,100,  480]
d2d = [0.602, 0.686, 0.715,  0.837, 0.877, 0.936]
d2d_2 = [0.725, 0.759, 0.823,  0.890,0.919, 0.967]
deep = [0.884, 0.893, 0.911,  0.936, 0.942, 0.967]

x = np.arange(len(n_exp))
y_grid = np.linspace(0.5, 1, 11)

plt.title('PR-AUC VS Number of Experiments', fontsize=fontsize_1)
plt.plot(n_exp, deep, marker='x', linewidth=linewidth, label='DeepProp')
plt.plot(n_exp, d2d, marker='x' ,linewidth=linewidth, label='D2D')
plt.plot(n_exp, d2d_2, marker='x', linewidth=linewidth, label='D2D Deconstructed')

plt.xscale('log')
plt.xticks(n_exp, n_exp)
plt.yticks(y_grid)
plt.grid()
plt.legend(fontsize=fontsize_2)
plt.xlabel('#Experiments', fontsize=fontsize_2)
plt.ylabel('PR-AUC', fontsize=fontsize_2)
plt.savefig(path.join(output_folder,'fig'))