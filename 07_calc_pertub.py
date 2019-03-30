import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
import progressbar
from TST_fun import *
from netCDF4 import Dataset
file = h5py.File('/home/benjamin/Met_ParametersTST/data/Tb_1Hz.mat','r')

Tb = file.get('Tb')
Tb = np.array(Tb)
Tb = np.rot90(Tb,-1, axes=(1,-1))
Tb = np.flip(Tb,-1)


# create_tst_pertubations has moving_mean_size setting as well standart is 
Tb_pertub = create_tst_pertubations_mm(Tb, moving_mean_size = 60)
#print(np.shape(Tb_pertub))
Tb_pertub = np.fliplr(Tb_pertub)


writeNetCDF("/home/benjamin/Met_ParametersTST/data", "Tb_pertub_py.nc", "Tb_pertub", Tb_pertub)


imgplot = plt.imshow(Tb_pertub[1])
plt.show()
