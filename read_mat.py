import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
import progressbar
from TST_fun import *
file = h5py.File('/home/benjamin/Met_ParametersTST/T0/data/Tb_1Hz.mat','r')

Tb = file.get('Tb')
#Tb = np.array(Tb)
#Tb = np.rot90(Tb,-1, axes=(1,-1))
#Tb = np.flip(Tb,-1)


Tb_test = Tb[0:3]
print(Tb_test)


#Tb_pertub = create_tst_pertubations_mm(Tb)
#print(np.shape(Tb_pertub))
#Tb_pertub = np.fliplr(Tb_pertub)


writeNetCDF("/home/benjamin/Met_ParametersTST/T0/data/", "Tb_1Hz_test.nc", "Tb_test", Tb_test)


#imgplot = plt.imshow(Tb[1])
#plt.show()
