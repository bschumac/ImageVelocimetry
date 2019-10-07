import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
import progressbar
from functions.TST_fun import *
from scipy.io import loadmat
file = loadmat('/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/Tb_streamwise_Area_EE1_3.mat')

Tb = file['Tb_streamwise_Area_EE3']
#Tb = np.array(Tb)
#Tb = np.rot90(Tb,-1, axes=(1,-1))
#Tb = np.flip(Tb,-1)


Tb_test = Tb[0:3]
print(Tb_test)


#Tb_pertub = create_tst_pertubations_mm(Tb)
#print(np.shape(Tb_pertub))
#Tb_pertub = np.fliplr(Tb_pertub)


writeNetCDF("/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/", "Tb_streamwise_Area_EE3.nc", "Tb", Tb)


#imgplot = plt.imshow(Tb[1])
#plt.show()
