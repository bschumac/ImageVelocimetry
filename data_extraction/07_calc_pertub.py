
import matplotlib.pyplot as plt
from functions.TST_fun import readnetcdftoarr, create_tst_pertubations_mm, writeNetCDF, create_tst_mean


datapath= "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"

Tb = readnetcdftoarr(datapath+"Tb_stab_1Hz.nc")


Tb = create_tst_mean(Tb, moving_mean_size = 4)

# create_tst_pertubations has moving_mean_size setting as well standart is 
Tb_pertub = create_tst_pertubations_mm(Tb, moving_mean_size = 60)
print(np.shape(Tb_pertub))
#Tb_pertub = np.fliplr(Tb_pertub)



writeNetCDF(datapath, "Tb_1Hz_pertub_60s_py.nc", "Tb_pertub", Tb_pertub)


imgplot = plt.imshow(Tb_pertub[1])
plt.show()
