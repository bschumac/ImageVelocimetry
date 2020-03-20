
import matplotlib.pyplot as plt
from functions.TST_fun import readnetcdftoarr, create_tst_pertubations_mm, writeNetCDF, create_tst_mean
datapath = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/"
    
#datapath= "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"

Tb = readnetcdftoarr(datapath+"Optris_data/Tb_stab_cut_red_20Hz.nc")


Tb = create_tst_mean(Tb, moving_mean_size = 10)



writeNetCDF("/mnt/Seagate_Drive1/out_test/", "Tb_stab_cut_red_20Hz_mean10s.nc", "Tb_mean", Tb)



# create_tst_pertubations has moving_mean_size setting as well standart is 
Tb_pertub = create_tst_pertubations_mm(Tb, moving_mean_size = 600)

print(np.shape(Tb_pertub))
#Tb_pertub = np.fliplr(Tb_pertub)



writeNetCDF("/mnt/Seagate_Drive1/out_test/", "Tb_20Hz_pertub_30s_py.nc", "Tb_pertub", Tb_pertub)


imgplot = plt.imshow(Tb_pertub[1])
plt.show()
