
import sys
sys.path.insert(1, '/home/rccuser/.jupyter/code/ImageVelocimetry/functions/')
from TST_fun import *
rec_freq = 80
out_freq = 80
datapath_csv = "/data/FIRE/Darfield_2019/Tier01/O80_220319_high_P2/"
outpath_tb_org = "/data/FIRE/Darfield_2019/Tier01/"
start_img = 5751
end_img = 20331

print("START P2..")
fls = os.listdir(datapath_csv)
arr = readcsvtoarr(datapath_csv_files=datapath_csv,start_img=start_img,end_img=end_img,interval=int(rec_freq/out_freq))
writeNetCDF(outpath_tb_org, "P2_Tb_org_"+str(rec_freq)+"Hz.nc", "Tb", arr)
print("DONE!")