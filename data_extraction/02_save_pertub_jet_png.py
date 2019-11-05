
from TST_fun import write_png_files_from_Tb
datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
Tb_filename = "Tb_1Hz.mat"
out_fld_name = "figure_Tb_RGB/"

write_png_files_from_Tb(datapath,Tb_filename,out_fld_name)
