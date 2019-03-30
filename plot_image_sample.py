import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors, cm
#tmp=xr.open_dataset("./Tb_Telops_burn_60Hz.mat")
tmp=xr.open_dataset("./ir.nc")

# plot x-z u
from mpl_toolkits.axes_grid1 import make_axes_locatable
for i in range(2000,3000):
    fig = plt.figure(figsize=(9,8))
    ax1 = plt.subplot(111)
    u_xz = tmp.Tb_Telops_burn_60Hz[i,:,:].T
    im=ax1.imshow(u_xz,vmin=300,vmax=1000,origin="upper",cmap="jet")
    ax1.axis("off")
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("/home/rccuser/simulation/ir_image/image/"+str(i)+".png",bbox_inches='tight',pad_inches = 0,transparent=True)
    plt.close()
