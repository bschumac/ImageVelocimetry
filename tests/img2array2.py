import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.cluster.vq as scv

def colormap2arr(arr,cmap):    
    gradient=cmap(np.linspace(0.0,1.0,1000))
    gradient= gradient[:,0:3]

    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
    arr2 = arr2[:,0:3]

    code,dist=scv.vq(arr2,gradient)

    values=code.astype('float')/gradient.shape[0]

    values=values.reshape(arr.shape[0],arr.shape[1])
    values=values[::-1]
    return values

for i in range(200,837):
    arr=plt.imread('./stable_sample/0'+str(i)+'.png')
    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
    values=colormap2arr(arr,cm.jet)
    values=values*(1200-320)+320
    fig = plt.figure(figsize=(6.4,5.12),dpi=100)
    ax1 = plt.subplot(111)
    im=ax1.imshow(values,interpolation=None, cmap=cm.jet,
                  origin='lower')
    plt.colorbar(im)
    plt.savefig("./stable_sample/retrive_"+str(i)+".png")
