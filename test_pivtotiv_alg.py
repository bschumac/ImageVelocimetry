from TST_fun import *
from openpiv_fun import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm
from joblib import Parallel, delayed
import multiprocessing
from TST_fun import *
import datetime
from math import sqrt
import copy
import scipy as sio
import scipy.stats as stats


datapath = "/home/benjamin/Met_ParametersTST/PALMS/data/"
outpath = datapath+"tiv/"

#case = "strong_wind"
case = "no_wind"
#case = "vertex_shedding"



if case == "strong_wind":
    file = h5py.File(datapath+'cbl_surf.nc','r')
elif case == "no_wind": 
    file = h5py.File(datapath+'cbl_nowind.nc','r')
elif case == "vertex_shedding":
    file = h5py.File(datapath+'shedding.nc','r')
    


u_model = file.get("u_xy")
v_model = file.get("v_xy")



if case == "strong_wind":
       
    u_model = np.reshape(u_model,(3600,256,512))
    v_model = np.reshape(v_model,(3600,256,512))

elif case == "no_wind":
       
    u_model = np.reshape(u_model,(3600,256,512))
    v_model = np.reshape(v_model,(3600,256,512))
    
        
elif case == "vertex_shedding": 
    u_model = np.reshape(u_model,(3600,160,640))
    v_model = np.reshape(v_model,(3600,160,640))
   

begin_value_images = 3540
interval=6
end_value_images = begin_value_images+interval+1

u_subset = u_model[begin_value_images:end_value_images]
v_subset = v_model[begin_value_images:end_value_images]

windspeed = copy.copy(u_model)

for j in range(0,len(windspeed)):
    windspeed[j] = calcwindspeed(v_model[j],u_model[j])


windspeed_subset = windspeed[begin_value_images:end_value_images]







frame_a = windspeed[0]

frame_b = windspeed[1]

plt.imshow(frame_a)

plt.imshow(frame_b)

datapath = "/home/benjamin/Met_ParametersTST/PALMS/test_iv_crosscorr/"

window_size=24
overlap = 23
search_area_size = 36

my_dpi = 100

for i in range(begin_value_images-120,end_value_images):
   
    frame_a = windspeed[i]
    frame_b = windspeed[i+1]
    
    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                              dt=1, search_area_size=search_area_size, nfftx=None, 
                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
    
    
    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )

    u1=np.flip(u1,0)
    v1=np.flip(v1,0)
    
    plt.figure()
    plt.imshow(frame_a)
    plt.quiver(x,y,u,v)
    plt.savefig(datapath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()

datapath = "/home/benjamin/Met_ParametersTST/PALMS/test_iv_greyscale/"



for i in range(begin_value_images-120,end_value_images):
    #i = begin_value_images   
    frame_a = windspeed[i]
    frame_b = windspeed[i+1]


    u1, v1= window_correlation_aiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="ssim")
    
    
    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
    
    
    plt.figure()
    plt.imshow(frame_a)
    plt.quiver(x1,y1,u1,v1)
    #plt.show()
    
    plt.savefig(datapath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()



window_size=48
overlap = 47
search_area_size = 64


i = begin_value_images   
frame_a = windspeed[i]
frame_b = windspeed[i+1]

u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                       search_area_size_x=search_area_size, search_area_size_y=0, corr_method="greyscale")


x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )


plt.figure()
plt.imshow(frame_a)
plt.quiver(x1,y1,u1,v1)
plt.show()








kmeans_test = tst_kmeans(dataset=windspeed_subset,n_clusters = 6)
plt.imshow(kmeans_test[1])
plt.colorbar()




from skimage import measure



# Find contours at a constant value of 0.8
contours = measure.find_contours(kmeans_test[1], 0)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(kmeans_test[1], interpolation='nearest', cmap=plt.cm.gray)


for n, contour in enumerate(contours):
    if n>4 and n <= 7:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)



ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


e = enumerate(contours)


contours[-1]
for n, contour in enumerate(contours):
    print(contour)




np.save("/home/benjamin/Met_ParametersTST/PALMS/data/test_kmeans.npy",kmeans_test[1])



window_size=8
overlap = 7
search_area_size = 16



frame_a = windspeed[begin_value_images]

frame_b = windspeed[begin_value_images+1]

plt.imshow(frame_a-frame_b)

u, v= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="ssim")
        
    
x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )


plt.figure()
plt.imshow(frame_a)
plt.quiver(x,y,u,v)
plt.show()


u_model_cut = get_org_data(frame_a=u_model[begin_value_images], search_area_size=search_area_size, overlap=overlap )

v_model_cut = get_org_data(frame_a=v_model[begin_value_images], search_area_size=search_area_size, overlap=overlap )

plt.imshow(u_model_cut)
plt.colorbar()

plt.imshow(v_model_cut)
plt.colorbar()

u[numpy.isnan(u)]=0
v[numpy.isnan(v)]=0
plt.imshow(u*2, vmin = -2, vmax= 2)
plt.colorbar()
plt.imshow(v*2, vmin = -2, vmax= 2)
plt.colorbar()
dif_u = u*2-u_model_cut
dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
n, x, _= plt.hist(dif_flatten, bins=np.linspace(-6, 6, 50), histtype=u'step', density=True)

density = stats.gaussian_kde(dif_flatten)
plt.plot(x, density(x))
plt.show()

ssim(u*2,u_model_cut, axis= (0,1))








streamplot(u,v)
 
sig2noise_method='peak2peak'

    # if we want sig2noise information, allocate memory
    if sig2noise_method is not None:
        sig2noise = np.zeros((n_rows, n_cols))


sig2noise_method=None
               
                # get signal to noise ratio
                if sig2noise_method is not None:
                    sig2noise[k,m] = sig2noise_ratio(
                        corr, sig2noise_method=sig2noise_method, width=width)
                
    
    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        return u/dt, v/dt, sig2noise
    else:
        return u/dt, v/dt

