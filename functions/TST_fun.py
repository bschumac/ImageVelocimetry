import numpy as np
import progressbar
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import scipy.cluster.vq as scv
import h5py
from matplotlib import cm 
from joblib import Parallel, delayed
import multiprocessing
from skimage.util.dtype import dtype_range
from scipy.signal import convolve2d
from numpy import log
from openpiv_fun import *
import sklearn.cluster
import copy


def calcwinddirection(u,v): 
    
    erg_dir_rad = np.arctan2(u, v)
    erg_dir_deg = np.degrees(erg_dir_rad)
    erg_dir_deg_pos = np.where(erg_dir_deg < 0.0, erg_dir_deg+360, erg_dir_deg)

    return(np.round(erg_dir_deg_pos,2))


def calcwindspeed(v,u): 
    ws = np.sqrt((u*u)+(v*v))
    return(np.round(ws,2))




def create_tst_pertubations_mm(array, moving_mean_size = 60):
    # creates a moving mean around each layer in array   

    resultarr = np.zeros(np.shape(array))
    bar = progressbar.ProgressBar(maxval=len(array), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for i in range(0,len(array)):
        # moving mean array = actarray:
        if i == 0:
            actarray = array[0:moving_mean_size*2+1]
        elif i != 0 and i != len(array) and i-(moving_mean_size)>= 0 and i+(moving_mean_size)<= len(array)-1:
            actarray = array[int(i-moving_mean_size):int(i+moving_mean_size)+1]
        elif i-(moving_mean_size)<= 0:
            actarray = array[0:moving_mean_size*2+1]   
        elif i+(moving_mean_size)>= len(array):
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]        
        if i == len(array)-1:
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]
        
        resultarr[i] = array[i]-np.mean(actarray, axis=0)
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(resultarr)



def writeNetCDF(out_dir, out_name, varname, array):
    
    # the output array to write will be nx x ny
    if len(array.shape) == 2:
        nx = array.shape[0]; ny = array.shape[1];
    else:
        nx = array.shape[1]; ny = array.shape[2]; nz = array.shape[0]
    # open a new netCDF file for writing.
    if out_dir[-1] == "/":
        file_dest = out_dir+out_name
    else:
        file_dest = out_dir+"/"+out_name
        
    ncfile = Dataset(file_dest,'w') 

    # create the x and y dimensions.
    ncfile.createDimension('x',nx)
    ncfile.createDimension('y',ny)
    if not len(array.shape) == 2:
        ncfile.createDimension('z',nz)
    # create the variable (4 byte integer in this case)
    # first argument is name of variable, second is datatype, third is
    # a tuple with the names of dimensions.
    if len(array.shape) == 2:
        data = ncfile.createVariable(varname,'f4',('x','y'))
    else:
        data = ncfile.createVariable(varname,'f4',('z','x','y'))

    # write data to variable.
    data[:] = array

        # close the file.
    ncfile.close()
    print ('*** SUCCESS writing example file'+ file_dest)


def write_png_files_from_Tb(datapath,Tb_filename,out_fld_name,my_dpi=100,verbose = True):   
    file = h5py.File(datapath+Tb_filename,'r')
    Tb = file.get("Tb")
    
    Tb_MIN = float(np.amin(Tb))
    Tb_MAX = float(np.amax(Tb))
    Tb_MIN = Tb_MIN - 1
    Tb_MAX = Tb_MAX + 1
    
    
    plt.rc('text', usetex=False)
    
    Tb = np.rot90(Tb,-1, axes=(1,-1))
    Tb = np.flip(Tb,-1)
    for i in range(0,len(Tb)):
        if verbose:
            print("Writing File " +str(i)+".png from "+str(len(Tb)))
        fig = plt.figure(figsize=(382/my_dpi, 288/my_dpi), dpi=my_dpi)
        ax1 = plt.subplot(111)
        u_xz = Tb[i]
        u_xz_norm = u_xz
        #u_xz_norm =  (u_xz - Tb_MIN) / (Tb_MAX-Tb_MIN)
        im=ax1.imshow(u_xz_norm,interpolation=None,cmap=cm.jet)
        ax1.axis("off")
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.savefig(datapath+out_fld_name+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
        plt.close()
        
 





def ssim(X,Y, axis = (1,2)):
        dmin, dmax = dtype_range[X.dtype.type]
        X.astype(np.float64)
        Y.astype(np.float64)
        K1 = 0.01
        K2 = 0.03
        L1= dmax - dmin
        #L2 = np.max(Y,axis=(1,2))-np.min(Y,axis=(1,2))
        C1 = (L1*K1)**2
        C2 = (L1*K2)**2
        mx = np.mean(X,axis=axis)
        my = np.mean(Y,axis=axis)
        mxy = np.mean(X*Y,axis=axis)
        mxx = np.mean(X*X,axis=axis)
        myy = np.mean(Y*Y,axis=axis)
        vx = np.var(X,axis=axis)
        vy = np.var(Y,axis=axis)
      
      
       
        cvxy = (mxy - mx * my)
        A1, A2, B1, B2 = ((2 * mx * my + C1,
               2 * cvxy + C2,
               mx ** 2 + my ** 2 + C1,
               vx + vy + C2))
        D = B1 * B2
        S = (A1 * A2) / D
        return S



# funktion calc_aiv inclusive method und auto window True/False

def calc_AIV_parallel(n, array, method = "greyscale"):
    # runs over all given bands
    #n = 0
    print("Processing Image")
    print(n)
    print("----------------")
    u = np.zeros((array.shape[1],array.shape[2]))
    v = np.zeros((array.shape[1], array.shape[2]))
    






def calc_aiv(frame_a, frame_b, window_size_x, window_size_y, overlap, max_displacement=None , corr_method="fft", subpixel_method='gaussian'):
    # fehlt: kmeans segmentation to determine window size
    # 
    
    if corr_method=="fft" or corr_method=="direct":
        search_area_size_x = window_size_x
        search_area_size_y = window_size_y
    else:
        search_area_size_x = window_size_x+round(max_displacement*overlap)
        search_area_size_y = window_size_y+round(max_displacement*overlap)
    
    
    if overlap >= window_size:
        raise ValueError('Overlap has to be smaller than the window_size')
    
    if search_area_size < window_size:
        raise ValueError('Search size cannot be smaller than the window_size')
        
    if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
        raise ValueError('window size cannot be larger than the image')
        
    
    if corr_method=="fft" or corr_method=="direct":
        u, v = cross_correlation_aiv(frame_a, frame_b, window_size = window_size_x, overlap = overlap, corr_method = corr_method, subpixel_method = subpixel_method)
        x, y = get_coordinates( image_size=frame_a.shape, window_size=window_size, overlap=overlap )
        return x, y, u, v
    else:
        u, v = window_correlation_aiv(frame_a, frame_b, window_size_x, window_size_y, overlap, search_area_size_x=search_area_size_x, search_area_size_y=search_area_size_y, corr_method=corr_method, subpixel_method=subpixel_method)
        x, y = get_coordinates( image_size=frame_a.shape, window_size=window_size, overlap=overlap )
        return x, y, u, v
        
    
    
    
def window_correlation_tiv(frame_a, frame_b, window_size_x, overlap, corr_method, search_area_size_x, search_area_size_y=0,  window_size_y=0):
    
    #search_area_size_x = search_area_size 
    window_size = window_size_x
   # print(window_size-((search_area_size_x-window_size)/2))
    if not (window_size-((search_area_size_x-window_size)/2))<= overlap:
        raise ValueError('Overlap or SearchArea has to be bigger: ws-(sa-ws)/2)<=ol')
        
     
    
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size_x, overlap )    
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))
    
    
    

    for k in range(n_rows):
        #k = 0# range(range(search_area_size/2, frame_a.shape[0] - search_area_size/2, window_size - overlap ):
        for m in range(n_cols):
            #k = 0# r
            #m = 0
            # range(search_area_size/2, frame_a.shape[1] - search_area_size/2 , window_size - overlap ):
            # Select first the largest window, work like usual from the top left corner
            # the left edge goes as: 
            # e.g. 0, (search_area_size - overlap), 2*(search_area_size - overlap),....
            il = k*(search_area_size_x - overlap)
            ir = il + search_area_size_x
            
            # same for top-bottom
            jt = m*(search_area_size_x - overlap)
            jb = jt + search_area_size_x
            
            # pick up the window in the second image
            window_b = frame_b[il:ir, jt:jb]            
            window_a_test = frame_a[il:ir, jt:jb]    
            
            rolling_wind_arr = moving_window_array(window_b, window_size, overlap)
            
            
            # now shift the left corner of the smaller window inside the larger one
            il += (search_area_size_x - window_size)//2
            # and it's right side is just a window_size apart
            ir = il + window_size
            # same same
            jt += (search_area_size_x - window_size)//2
            jb =  jt + window_size
        
            window_a = frame_a[il:ir, jt:jb]
            
            rolling_wind_arr_test = moving_window_array(window_a_test, window_size, overlap)

            window_a_test.itemsize
            rep_window_a = np.repeat(window_a[ :, :,np.newaxis], rolling_wind_arr.shape[0], axis=2)
            rep_window_a = np.rollaxis(rep_window_a,2)
           
#            test = rep_window_a - rolling_wind_arr_test
#            for i in range(0,81):
#                if test[40]==0:
#                    print(i)
#            
            #plt.imshow(rolling_wind_arr[1], vmin=2.5, vmax=4)
            #plt.colorbar()
#            plt.imshow(window_a_test[4:12,4:12], vmin=2, vmax=4)
#            
            #plt.imshow(window_b, vmin=2.5, vmax=4)
            #plt.imshow(rolling_wind_arr_test[], vmin=2, vmax=4)
            #plt.colorbar()
            #plt.imshow(window_a_test, vmin=2.5, vmax=4)
            #plt.colorbar()
            
            
            
            if corr_method == "greyscale": 
            
                dif = rep_window_a - rolling_wind_arr
                dif_sum = np.sum(abs(dif),(1,2))
                #dif_sum_min_idx = np.argmin(dif_sum)
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(dif_sum, (shap,shap))
                dif_sum_reshaped = (dif_sum_reshaped*-1)+np.max(dif_sum_reshaped)
                #plt.imshow(dif_sum_reshaped)
                
                
                
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                #row =  row -(((search_area_size_x - window_size)//2))
                #col =  col -(((search_area_size_x - window_size)//2))            
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                #print("Row")
                #print(row)
                #print("Col")
                #print(col)

                u[k,m],v[k,m] = col, row
            
            if corr_method == "rmse":         
                rmse = np.sqrt(np.mean((rolling_wind_arr-rep_window_a)**2,(1,2)))
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                shap = int(np.sqrt( rep_window_a.shape[0]))
                rmse_reshaped = np.reshape(rmse, (shap,shap))
                rmse_reshaped = (rmse_reshaped*-1)+np.max(rmse_reshaped)
                #plt.imshow(dif_sum_reshaped)
                
                row, col = find_subpixel_peak_position(rmse_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                #row =  row -(((search_area_size_x - window_size)//2))
                #col =  col -(((search_area_size_x - window_size)//2))             
                
                #row = lookup_direc_lst[dif_sum_min_idx][0]
                #col = lookup_direc_lst[dif_sum_min_idx][1]
                u[k,m],v[k,m] = col, row
                

            if corr_method == "ssim":
                ssim_lst = ssim(rolling_wind_arr,rep_window_a)
                #dif_sum_min_idx = np.argmax(ssim_lst)
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(ssim_lst, (shap,shap))
                #row = lookup_direc_lst[dif_sum_min_idx][0]
                #col = lookup_direc_lst[dif_sum_min_idx][1]
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                #row =  row -(((search_area_size_x - window_size)//2)) 
                #col =  col -(((search_area_size_x - window_size)//2))  
                u[k,m],v[k,m] = col, row
    
    return u, v        
    

    

def tst_kmeans(dataset,n_clusters = 8, outpath="", my_dpi = 100):
    kmean_arr = copy.copy(dataset)
    labels= []
    centers = []
    for i in range(0,len(dataset)):
        print(i)
        
              
        arr= dataset[i]
        
        original_shape = arr.shape # so we can reshape the labels later
        
        samples = np.column_stack([arr.flatten()])
        
        clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
        lab = clf.fit_predict(samples).reshape(original_shape)
        labels.append(lab)
        centers.append(clf.cluster_centers_)
     
    
    
    baseline_center = centers[0]
    mean_centers = []
    
    minarray = np.zeros((n_clusters,2))
    
    for i in range(0,len(centers[0])):
        minarray[i,0] = i
        act_number = baseline_center[i]
        collected_numbers=[]
        for center_arr in centers[1:]:     
            for k in range(0,len(centers[0])):
                minarray[k,1] = abs(center_arr[k]-act_number)
            
            #print(center_arr[(np.argmin(minarray[:,1]))])
            #print(min(minarray[:,1]))
            collected_numbers.append(center_arr[(np.argmin(minarray[:,1]))])
        mean_centers.append(np.mean(collected_numbers))
    
    
    b = np.zeros((n_clusters,1))
    b[:,0] = mean_centers            
    
    
    for i in range(0,len(dataset)):
        print(i)
        
             
        arr= dataset[i]
        
        original_shape = arr.shape # so we can reshape the labels later
        
        samples = np.column_stack([arr.flatten()])
        
        clf = sklearn.cluster.KMeans(n_clusters=n_clusters, init=b)
        lab = clf.fit_predict(samples).reshape(original_shape)
        lab = np.flipud(lab)
        labels.append(lab)
        kmean_arr[i] = lab
        if outpath:
            fig = plt.figure(figsize=(382/my_dpi, 288/my_dpi), dpi=my_dpi)
            ax1 = plt.subplot(111)
            im=ax1.imshow(lab,interpolation=None, cmap = cm.gray)
            plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
            plt.close()
    return(kmean_arr)
        












### DEPRICATED USE image_to_val instead!





def calc_TIV(n, pertub, iterx, itery, interval = 1, max_displacement=25, inter_window_size_y= 35, inter_window_size_x=35 ,inter_window_size_dif_x=10, inter_window_size_dif_y=10, iter_lst = [], method = "greyscale"):
    # runs over all given bands
    #n = 0
    print("Processing Image")
    print(n)
    print("----------------")
    u = np.zeros((pertub.shape[1],pertub.shape[2]))
    v = np.zeros((pertub.shape[1], pertub.shape[2]))
    
    iterx = max_displacement+inter_window_size_dif_x -1

    for l in range(max_displacement+inter_window_size_dif_y,(pertub.shape[1])-(max_displacement+inter_window_size_dif_x)):
        # runs over all number of columns
        #print(l)
        #print("of "+str((pertub.shape[1])-(max_displacement+inter_window_size_dif_x)))
        #l = 9
        centery = l
        iterx+=1
        itery = max_displacement+inter_window_size_dif_x -1

        for m in range(max_displacement+inter_window_size_dif_x,(pertub.shape[2])-(max_displacement+inter_window_size_dif_y)):
            # runs over all number of rows
            #m = 10
            centerx = m
            itery+=1

            disp_arr = pertub[n+interval][centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
            rolling_wind_arr = rolling_window(disp_arr, (inter_window_size_y, inter_window_size_x))
            
            rolling_wind_arr_reshaped = np.reshape(rolling_wind_arr,(rolling_wind_arr.shape[0]*rolling_wind_arr.shape[1],rolling_wind_arr.shape[2],rolling_wind_arr.shape[3]))
            
            #inter_wind_disp_arr = pertub[n][centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
            inter_wind1 = pertub[n][centery-inter_window_size_dif_y:centery+inter_window_size_dif_y+1,centerx-inter_window_size_dif_x:centerx+inter_window_size_dif_x+1]
            rep_inter_wind1 = np.repeat(inter_wind1[ :, :,np.newaxis], rolling_wind_arr.shape[0]*rolling_wind_arr.shape[1], axis=2)
            rep_inter_wind1 = np.rollaxis(rep_inter_wind1,2)
            
            
            # displace interigation window with rolling window
            
           
            
            if method == "greyscale": 
            
                dif = rep_inter_wind1 - rolling_wind_arr_reshaped
                dif_sum = np.sum(abs(dif),(1,2))
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                dif_sum_min_idx = np.argmin(dif_sum)
                dif_sum_min = min(dif_sum)
                v[iterx,itery] = iter_lst[dif_sum_min_idx][0]
                u[iterx,itery] = iter_lst[dif_sum_min_idx][1]
                if np.partition(dif_sum, 4)[4] == dif_sum_min  and np.partition(dif_sum, 5)[5] == dif_sum_min:
                    v[iterx,itery] = 0
                    u[iterx,itery] = 0
            
            if method == "rmse":         
                rmse = np.sqrt(np.mean((rolling_wind_arr_reshaped-rep_inter_wind1)**2,(1,2)))
               
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                dif_sum_min_idx = np.argmin(rmse)
                dif_sum_min = min(rmse)
                v[iterx,itery] = iter_lst[dif_sum_min_idx][0]
                u[iterx,itery] = iter_lst[dif_sum_min_idx][1]
                if np.partition(rmse, 4)[4] == dif_sum_min  and np.partition(rmse, 5)[5] == dif_sum_min:
                    v[iterx,itery] = 0
                    u[iterx,itery] = 0

            if method == "ssim":
                ssim_lst = ssim(rolling_wind_arr_reshaped,rep_inter_wind1)
                dif_sum_min_idx = np.argmax(ssim_lst)
                v[iterx,itery] = iter_lst[dif_sum_min_idx][0]
                u[iterx,itery] = iter_lst[dif_sum_min_idx][1]
             
            
    return(np.stack((u, v)))


def streamplot_old(U, V, topo = None, vmin = -2, vmax = 2):
    import copy
    fig = plt.figure()
    if topo is None:
        topo = copy.copy(U)
    
    I = plt.imshow(topo, cmap = "rainbow",vmin=vmin, vmax=vmax)
    fig.colorbar(I)
    X = np.arange(0, U.shape[1], 1)
    Y = np.arange(0, U.shape[0], 1)
    X1 = 125
    Y1 = 75
    
    speed = np.sqrt(U*U + V*V)
    
    lw = 2*speed / speed.max()
    Q = plt.streamplot(X, Y, U, V, density=1, color='k', linewidth=lw)
    E = plt.quiver(X1, Y1, np.mean(U)*20, np.mean(V)*20*-1, color = "red",width=0.01, headwidth=3, scale=1000)
    return(fig) 
        


def streamplot(U, V, X, Y, topo = None, vmin = -2, vmax = 2):
    
    
    if topo is None:
        topo = copy.copy(U)
    fig = plt.figure()
    I = plt.imshow(topo, cmap = "rainbow",vmin=vmin, vmax=vmax)
    fig.colorbar(I)

    speed = np.sqrt(U*U + V*V)   
    lw = 2*speed / speed.max()
    Q = plt.streamplot(X, Y, U, V, density=1, color='k', linewidth=lw)
    #E = plt.quiver(X1, Y1, np.mean(U)*20, np.mean(V)*20*-1, color = "red",width=0.01, headwidth=3, scale=1000)
    return(fig) 





def calc_TIV_test(n, pertub, iterx, itery, interval = 1, max_displacement=25, inter_window_size_y= 35, inter_window_size_x=35 ,inter_window_size_dif_x=10, inter_window_size_dif_y=10, iter_lst = [], method = "greyscale+"):
    # runs over all given bands
    n=0
    print("Processing Image")
    print(n)
    print("----------------")
    u = np.zeros((pertub.shape[1],pertub.shape[2]))
    v = np.zeros((pertub.shape[1], pertub.shape[2]))
    
    iterx = max_displacement+inter_window_size_dif_x -1

    for l in range(max_displacement+inter_window_size_dif_y,(pertub.shape[1])-(max_displacement+inter_window_size_dif_x)):
        # runs over all number of columns
        l = 14
        print(l)
        print("of "+str((pertub.shape[1])-(max_displacement+inter_window_size_dif_x)))
      
        centery = l
        iterx+=1
        itery = max_displacement+inter_window_size_dif_x -1
        method = "direct"
        
        for m in range(max_displacement+inter_window_size_dif_x,(pertub.shape[2])-(max_displacement+inter_window_size_dif_y)):
            # runs over all number of rows
            #m = 43
            print(m)
            centerx = m
            itery+=1

            disp_arr = pertub[n+interval][centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
            inter_wind1 = pertub[n][centery-inter_window_size_dif_y:centery+inter_window_size_dif_y+1,centerx-inter_window_size_dif_x:centerx+inter_window_size_dif_x+1]
            #plt.imshow(disp_arr)
            #plt.imshow(inter_wind1)
            if method == "fft": 
                corr = correlate_windows(inter_wind1, disp_arr,
                                         corr_method=method, 
                                         nfftx=nfftx, nffty=nffty)
                
            
            if method == "direct": 
                 corr = correlate_windows(inter_wind1, disp_arr,
                                         corr_method=method, 
                                         nfftx=None, nffty=None)
            
            #plt.imshow(corr)    
            #plt.colorbar()
            #plt.imshow(pertub[0])
            
            row, col = find_subpixel_peak_position(corr, subpixel_method="gaussian")
            col = np.argmax(np.max(corr, axis=1))
            row = np.argmax(np.max(corr, axis=0))        
            search_area_size = disp_arr.shape[0]
            window_size = inter_window_size_x
            row -=  (search_area_size + window_size - 1)//2
            col -=  (search_area_size + window_size - 1)//2

            # get displacements, apply coordinate system definition
            try:
                v[iterx,itery] =  row
                u[iterx,itery] = -col
            except:
                pass
            #u[k,m],v[k,m] = -col, row

      
            
    return(np.stack((u, v)))







def colormap2arr(arr,cmap):    
    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    gradient=cmap(np.linspace(0.0,1.0,100))
    #gradient= gradient[:,0:3]
    print(np.shape(gradient))
    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
    #arr2 = arr2[:,0:3]
    print(np.shape(arr2))
    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    code,dist=scv.vq(arr2,gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values=code.astype('float')/gradient.shape[0]

    # Reshape values back to (240,240)
    values=values.reshape(arr.shape[0],arr.shape[1])
    values=values[::-1]
    return values

def covert_image_to_values(image_file, colormap, value_min, value_max):
    #use iamgefile and colormap and vmin, vmax to retrive the actual value,
    #return vaules as numpy array
    #image_file: file path as string, colormap, matplotlib colormap instance, vmin & vmax float values
    arr=plt.imread(image_file)
    values=colormap2arr(arr,colormap)
    print(values)
    values= (values*(value_max-value_min))+value_min
    return values


