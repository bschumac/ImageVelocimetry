import numpy as np
import progressbar
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import copy
import os

from matplotlib import cm 
from joblib import Parallel, delayed
import multiprocessing
from skimage.util.dtype import dtype_range
from scipy.signal import convolve2d
from numpy import log
from openpiv_fun import *
import h5py
from statistics import mode 
from collections import Counter
from hht import hht
import copy


from PyEMD.EEMD import *
import math
from scipy.fftpack import *
import gc
from PIL import Image


from scipy import interpolate

def interpolate_nan(arr):
    """
    Interpolate pixels of missing data within a 3d numpy array: 1st dimension is time.
    Parameters
    ----------
    arr : np.ndarray
        3d array containing data (time, x,y)
    Returns
    -------
    
    out_array : np.ndarray
        3d array with interpolated data
    """
    
    out_array = np.empty(arr.shape)
    bar = progressbar.ProgressBar(maxval=arr.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for i in range(0,len(arr)):
        layer = arr[i]
        if np.sum(np.isnan(layer))!=0:
            x = np.arange(0, layer.shape[1])
            y = np.arange(0, layer.shape[0])
            #mask invalid values
            layer_masked = np.ma.masked_invalid(layer)

            xx, yy = np.meshgrid(x, y)
            ##get only the valid values
            x1 = xx[~layer_masked.mask]
            y1 = yy[~layer_masked.mask]
            newarr = layer_masked[~layer_masked.mask]

            GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                      (xx, yy),
                                         method='cubic')
            out_array[i] = GD1
        else:
            out_array[i] = layer
        bar.update(bar_iterator+1)
        bar_iterator += 1
    bar.finish()
    return(out_array)



def to_npy_info(dirname, dtype, chunks, axis):
    with open(os.path.join(dirname, 'info'), 'wb') as f:
        pickle.dump({'chunks': chunks, 'dtype': dtype, 'axis': axis}, f)


def calcwinddirection(u,v): 
    
    erg_dir_rad = np.arctan2(u, v)
    erg_dir_deg = np.degrees(erg_dir_rad)
    erg_dir_deg_pos = np.where(erg_dir_deg < 0.0, erg_dir_deg+360, erg_dir_deg)

    return(np.round(erg_dir_deg_pos,2))


def calcwindspeed(v,u): 
    ws = np.sqrt((u*u)+(v*v))
    return(np.round(ws,2))



def readnetcdftoarr(datapath_to_file, var = 'Tb'):
    file = h5py.File(datapath_to_file,'r')
    arr = file.get(var)
    nparr = np.array(arr)
    return(nparr)


def removeSteadyImages(arr, rec_feq = 27, print_out = True):
    try:
        for i in range(0, len(arr)):
            act_val = arr[i]
            if np.all(act_val == arr[i:i+rec_feq]):
                if print_out:
                    print("yes")
                    print(i)
                #print(len(arr))
                arr = np.delete(arr,[range(i,i+rec_feq)],0)
                #print(len(arr))
    except:
        pass
    return(arr)


def readcsvtoarr(datapath_csv_files,start_img=0,end_img=0,interval=1):
    fls = os.listdir(datapath_csv_files)
    fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
    if end_img == 0:
        end_img = len(fls)-1
    
    counter = 0
    
    for i in range(start_img,end_img, interval): 
        
        if counter%100 == 0:
            print(str(counter)+" of "+str((end_img-start_img)/interval))
        my_data = np.genfromtxt(datapath_csv_files+fls[i], delimiter=',', skip_header=1)
        my_data = np.reshape(my_data,(1,my_data.shape[0],my_data.shape[1]))
        if counter == 0:
            org_data = copy.copy(my_data)
        else:
            org_data = np.append(org_data,my_data,0)
        #org_data[counter] = my_data 
        counter+=1
    
    return(org_data)


def create_tst_subsample_mean(array, size=9):
    cut_to = int(np.floor(len(array)/size)*size)
    array = array[0:cut_to]
    split_size = len(array)/size
    a_split = np.array_split(array,split_size)
    a_split_avg = np.array([np.mean(arr,0) for arr in a_split])
    return(a_split_avg)
    

def create_tst_subsample(array, size = 9):
    return(array[1::size])
    
def create_tst_mean(array, moving_mean_size = 60):
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
        
        resultarr[i] = np.mean(actarray, axis=0)
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(resultarr)






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







def MatToNpy(matfld, npyfld):
    """
    Transfer .mat files (version 7.3) into numpy files making them accesable for dask arrays. 
     
    
    Parameters
    ----------
    matfld: string
        The folder path where the mat files are stored.
    npyfld: string
        The folder path where the npy files will be stored
    
    
    """
    fls = os.listdir(matfld)
    print("Files to read:")
    print(len(fls))
    counter = 0
    for file in fls:
        print(counter)
        arrays = {}
        f = h5py.File(filepath+file)
        for k, v in f.items():
            arrays[k] = np.array(v)
        
        file = file.replace(".mat", "")
        arr = arrays["Data_im"]
        np.save(npyfld+file, arr)
        counter +=1




def find_interval(signal, fs, imf_no = 1):
    """
    Compute the interval setting for the TIV. 
    Calculating a hilbert transform from the instrinct mode functions (imfs).
    This is based on the hilbert-huang transform assuming non-stationarity of the given dataset.
    The function returns a suggested interval (most powerful frequency) based on the weighted average. 
    The weights are the calculated instantaneous energies. 
    
    Parameters
    ----------
    signal: 1d np.ndarray
        a one dimensional array which contains the brightness temperature (perturbation) of one pixel over time.
    fs: int 
        the fps which was used to record the imagery 
    imf_no: int (default 1)
        The imf which will be used to calculate the interval on. IMF 1 is the one with the highest frequency.
    Returns
    -------
    recommended_interval : float
        The found most powerful interval in float. Needs rounding to the next int.
    
    """
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    imf = imfs[imf_no-1,:]        
    
    from scipy.signal import hilbert
    sig = hilbert(imf)
    
    energy = np.square(np.abs(sig))
    phase = np.arctan2(sig.imag, sig.real)
    omega = np.gradient(np.unwrap(phase))
    
    omega = fs/(2*math.pi)*omega
    #omegaIdx = np.floor((omega-F[0])/FResol)+1;
    #freqIdx = 1/omegaIdx;
    
    insf = omega
    inse = energy
    
    rel_inse = inse/np.nanmax(inse)
    insf_weigthed_mean = np.average(insf,weights=rel_inse)
    insp = 1/insf_weigthed_mean
    recommended_interval = np.round(fs*insp,1)
    del(eemd)
    del(imfs)
    del(imf)
    del(sig)
    del(energy)
    del(phase)
    del(omega)
    del(insf)
    del(inse)
    del(rel_inse)
    gc.collect()
    return(recommended_interval)
    








def randomize_find_interval (data,  rec_freq = 1, plot_hht = False, outpath = "/", figname = "hht_fig"):
    """
    Compute the interval setting for the TIV. 
    Basically a wrapper function for the find_interval function
    
    
    Parameters
    ----------
    data: 3d np.ndarray
        a three dimensional array which contains the brightness temperature (perturbation).
    rec_freq: int (default 1)
        the fps which was used to record the imagery 
    plot_hht: boolean (default False) (not implemented yet!)
        Boolean flag to plot the results for review (not implemented yet!)
    outpath: string (default "/") (not implemented yet!)
        The outpath for the plots - only the last plot of 10 plots is saved in this directory
        Set to a proper directory when used with Boolean flag. (not implemented yet!)
    figname: string (default "hht_fig") (not implemented yet!)
        The output figure name. (not implemented yet!)
    Returns
    -------
    [mode(interval_lst),interval_lst] : list
        The found most occuring and powerful period, and the list which was used to calculate this
    
    """
    masked_boo = True     
    for i in range(0,11):
        while masked_boo:
            rand_x = np.round(np.random.rand(),2)
            rand_y = np.round(np.random.rand(),2)
            
            x = np.round(50+(225-50)*rand_x)
            y = np.round(50+((225-50)*rand_y))
            
            if plot_hht:
                print(x)
                print(y)
            
            
        
        #print(data.shape)
            pixel = data[:,int(x),int(y)]
            
            if np.isnan(np.sum(pixel)):
                masked_boo = True
            else:
                masked_boo = False
                
                
            
        #print(data.shape)
            
        
        act_interval1 = find_interval(pixel, rec_freq, imf_no = 1)
        act_interval2 = find_interval(pixel, rec_freq, imf_no = 2)
        
        act_intervals = [round(act_interval1),round(act_interval2)]
        act_intervals2 = [act_interval1,act_interval2]
        
        
        if i == 0:
            interval_lst = copy.copy([act_intervals])
            interval_lst2 = copy.copy([act_intervals2])
        else:
            interval_lst.append(act_intervals) 
            interval_lst2.append(act_intervals2)
            #np.append(interval_lst,act_interval,0)
    
    try:
        first_most = mode(list(zip(*interval_lst))[0])
    except:
        d_same_count_intervals = Counter(list(zip(*interval_lst))[0])
        d_same_count_occ = Counter(d_same_count_intervals.values())
        for value in d_same_count_occ.values():
            if value == 2:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:2])
            if value == 3:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:3])
            if value == 4:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:4])
            if value == 5:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:5])   
    try:
        second_most = mode(list(zip(*interval_lst))[1]) 
    except:
        sec_most_lst = list(zip(*interval_lst))[1]
        d_same_count_intervals = Counter(list(zip(*interval_lst))[1])
        # 
        try:
            if first_most in d_same_count_intervals.keys():
                sec_most_lst = np.delete(sec_most_lst, np.where(sec_most_lst == first_most))       
                second_most = mode(sec_most_lst)
            else:
                raise(ValueError())
        except:
            d_same_count_occ = Counter(d_same_count_intervals.values())
            for value in d_same_count_occ.values():
                if value == 2:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:2])
                if value == 3:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:3])
                if value == 4:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:4])
                if value == 5:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:5])            
 
    return([first_most, second_most, interval_lst2])




def read_stab_red_imgs(stab_path, subtraction=0):
    # needs hard improvement
    
    import os
    import progressbar
    import copy
    
    mean_dif = []
    length_fls = len(os.listdir(stab_path))
    bar = progressbar.ProgressBar(maxval=length_fls, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for i in range(1, length_fls):      
        
        #read_img = np.array(plt.imread(stab_path+format(i, '04d')+".tif")[:,:,:3])
        read_img = np.asarray(Image.open(stab_path+format(i, '04d')+".tif"))
        read_img = read_img[:,:,0]
        Tb_stab = (read_img/10)+subtraction

        
        #mean_error = np.nanmean(Tb_org[i-1,50:250,50:250] - Tb_stab[50:250,50:250])
        #mean_dif.append(mean_error)
        
        Tb_stab= np.reshape(Tb_stab,((1,Tb_stab.shape[0],Tb_stab.shape[1])))
        bar.update(bar_iterator+1)
        bar_iterator += 1
        
        if i == 1:
            Tb_stab_final =  copy.copy(Tb_stab)
    
        else:
            Tb_stab_final = np.append(Tb_stab_final,Tb_stab,0)
    
    return([Tb_stab_final,mean_dif])

        












def writeNetCDF(out_dir, out_name, varname, array):
    """
    Write an numpy array to disk. 
    
    
    Parameters
    ----------
    out_dir: string
        output directory
    out_name: string
        output file name
    varname: string
        variable name of the array in the netcdf file
    outpath: string (default "/")
        The outpath for the plots - only the last plot of 10 plots is saved in this directory
        Set to a proper directory when used with Boolean flag.
    figname: string (default "hht_fig")
        The output figure name.
    Returns
    -------
    [mode(interval_lst),interval_lst] : list
        The found most occuring and powerful period, and the list which was used to calculate this
    
    """
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

def calc_AIV_parallel(n, array, method = "greyscale", ):
    # runs over all given bands
    #n = 0
    print("Processing Image")
    print(n)
    print("----------------")
    u = np.zeros((array.shape[1],array.shape[2]))
    v = np.zeros((array.shape[1], array.shape[2]))
    
    
    






def calc_aiv(frame_a, frame_b, window_size_x, window_size_y, overlap, max_displacement=None , corr_method="fft", subpixel_method='gaussian'): # method = PIV/AIV
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
        
    
    
    
def window_correlation_tiv(frame_a, frame_b, window_size_x, overlap_window, overlap_search_area, corr_method, search_area_size_x, 
                           search_area_size_y=0,  window_size_y=0, mean_analysis = True, std_analysis = True, std_threshold = 10):
    #corr_method = method
    #search_area_size_x = search_area_size 
    window_size = window_size_x
   # print(window_size-((search_area_size_x-window_size)/2))
    if not (window_size-((search_area_size_x-window_size)/2))<= overlap_search_area:
        raise ValueError('Overlap or SearchArea has to be bigger: ws-(sa-ws)/2)<=ol')
        
     
    
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size_x, overlap_search_area )   
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))
    
    
    

    for k in range(n_rows):

        for m in range(n_cols):
            #print(m)
            #k = 1# r
            # range(search_area_size/2, frame_a.shape[1] - search_area_size/2 , window_size - overlap ):
            # Select first the largest window, work like usual from the top left corner
            # the left edge goes as: 
            # e.g. 0, (search_area_size - overlap), 2*(search_area_size - overlap),....
            il = k*(search_area_size_x - overlap_search_area)#k*(search_area_size_x - (search_area_size_x-1)) #
            ir = il + search_area_size_x
            
            # same for top-bottom
            jt = m*(search_area_size_x - overlap_search_area)#m*(search_area_size_x - (search_area_size_x-1)) #
            jb = jt + search_area_size_x
            
            # pick up the window in the second image
            window_b = frame_b[il:ir, jt:jb]            
            #plt.imshow(window_b)
            #plt.imshow(frame_b)
            #window_a_test = frame_a[il:ir, jt:jb]    
          
            rolling_wind_arr = moving_window_array(window_b, window_size, overlap_window)
            
            
            # now shift the left corner of the smaller window inside the larger one
            il += (search_area_size_x - window_size)//2
            # and it's right side is just a window_size apart
            ir = il + window_size
            # same same
            jt += (search_area_size_x - window_size)//2
            jb =  jt + window_size
        
            window_a = frame_a[il:ir, jt:jb]
            #plt.imshow(window_a)
            #rolling_wind_arr_test = moving_window_array(window_a_test, window_size, overlap)
            
            
            rep_window_a = np.repeat(window_a[ :, :,np.newaxis], rolling_wind_arr.shape[0], axis=2)
            rep_window_a = np.rollaxis(rep_window_a,2)
            

            
            

            
            
            if corr_method == "greyscale": 
            
                dif = rep_window_a - rolling_wind_arr
                dif_sum = np.sum(abs(dif),(1,2))
                #dif_sum_min_idx = np.argmin(dif_sum)
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(dif_sum, (shap,shap))
                dif_sum_reshaped = (dif_sum_reshaped*-1)+np.max(dif_sum_reshaped)
                #plt.imshow(dif_sum_reshaped)
                row, col = find_subpixel_peak_position(corr=dif_sum_reshaped)
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                #row =  row -(((search_area_size_x - window_size)//2))
                #col =  col -(((search_area_size_x - window_size)//2))            
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                #if row < 1000 or col < 1000:
                #    print(row)
                #    print(col)
                #    print(k)
                #   print(m)
                
                
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)): #and (not np.isnan(col) or not np.isnan(col)):
                #        print(k)
                #        print(m)
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                        
                
                
                u[k,m],v[k,m] = col, row
      
            
            if corr_method == "rmse":         
                rmse = np.sqrt(np.mean((rolling_wind_arr-rep_window_a)**2,(1,2)))
      
                shap = int(np.sqrt( rep_window_a.shape[0]))
                rmse_reshaped = np.reshape(rmse, (shap,shap))
                rmse_reshaped = (rmse_reshaped*-1)+np.max(rmse_reshaped)
               
                
                row, col = find_subpixel_peak_position(rmse_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                 
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)): #and (not np.isnan(col) or not np.isnan(col)):
                #        print(k)
                #        print(m)
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                

                u[k,m],v[k,m] = col, row
                

            if corr_method == "ssim":
                ssim_lst = ssim(rolling_wind_arr,rep_window_a)
                #dif_sum_min_idx = np.argmax(ssim_lst)
                shap = int(np.sqrt( rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(ssim_lst, (shap,shap))
                
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                row =  row -((shap-1)/2)
                col =  col - ((shap-1)/2)
                
                if mean_analysis:
                    if np.all(window_a==np.mean(window_a)): #and (not np.isnan(col) or not np.isnan(col)):
                #        print(k)
                #        print(m)
                        col = np.nan
                        row = np.nan
                
                if std_analysis:
                    if np.std(window_a)< std_threshold:
                        col = np.nan
                        row = np.nan
                
                
                u[k,m],v[k,m] = col, row
    
    return u, v*-1        
    



def remove_outliers(array, filter_size=5, sigma=1.5):
    returnarray = copy.copy(array)
    filter_diff = int(filter_size/2)
    #roll_arr = rolling_window(array, (window_size,window_size))
    bar = progressbar.ProgressBar(maxval=array.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
    bar.start()
    bar_iterator = 0
    for o in range(array.shape[0]):
        for p in range(array.shape[1]):
            act_px = array[o,p]
            try:
                act_arr = array[o-filter_diff:o+filter_diff+1,p-filter_diff:p+filter_diff+1]
                upperlim = np.nanmean(act_arr) + sigma*np.nanstd(act_arr)
                lowerlim = np.nanmean(act_arr) - sigma*np.nanstd(act_arr)
                
                if act_px< lowerlim or act_px> upperlim:
                    returnarray[o,p] = np.nanmean(act_arr)
            except:
                pass
        bar.update(bar_iterator+1)
        bar_iterator += 1
                
    bar.finish()
    return(returnarray)
                

    
    
    


 





### DEPRICATED USE image_to_val instead!

def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    roll_wind = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)
    roll_wind2 = np.reshape(roll_wind,(roll_wind.shape[0]*roll_wind.shape[1],roll_wind.shape[2],roll_wind.shape[3]))
    return roll_wind2
#
#iter_lst = []
#for i in range(-max_displacement,max_displacement):
#    # displace in x direction
#    for k in range(-max_displacement,max_displacement):
#        #displace in y direction
#        iter_lst.append((i,k))
#



def calc_IV(frame_a, frame_b, max_displacement=15, inter_window_size_y= 11, inter_window_size_x=11 , iter_lst = [], method = "greyscale"):
    # runs over all given bands
    """Compute Image Velocimetry based on each pixel in the image. 
       
       
        ----------
        Parameters
        ----------
        window_a : 2d np.ndarray
            a two dimensions array for the first interrogation window, 
        frame_a : 2d np.ndarray
            a two dimensions array for the first image 
        frame_b : 2d np.ndarray
            a two dimensions array for the second image 
        
        max_displacement : integer
            gives the allowance of displacement in pixels from the center of each interrogation window. Has high impact on computation time
            default = 25 
        inter_window_size_y/inter_window_size_x : integer
            gives the size of the interrogation window in x and y direction
            default = 35
        iter_lst : list
            depricated. This list was used to find the displacement in x and y direction
            
        method : defines the method after which the interrogation windows are compared to each other. 
            default = "greyscale"
        
        
        Returns
        -------
        stack : 3d np.ndarray
            3d stack first one u values for corresponding image, second one v values for corresponding image
        
        
    """ 
    
    
    
    inter_window_size_dif_x = int((inter_window_size_x-1)/2)    
    inter_window_size_dif_y = int((inter_window_size_y-1)/2) 
    
    
    
    print("Processing Image")
    print(n)
    print("----------------")
    u = np.zeros((frame_a.shape[0],frame_a.shape[1]))
    v = np.zeros((frame_a.shape[0], frame_a.shape[1]))
    
    iterx = max_displacement+inter_window_size_dif_x -1

    for l in range(max_displacement+inter_window_size_dif_y,(frame_a.shape[0])-(max_displacement+inter_window_size_dif_x)):
        # runs over all number of columns
        centery = l
        iterx+=1
        itery = max_displacement+inter_window_size_dif_x -1

        for m in range(max_displacement+inter_window_size_dif_x,(frame_a.shape[1])-(max_displacement+inter_window_size_dif_y)):
            # runs over all number of rows
            centerx = m
            itery+=1

            disp_arr = frame_b[centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
            rolling_wind_arr_reshaped = rolling_window(disp_arr, (inter_window_size_y, inter_window_size_x))
            
            #old - changed function
            #rolling_wind_arr_reshaped = np.reshape(rolling_wind_arr,(rolling_wind_arr.shape[0]*rolling_wind_arr.shape[1],rolling_wind_arr.shape[2],rolling_wind_arr.shape[3]))
            
            #inter_wind_disp_arr = frame_a[centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
            inter_wind1 = frame_a[centery-inter_window_size_dif_y:centery+inter_window_size_dif_y+1,centerx-inter_window_size_dif_x:centerx+inter_window_size_dif_x+1]
            rep_inter_wind1 = np.repeat(inter_wind1[ :, :,np.newaxis], rolling_wind_arr.shape[0]*rolling_wind_arr.shape[1], axis=2)
            rep_inter_wind1 = np.rollaxis(rep_inter_wind1,2)
            
            
            # displace interigation window with rolling window
            
           
            
            if method == "greyscale": 
            
                dif = rep_inter_wind1 - rolling_wind_arr_reshaped
                dif_sum = np.sum(abs(dif),(1,2))
                #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
                
                shap = int(np.sqrt( rep_inter_wind1.shape[0]))
                dif_sum_reshaped = np.reshape(dif_sum, (shap,shap))
                dif_sum_reshaped = (dif_sum_reshaped*-1)+np.max(dif_sum_reshaped)
                   
                
                
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                
                
                row = row-max_displacement 
                col = col-max_displacement 
                v[iterx,itery] = row
                u[iterx,itery] = col
                
                
                # old calculation
                #dif_sum_min_idx = np.argmin(dif_sum)
                #v[iterx,itery] = iter_lst[dif_sum_min_idx][0]
                #u[iterx,itery] = iter_lst[dif_sum_min_idx][1]
                
                
            
            if method == "rmse":         
                rmse = np.sqrt(np.mean((rolling_wind_arr_reshaped-rep_inter_wind1)**2,(1,2)))
                
                shap = int(np.sqrt( rep_inter_wind1.shape[0]))
                rmse_reshaped = np.reshape(rmse, (shap,shap))
                rmse_reshaped = (rmse_reshaped*-1)+np.max(rmse_reshaped)
                
                
                row, col = find_subpixel_peak_position(rmse_reshaped)
                
                row = row-max_displacement 
                col = col-max_displacement 
                
                v[iterx,itery] = row
                u[iterx,itery] = col
                
                
               

            if method == "ssim":
                
                
                ssim_lst = ssim(rolling_wind_arr_reshaped,rep_inter_wind1)
                #dif_sum_min_idx = np.argmax(ssim_lst)
                shap = int(np.sqrt( rep_inter_wind1.shape[0]))
                dif_sum_reshaped = np.reshape(ssim_lst, (shap,shap))
                plt.imshow(dif_sum_reshaped)
                row, col = find_subpixel_peak_position(dif_sum_reshaped)
                
                row = row-max_displacement 
                col = col-max_displacement 
                
                v[iterx,itery] = row
                u[iterx,itery] = col
                
                
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
    
    lw = 100*speed / speed.max()
    Q = plt.streamplot(X, Y, U, V, density=1, color='k', linewidth=lw)
    #E = plt.quiver(X1, Y1, np.mean(U)*20, np.mean(V)*20*-1, color = "red",width=0.01, headwidth=3, scale=1000)
    return(fig) 




#
#
def streamplot(U, V, X, Y, enhancement = 2, topo = None, vmin = -2, vmax = 2, cmap = "gist_rainbow", lw="ws", den = 1):
    
    
    if topo is None:
        topo = copy.copy(U)
    fig = plt.figure()
    I = plt.imshow(topo, cmap = cmap,vmin=vmin, vmax=vmax)
    fig.colorbar(I)

    if lw == "ws":
        speed = np.sqrt(U*U + V*V)   
        lw = enhancement*speed / np.nanmax(speed)
    
    Q = plt.streamplot(X, Y, U, V, density=den, color='k', linewidth=lw)
    #E = plt.quiver(X1, Y1, np.mean(U)*20, np.mean(V)*20*-1, color = "red",width=0.01, headwidth=3, scale=1000)
    return(fig) 
#
#
#
#
#
#def calc_TIV_test(n, pertub, iterx, itery, interval = 1, max_displacement=25, inter_window_size_y= 35, inter_window_size_x=35 ,inter_window_size_dif_x=10, inter_window_size_dif_y=10, iter_lst = [], method = "greyscale+"):
#    # runs over all given bands
#    n=0
#    print("Processing Image")
#    print(n)
#    print("----------------")
#    u = np.zeros((pertub.shape[1],pertub.shape[2]))
#    v = np.zeros((pertub.shape[1], pertub.shape[2]))
#    
#    iterx = max_displacement+inter_window_size_dif_x -1
#
#    for l in range(max_displacement+inter_window_size_dif_y,(pertub.shape[1])-(max_displacement+inter_window_size_dif_x)):
#        # runs over all number of columns
#        l = 14
#        print(l)
#        print("of "+str((pertub.shape[1])-(max_displacement+inter_window_size_dif_x)))
#      
#        centery = l
#        iterx+=1
#        itery = max_displacement+inter_window_size_dif_x -1
#        method = "direct"
#        
#        for m in range(max_displacement+inter_window_size_dif_x,(pertub.shape[2])-(max_displacement+inter_window_size_dif_y)):
#            # runs over all number of rows
#            #m = 43
#            print(m)
#            centerx = m
#            itery+=1
#
#            disp_arr = pertub[n+interval][centery-max_displacement-inter_window_size_dif_y:centery+max_displacement+inter_window_size_dif_y,centerx-max_displacement-inter_window_size_dif_x:centerx+max_displacement+inter_window_size_dif_x]
#            inter_wind1 = pertub[n][centery-inter_window_size_dif_y:centery+inter_window_size_dif_y+1,centerx-inter_window_size_dif_x:centerx+inter_window_size_dif_x+1]
#            #plt.imshow(disp_arr)
#            #plt.imshow(inter_wind1)
#            if method == "fft": 
#                corr = correlate_windows(inter_wind1, disp_arr,
#                                         corr_method=method, 
#                                         nfftx=nfftx, nffty=nffty)
#                
#            
#            if method == "direct": 
#                 corr = correlate_windows(inter_wind1, disp_arr,
#                                         corr_method=method, 
#                                         nfftx=None, nffty=None)
#            
#            #plt.imshow(corr)    
#            #plt.colorbar()
#            #plt.imshow(pertub[0])
#            
#            row, col = find_subpixel_peak_position(corr, subpixel_method="gaussian")
#            col = np.argmax(np.max(corr, axis=1))
#            row = np.argmax(np.max(corr, axis=0))        
#            search_area_size = disp_arr.shape[0]
#            window_size = inter_window_size_x
#            row -=  (search_area_size + window_size - 1)//2
#            col -=  (search_area_size + window_size - 1)//2
#
#            # get displacements, apply coordinate system definition
#            try:
#                v[iterx,itery] =  row
#                u[iterx,itery] = -col
#            except:
#                pass
#            #u[k,m],v[k,m] = -col, row
#
#      
#            
#    return(np.stack((u, v)))
#
#
#
#



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


