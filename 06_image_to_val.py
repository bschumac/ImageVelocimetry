import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from sklearn.model_selection import train_test_split
from TST_fun import create_tst_pertubations_mm, writeNetCDF
import progressbar
import os

from scipy import ndimage

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model 
from pandas import DataFrame  


import matplotlib.cm as cm
import scipy.cluster.vq as scv


    
######################################################################################################################
### FILE 1: Write Data to RBG PNG Files
### read them and create a random forest model

T1 = True
fire1 = False

if T1:
    start_img = 18500
    end_img = 68000
    
#start_img = 15300
#end_img = start_img+1000

org_start = 12300
#end_img = 68001



if T1:
    #org_datapath = "/home/benjamin/Met_ParametersTST/T1/"
    #img_datapath1 = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif/"
    #img_datapath1 = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif_viridis/"
    # first try:
    #img_datapath2 = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab/"
    
    # black edge not anti alias
    #img_datapath2="/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_NN_cut/"
    
    # virdris images:
    #img_datapath2="/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_virdris_20Hz/"
    
    
    
    #org_datapath = "/data/TST/data/"
    #img_datapath1 = "/data/TST/data/Tier01/Flight03_O80_1616_tif_viridis/"
    #img_datapath2="/data/TST/data/Tier02/Flight03_O80_1616_stab_tif_virdris_20Hz/"
    
    #datapath = "/data/TST/data/Tier01/Flight03_O80_1616/"
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616/"
    image_type1 = ".tif"
    image_type2 = ".tif"
    fls = os.listdir(datapath)
    fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])

elif fire1:
    org_datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/"
    #img_datapath1 = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif/"
    img_datapath1 = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier03/Optris_ascii/O80_220319_high_P1_RGB/"
    # first try:
    #img_datapath2 = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab/"
    
    # black edge not anti alias
    #img_datapath2="/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_NN_cut/"
    
    # virdris images:
    img_datapath2="/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier04/O80_220319_high_P1_RGB_stab/"
    
    
    datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier02/Optris_ascii/O80_220319_high_P1/"
    
    
      
else:
    image_type1 = ".tif"
    image_type2 = ".png"
    org_datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    img_datapath = org_datapath+"figure_Tb_RGB_tif/"
    file = h5py.File(org_datapath+'Tb_1Hz.mat','r')
    tb= file.get("Tb")
    end_img = len(tb)

    
 


#end_img

# original data
#org_data = np.genfromtxt(datapath+fls[i+18500], delimiter=',', skip_header=1)

def retrievevalues(datapath_csv_files, datapath_rbg_files, datapath_stab_rgb_files, end_img, start_img = 0, interval=1, image_type1 = ".tif", 
                   image_type2 = ".tif", method="ExtraRegressor"):

    fls = os.listdir(datapath_csv_files)
    fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
    
    
    
    fls2 = os.listdir(datapath_stab_rgb_files)
    fls2 = sorted(fls2, key = lambda x: x.rsplit('.', 1)[0])
        
  
    
    
    print("Reading original data to RAM...")
    counter = 0
    for i in range(start_img,end_img, interval): 
        
        if counter%100 == 0:
            print(str(counter)+" of "+str((end_img-start_img)/4))
        my_data = np.genfromtxt(datapath_csv_files+fls[i], delimiter=',', skip_header=1)
        my_data = np.reshape(my_data,(1,my_data.shape[0],my_data.shape[1]))
        if counter == 0:
            org_data = copy.copy(my_data)
        else:
            org_data = np.append(org_data,my_data,0)
        #org_data[counter] = my_data 
        counter+=1
    print("...finished!")
    
    
    print("Start modelling...")
    
    
    if method != "ExtraRegressor":
        from sklearn.ensemble import RandomForestRegressor
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        
    
    MAE_lst = []
    RMSE_lst = []
    for i in range(0,int(((end_img-start_img)/4)-1)):
        
        

        print(i)
        image_file=datapath_rbg_files+str(i)+image_type1
        org_data_rgb=plt.imread(image_file)
        org_data_rgb=org_data_rgb[:,:,0:3]
        #org_data_rgb=org_data_rgb[50:200,50:250,0:3]
        #plt.imshow(org_data_rgb)
        #plt.imshow(final_pred_stab[0],cmap=cm.jet)
        
        
        image_file=datapath_stab_rgb_files+format(i+1, '04d')+image_type2
        stab_data_rgb=plt.imread(image_file)
        stab_data_rgb=stab_data_rgb[:,:,0:3]
        #stab_data_rgb = stab_data_rgb*255
         
        org_data_subset = org_data[i]#[50:200,50:250]
        org_data_labels = org_data_subset.reshape(org_data_subset.shape[0]*org_data_subset.shape[1])
        org_data_rgb_features = org_data_rgb.reshape((org_data_rgb.shape[0]*org_data_rgb.shape[1],org_data_rgb.shape[2]))
        
        features = org_data_rgb_features
        labels =  org_data_labels
    
        print("Create Random Forest Model ...")
        
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.05, random_state = 42)
        
        # Import the model we are using
    
        # Instantiate model with 1000 decision trees
        rf = ExtraTreesRegressor(n_estimators = 500, random_state = 42, n_jobs = -1)
        
        # Train the model on training data
        rf.fit(train_features, train_labels);
        
        
        
        predictions = rf.predict(test_features)
        
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        rmse_error = np.sqrt(np.mean((predictions-test_labels)**2))
        MAE_lst.append(errors)
        RMSE_lst.append(rmse_error)
        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees. The RMSE is: '+ str(round(rmse_error,3)))
        print("Finished Creating Model")
        print("-----------------------------------------------------")
        
        
        print("Predicting whole stabilzed dataset")
    
        stab_data_rgb_features = stab_data_rgb.reshape((stab_data_rgb.shape[0]*stab_data_rgb.shape[1],stab_data_rgb.shape[2]))
        
        predictions_stab = rf.predict(stab_data_rgb_features)
        pred_stab_reshaped = np.reshape(predictions_stab,(1,stab_data_rgb.shape[0],stab_data_rgb.shape[1]))
        if i == 0:
            final_pred_stab = copy.copy(pred_stab_reshaped)
        else:
            final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)
    
    
    print("Modelling finished...!")
    return([final_pred_stab,MAE_lst,RMSE_lst])



final_pred_stab = retrievevalues(datapath_csv_files = datapath, datapath_rbg_files= img_datapath1, datapath_stab_rgb_files= img_datapath2, 
                                 start_img = start_img, end_img=end_img, interval=4, image_type1 = ".tif", image_type2 = ".tif")



writeNetCDF(org_datapath,"Tb_stab_20Hz.nc","Tb",final_pred_stab[0])
Tb_stab_pertub_py = create_tst_pertubations_mm(final_pred_stab, 400)
writeNetCDF(org_datapath,"Tb_stab_pertub_py_virdris_20Hz.nc","Tb_pertub",Tb_stab_pertub_py)

with open(org_datapath+'MAE_lst.txt', 'w') as f:
    for item in final_pred_stab[1]:
        f.write("%s\n" % item)


with open(org_datapath+'RMSE_lst.txt', 'w') as f:
    for item in final_pred_stab[2]:
        f.write("%s\n" % item)



























# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



################### OLD STUFF ###########################




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#
#
#
#
#

#
#
#
#
#
#
#
#
#
#counter = 0
#outpath = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif_BW/"
#for i in range(0,len(org_data), 1):      
#    my_data = org_data[i]
#    print("Writing File " +str(i)+".png from "+str(len(fls)))
#    fig = plt.figure(figsize=(322/my_dpi, 243/my_dpi), dpi=my_dpi)
#    ax1 = plt.subplot(111)
#    #u_xz_norm =  (u_xz - Tb_MIN) / (Tb_MAX-Tb_MIN)
#    im=ax1.imshow(my_data,interpolation=None,cmap=cm.Greys)
#    ax1.axis("off")
#    im.axes.get_xaxis().set_visible(False)
#    im.axes.get_yaxis().set_visible(False)
#    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#    plt.savefig(outpath+str(counter)+".tif",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#    plt.close()
#    counter +=1
#
#
#
#stab_data_bw_arr = copy.copy(final_pred_stab)
#img_datapath2="/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_BW/"
#
#
#for i in range(0,len(org_data)):
#    print(i)
#    image_file=img_datapath2+format(i, '04d')+image_type2
#    stab_data_bw=plt.imread(image_file)
#    stab_data_bw_arr[i] = stab_data_bw
#
#
#
#
#
## template matching
#     
#
#
#plt.imshow(pred_stab_reshaped)
#
#
#
#
#Tb_org_pertub = create_tst_pertubations_mm(final_pred_stab, 40)
#
#
#
#
#
#
#
#
#
#
#
#
#import cv2
#import numpy as np
#from matplotlib import pyplot as plt
#
#top_left_lst = [(1,1)]
#      
#for i in range(0,len(org_data)):
#    print(i)
#    org_rbg_path = img_datapath1
#    image_file=org_rbg_path+str(i)+image_type1
#    org_data_rgb=plt.imread(image_file)
#    org_data_rgb=org_data_rgb[:,:,0:3]
#    
#    stab_rbg_path = img_datapath2
#    image_file=stab_rbg_path+format(i, '04d')+image_type2
#    stab_data_rgb=plt.imread(image_file)
#    stab_data_rgb=stab_data_rgb[:,:,0:3]
#    
#   
#    
#    img = org_data_rgb
#    img2 = img.copy()
#    template = stab_data_rgb
#    w, h = template.shape[1], template.shape[0]
#    
#    # All the 6 methods for comparison in a list
#    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#    
#    #for meth in methods:
#    img = img2.copy()
#    meth = 'cv2.TM_SQDIFF'
#    method = eval(meth)
#    
#    # Apply template Matching
#    res = cv2.matchTemplate(img,template,method)
#    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#    
#    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#        top_left = min_loc
#    else:
#        top_left = max_loc
#    
#    if top_left[1] < top_left_lst[i-1][1]-10 or top_left[1] > top_left_lst[i-1][1]+10  and i > 1:
#        top_left = top_left_lst[i-1]
#    top_left_lst.append(top_left)
#        
#    
#    
#    #bottom_right = (top_left[0] + w, top_left[1] + h)
#    
#    #bottom_left = (top_left[0] + w, top_left[1] + h) 
#    
#    #top_right = (top_left[0],top_left[1] + h)
#    
#    
#    pred_stab = org_data[i][top_left[1]:top_left[1] + h,top_left[0]:top_left[0]+ w]
#    pred_stab_reshaped= np.reshape(pred_stab,(1,pred_stab.shape[0],pred_stab.shape[1]))
#    
#    if i == 0:
#        final_pred_stab = copy.copy(pred_stab_reshaped)
#    else:
#        final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)
#    
#
#
#np.save("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Tb_2Hz_org_data_TM.npy", final_pred_stab)
#
#
#final_pred_stab= np.load("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Tb_2Hz_org_data_TM.npy")
#
#
## second round of stabalizing:
#        
#        
#my_dpi = 300
#
#i = -1
#
#counter = 0
#outpath = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_NN_cut_TM/"
#for i in range(0,len(final_pred_stab), 1):      
#    my_data = final_pred_stab[i]
#    print("Writing File " +str(i)+".png from "+str(len(fls)))
#    fig = plt.figure(figsize=(322/my_dpi, 243/my_dpi), dpi=my_dpi)
#    ax1 = plt.subplot(111)
#    #u_xz_norm =  (u_xz - Tb_MIN) / (Tb_MAX-Tb_MIN)
#    im=ax1.imshow(my_data,interpolation=None,cmap=cm.Greys)
#    ax1.axis("off")
#    im.axes.get_xaxis().set_visible(False)
#    im.axes.get_yaxis().set_visible(False)
#    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
#    plt.savefig(outpath+str(counter)+".tif",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#    plt.close()
#    counter +=1
#
# 
#
#org_data2 = copy.copy(final_pred_stab)
#org_data3 = copy.copy(final_pred_stab)
#
##final_pred_stab = copy.copy(org_data)
#
#
#
#
#
#
## vectorisation
#
#
#def colormap2arr(arr,cmap):    
#    # http://stackoverflow.com/questions/3720840/how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
#   
#    gradient=cmap(np.linspace(0.0,1.0,42))
#    gradient= gradient[:,0:3]
#    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
#    arr2=arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
#    arr2 = arr2[:,0:3]
#    # Use vector quantization to shift the values in arr2 to the nearest point in
#    # the code book (gradient).
#    code,dist=scv.vq(arr2,gradient)
#
#    # code is an array of length arr2 (240*240), holding the code book index for
#    # each observation. (arr2 are the "observations".)
#    # Scale the values so they are from 0 to 1.
#    values=code.astype('float')/gradient.shape[0]
#
#    # Reshape values back to (240,240)
#    values=values.reshape(1,arr.shape[0],arr.shape[1])
#    values=values[::-1]
#    return values
#
#
#
#for i in range(0,len(org_data)):
#    print(i)
#    image_file=img_datapath1+str(i)+image_type1
#    org_data_rgb=plt.imread(image_file)
#    org_data_rgb=org_data_rgb[:,:,0:3]
#    #org_data_rgb=org_data_rgb[100:250,100:350,0:3]
#
#    
#    image_file=img_datapath2+format(i, '04d')+image_type2
#    stab_data_rgb=plt.imread(image_file)
#    stab_data_rgb=stab_data_rgb[:,:,0:3]
#
#
#    arr = stab_data_rgb
#    pred_stab_reshaped = colormap2arr(arr,cm.jet)
#    pred_stab_reshaped=pred_stab_reshaped*(org_data[i].max()-org_data[i].min())+org_data[i].min()
#    if i == 0:
#        final_pred_stab = copy.copy(pred_stab_reshaped)
#    else:
#        final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)         
#
#
#
#
#
#
#
#
#
#
#
#for l in range(start_img,end_img,n_pics): 
#    N_len = l+1
#    #n_pics
#    
#    if N_len > end_img:
#        N_len = end_img
#        my_data
#    print("Reading original Image...")
#    
#    
#    if T1:
#        
#        for i in range(l,N_len):
#            
#            tb_mat = np.genfromtxt(datapath+fls[i-1], delimiter=',', skip_header=1)
#            #tb_mat = np.rot90(tb_mat,3)
#            #tb_mat = np.fliplr(tb_mat)  
#            Tb_norm = copy.copy(tb_mat)
#            Tb_norm = Tb_norm[50:250,50:350]
#            if i == l:
#                Tb_labels = Tb_norm.reshape(Tb_norm.shape[0]*Tb_norm.shape[1])
#            else:
#                Tb_labels = np.append(Tb_labels,Tb_norm.reshape(Tb_norm.shape[0]*Tb_norm.shape[1]))
#        
#    else:
#        for i in range(l,N_len):
#            tb_mat = tb[i]
#            tb_mat = np.rot90(tb_mat,3)
#            tb_mat = np.fliplr(tb_mat)  
#            Tb_norm = copy.copy(tb_mat)
#            if i == l:
#                Tb_labels = tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1])
#            else:
#                Tb_labels = np.append(Tb_labels,tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1]))    
#
#    
#    ### read RBG PNG Files
#    print("Reading RGB values from original written PNG files ...")
#    
#    if T1:
#        for i in range(l,N_len):
#            image_file=img_datapath1+str(i-start_img)+image_type2
#            arr=plt.imread(image_file)
#            arr = arr[50:250,50:350,1:4]
#            if i == l:
#                arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
#            else:
#                arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)      
#    #plt.imshow(arr)
#    #plt.imshow(tb_mat) 
#    else:
#        for i in range(l,N_len):
#            image_file=img_datapath+str(i)+image_type1
#    
#            arr=plt.imread(image_file)
#            arr = arr[:,:,1:4]
#            if i == l:
#                arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
#            else:
#                arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)
#
#    features = arr_features
#    labels =  Tb_labels
#
#    print("Create Random Forest Model ...")
#    
#    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10, random_state = 42)
#    
#    # Import the model we are using
#    from sklearn.ensemble import RandomForestRegressor
#    
#    # Instantiate model with 1000 decisiondimage.median_filter(final_pred_stab[i], size=3 )n trees
#    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs = -1)
#    
#    # Train the model on training data
#    rf.fit(train_features, train_labels);
#    
#    
#    
#    predictions = rf.predict(test_features)
#    
#    # Calculate the absolute errors
#    errors = abs(predictions - test_labels)
#    rmse_error = np.sqrt(np.mean((predictions-test_labels)**2))
#    MAE_lst.append(errors)
#    RMSE_lst.append(rmse_error)
#    # Print out the mean absolute error (mae)
#    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees. The RMSE is: '+ str(round(rmse_error,3)))
#    print("Finished Creating Model")
#    print("-----------------------------------------------------")
#     
#    print("Predicting whole stabilzed dataset")
#    
#    
#    
#    for k in range(l,N_len):
#        k = l
#        print(k)
#        if not T1:    
#            k = k+1
#        print(k)
#        print("start reading:" +str(k))
#        
#        if T1:
#            image_file = img_datapath2+format(k-start_img, '04d')+image_type2
#        else:
#            image_file=datapath+'figure_Tb_RGB_tif_rect_stab_rectOCV/'+format(k, '04d')+'.png'       
#        
#        
#        arr=plt.imread(image_file)
#       
#        if not T1:
#            arr = arr*255 
#                
#        
#        if k-1 == l and not T1:
#            arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
#        elif k == l:
#            arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))   
#        else:
#            arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)
#    
#    
#    predictions_stab = rf.predict(arr_features)
#    
#    pred_stab_reshaped = np.reshape(predictions_stab,(1,tb_mat.shape[0],tb_mat.shape[1]))
#
#    if l <= start_img:
#        final_pred_stab = copy.copy(pred_stab_reshaped)
#    else:
#        final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)
#
#    bar4.update(bar4_iterator+1)
#    bar4_iterator += 1
#    ndimage.median_filter(final_pred_stab[i], size=3 )
#bar4.finish()
#        
#arr[95,200,:]     
#pred_stab_reshaped[95,200]  
#
##plt.imshow(final_pred_stab[5000])
#plt.imshow(final_pred_stab[1])
#plt.colorbar()
#plt.imshow(arr)
#plt.colorbar()
#
##pred_stab_reshaped2 = np.rot90(pred_stab_reshaped,-2, axes=(0,1))
#org_datapath = "/home/benjamin/Met_ParametersTST/T1/"
#pertub_1s= np.fliplr(pertub_1s)
#
#plt.imshow(final_pred_stab[1])
#plt.imshow()
#
#pertub_1s = np.zeros((326,final_pred_stab.shape[1],final_pred_stab.shape[2]))
#counter = 0
#final_pred_stab_medfilter = copy.copy(final_pred_stab)
#for i in range(0,len(final_pred_stab)):
#    print(i)
#    final_pred_stab_medfilter[i] = ndimage.median_filter(final_pred_stab[i], size=3 )
#print(counter)   
#    
#Tb_1s= copy.copy(pertub_1s)
#
#pertub_1s= create_tst_pertubations_mm(final_pred_stab,60)
#
#pertub_1s= np.fliplr(pertub_1s)
#writeNetCDF(org_datapath,"Tb_1Hz_rect_stab.nc","Tb",final_pred_stab)
#
#with open(org_datapath+'RF_RMSE_lst.txt', 'w') as f:
#    for item in RMSE_lst:
#        f.write("%s\n" % item)
#with open(org_datapath+'RF_MAE_lst.txt', 'w') as f:
#    for item in MAE_lst:
#         f.write("%s\n" % np.mean(item))
#
#
#Tb_stab_pertub_py = create_tst_pertubations_mm(pred_stab_reshaped2)
#writeNetCDF(org_datapath,"Tb_stab_rect_pertub_py.nc","Tb_pertub",Tb_stab_pertub_py)
#
#    
#
#
## linear model 
#
#for i in range(0,len(org_data)):
#    print(i)
#    image_file=img_datapath1+str(i)+image_type1
#    org_data_rgb=plt.imread(image_file)
#    org_data_rgb=org_data_rgb[:,:,0:3]
#    #org_data_rgb=org_data_rgb[100:250,100:350,0:3]
#
#    
#    image_file=img_datapath2+format(i, '04d')+image_type2
#    stab_data_rgb=plt.imread(image_file)
#    stab_data_rgb=stab_data_rgb[:,:,0:3]
#    
#    Stock_Market ={'B': org_data_rgb[:,:,0].flatten().tolist(),
#                    'G': org_data_rgb[:,:,1].flatten().tolist(),
#                    'R': org_data_rgb[:,:,2].flatten().tolist(),
#                    'Val':  org_data[i].flatten().tolist()}      
#    
#    df = DataFrame(Stock_Market,columns=['B','G','R','Val'])
#    
#    
#    X = df[['B','G', 'R']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
#    Y = df['Val']
#    
#    
#    
#    predictors = {'B': stab_data_rgb[:,:,0].flatten().tolist(),
#                    'G': stab_data_rgb[:,:,1].flatten().tolist(),
#                    'R': stab_data_rgb[:,:,2].flatten().tolist()}
#    
#    dfpred = DataFrame(predictors,columns=['B','G','R'])
#    pred = dfpred[['B','G', 'R']]
#    len(org_data_rgb[2].flatten().tolist()) 
#    # with sklearn
#    clf = linear_model.LinearRegression()
#    clf.fit(X, Y)  
#    clf.coef_      
#    pred_stab = clf.predict(pred)        
#    pred_stab_reshaped = np.reshape(pred_stab,(1,stab_data_rgb.shape[0],stab_data_rgb.shape[1]))
#    if i == 0:
#        final_pred_stab = copy.copy(pred_stab_reshaped)
#    else:
#        final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)       
#    
#
#
#
#
## search algorithm parallized
#        
#        
#from joblib import Parallel, delayed        
#num_cores = 8
#
#out_arr = np.zeros((stab_data_rgb.shape[0],stab_data_rgb.shape[1]))
#
#def rgb2values(n, org_data, org_rbg_path, stab_rbg_path, search_area_size = 80, out_arr = out_array):
#    n = 0
#    print(n)
#    org_rbg_path = img_datapath1
#    image_file=org_rbg_path+str(n)+image_type1
#    org_data_rgb=plt.imread(image_file)
#    org_data_rgb=org_data_rgb[:,:,0:3]
#    
#    stab_rbg_path = img_datapath2
#    image_file=stab_rbg_path+format(n, '04d')+image_type2
#    stab_data_rgb=plt.imread(image_file)
#    stab_data_rgb=stab_data_rgb[:,:,0:3]
#    
#    act_org_data = org_data[n] 
#    
#    for l in range(0, stab_data_rgb.shape[0]):
#        print(l)
#        for m in range(0,stab_data_rgb.shape[1]):
#            act_stab_pixel = stab_data_rgb[l,m,:]
#            
#            if l < search_area_size//2:
#                search_area_min_x = 0
#                search_area_max_x = search_area_size
#            elif l+search_area_size/2 >stab_data_rgb.shape[0]:
#                search_area_min_x = stab_data_rgb.shape[0]-search_area_size
#                search_area_max_x = stab_data_rgb.shape[0]
#                
#            else:
#                search_area_min_x = l-search_area_size//2
#                search_area_max_x = l+search_area_size//2
#                
#            if m < search_area_size//2:
#                search_area_min_y = 0
#                search_area_max_y = search_area_size
#            elprint(n)
#
#
#                search_area_min_y = stab_data_rgb.shape[1]-search_area_size
#                search_area_max_y= stab_data_rgb.shape[1]  
#            else:
#                search_area_min_y = m-search_area_size//2
#                search_area_max_y = m+search_area_size//2
#            
#            act_org_rbg_area= org_data_rgb[search_area_min_x:search_area_max_x,search_area_min_y:search_area_max_y,:]
#            act_org_data_area = act_org_data[search_area_min_x:search_area_max_x,search_area_min_y:search_area_max_y]
#            stab_data_rgb_area= stab_data_rgb[search_area_min_x:search_area_max_x,search_area_min_y:search_area_max_y,:]
#            
#            
#            
#            dif_rbg = act_org_rbg_area-act_stab_pixel
#            dif_rbg_sum = np.sum(dif_rbg, axis=2)
#            
#            out_arr[l,m] = act_org_data_area[np.where(dif_rbg_sum == dif_rbg_sum.min())[0][0],np.where(dif_rbg_sum == dif_rbg_sum.min())[1][0]]
#            
#            
#            
#            
#            
#            plt.imshow(out_arr)
#            plt.colorbar()
#            
#            
#            plt.imshow(dif_rbg_sum)
#            plt.colorbar()
#            plt.imshow(act_org_rbg_area)
#            plt.colorbar()
#            plt.imshow(stab_data_rgb)
#            plt.colorbar()
#            
#            
#            
#            plt.imshow(np.all(act_org_rbg_area == act_stab_pixel, axis=-1))
#            
#
#            plt.imshow(dif_rbg)
#            plt.colorbar()
#            
#            
#            
#            
#            
#            c = org_data_rgb-stab_data_rgb
#            c.shape
#            d = np.sum(c,axis=2)
#            plt.imshow(a)
#            plt.colorbar()
#            
#            
#            
#            np.argmin(dif_rbg_sum)
#            np.argmin(dif_rbg_sum, axis=0)
#            plt.imshow(act_org_rbg_area[dif_rbg_sum == 0.0039215684])
#            
#            
#            dif_rbg_sum[2,39]
#            dif_rbg_sum[34,30]
#            np.where(RSS == RSS_min)
#            a= np.array(([255,0,125],[220,5,150]))
#            b = np.array=(([255,0,125]))
#            a-b
#            minpos = a[np.argmin(a)]
#            a[minpos[1],minpos[0]]
#            org_data_rgb[]
#            
#            
#            
#            
#            value_org_pixel = act_org_data[l,m]
#            
#    
#    
# 
#Parallel(n_jobs=num_cores)(delayed(rgb2values)(n, org_data=org_data, ))
#
# 
    
