import matplotlib.pyplot as plt
import numpy as np
import h5py
import copy
from sklearn.model_selection import train_test_split
from TST_fun import create_tst_pertubations_mm, writeNetCDF
import progressbar
import os
import numpy as np
######################################################################################################################
### FILE 1: Write Data to RBG PNG Files
### read them and create a random forest model

T1 = True
start_img = 19500
end_img = 68000
#end_img = 68001


if T1:
    org_datapath = "file:///home/benjamin/Met_ParametersTST/T1/"
    img_datapath1 = org_datapath+"Flight03_O80_1616_pngs/"
    img_datapath2 = org_datapath+"Flight_03_O80_rect_stab/"
    datapath = "/media/benjamin/XChange/T1/data/Optris_data_120119/Tier01/Flight03_O80_1616/"
    add_prefix_pngfile = "Flight_03_O80_stab"
    image_type1 = ".tif"
    image_type2 = ".png"
    fls = os.listdir(datapath)
    fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
    
    
    
else:
    image_type1 = ".tif"
    image_type2 = ".png"
    org_datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    img_datapath = org_datapath+"figure_Tb_RGB_tif/"
    file = h5py.File(org_datapath+'Tb_1Hz.mat','r')
    tb= file.get("Tb")
    end_img = len(tb)
    
n_pics = 1

bar4 = progressbar.ProgressBar(maxval=end_img-start_img, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
bar4.start()
bar4_iterator = 0

MAE_lst = []
RMSE_lst = []
#end_img
for l in range(start_img,end_img,n_pics):
    N_len = l+n_pics
    
    if N_len > end_img:
        N_len = end_img
        
    print("Reading original Image...")
    
    
    if T1:
        for i in range(l,N_len):
            tb_mat = np.genfromtxt(datapath+fls[i-1], delimiter=',', skip_header=1)
            #tb_mat = np.rot90(tb_mat,3)
            #tb_mat = np.fliplr(tb_mat)  
            Tb_norm = copy.copy(tb_mat)
            if i == l:
                Tb_labels = tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1])
            else:
                Tb_labels = np.append(Tb_labels,tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1]))
        
    else:
        for i in range(l,N_len):
            tb_mat = tb[i]
            tb_mat = np.rot90(tb_mat,3)
            tb_mat = np.fliplr(tb_mat)  
            Tb_norm = copy.copy(tb_mat)
            if i == l:
                Tb_labels = tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1])
            else:
                Tb_labels = np.append(Tb_labels,tb_mat.reshape(Tb_norm.shape[0]*Tb_norm.shape[1]))    

    
    ### read RBG PNG Files
    print("Reading RGB values from original written PNG files ...")
    
    if T1:
        for i in range(l,N_len):
            image_file=img_datapath1+str(i)+image_type2
            arr=plt.imread(image_file)
            if i == l:
                arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
            else:
                arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)      
    else:
        for i in range(l,N_len):
            image_file=img_datapath+str(i)+image_type1
    
            arr=plt.imread(image_file)
            if i == l:
                arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
            else:
                arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)

    features = arr_features
    labels =  Tb_labels

    print("Create Random Forest Model ...")
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10, random_state = 42)
    
    # Import the model we are using
    from sklearn.ensemble import RandomForestRegressor
    
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, n_jobs = 40)
    
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
    
    for k in range(l,N_len):
        print(k)
        if not T1:    
            k = k+1
        print(k)
        print("start reading:" +str(k))
        
        if T1:
            image_file = img_datapath2+add_prefix_pngfile+str(k-18000)+image_type2
        else:
            image_file=datapath+'figure_Tb_RGB_tif_rect_stab_rectOCV/'+format(k, '04d')+'.png'       
            
        
        arr=plt.imread(image_file)
        arr = arr*255     
        
        if k-1 == l and not T1:
            arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))
        elif k == l:
            arr_features = arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2]))   
        else:
            arr_features = np.append(arr_features,arr.reshape((arr.shape[0]*arr.shape[1],arr.shape[2])),0)
    
    
    predictions_stab = rf.predict(arr_features)
    pred_stab_reshaped = np.reshape(predictions_stab,(n_pics,tb_mat.shape[0],tb_mat.shape[1]))
    if l <= start_img:
        final_pred_stab = copy.copy(pred_stab_reshaped)
    else:
        final_pred_stab = np.append(final_pred_stab,pred_stab_reshaped,0)

    bar4.update(bar4_iterator+1)
    bar4_iterator += 1
    
bar4.finish()
        
        
    

plt.imshow(final_pred_stab[5000])
plt.imshow(arr*255)
plt.colorbar()
#pred_stab_reshaped2 = np.rot90(pred_stab_reshaped,-2, axes=(0,1))
pred_stab_reshaped2= np.fliplr(final_pred_stab)
writeNetCDF(org_datapath,"Tb_1Hz_rect_stab.nc","Tb",pred_stab_reshaped2)

with open(org_datapath+'RF_RMSE_lst.txt', 'w') as f:
    for item in RMSE_lst:
        f.write("%s\n" % item)
with open(org_datapath+'RF_MAE_lst.txt', 'w') as f:
    for item in MAE_lst:
         f.write("%s\n" % np.mean(item))


Tb_stab_pertub_py = create_tst_pertubations_mm(pred_stab_reshaped2)
writeNetCDF(org_datapath,"Tb_stab_rect_pertub_py.nc","Tb_pertub",Tb_stab_pertub_py)

    





