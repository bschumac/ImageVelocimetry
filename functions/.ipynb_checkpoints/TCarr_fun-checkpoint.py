import numpy as np
from math import acos, degrees, log
import pandas as pd
def TCcalcdistangle(TC, base_TC, x, y, prev_angle=0, pix_resolution = 0.2):    
#base_TC = 2
#TC = 5

    WD = 0
    distance = np.sqrt(np.square(x[base_TC-1]-x[TC-1])+ np.square(y[base_TC-1]-y[TC-1]))
    if (y[base_TC-1]-y[TC-1]) == 0:
        if (x[base_TC-1]-x[TC-1])<0:
            WD = 270
        elif (x[base_TC-1]-x[TC-1])>0:
            WD = 90
    elif (x[base_TC-1]-x[TC-1]) == 0:
        if (y[base_TC-1]-y[TC-1])<0:
            WD = 0
        elif (y[base_TC-1]-y[TC-1])>0:
            WD = 180
    else:
        A = x[base_TC-1]-x[TC-1]
        B = y[base_TC-1]-y[TC-1]
        C = np.sqrt(np.square(x[base_TC-1]-x[TC-1])+ np.square(y[base_TC-1]-y[TC-1]))
        #print(A)
        if A < 0:
            WD = 180 + degrees(acos((B * B + C * C - A * A)/(2.0 * B * C)))
        if A > 0:
            WD = 180 - degrees(acos((B * B + C * C - A * A)/(2.0 * B * C)))

    #print(WD) 
        #print(B)
    return(distance*pix_resolution, WD)







def correlateTC(base_TC, TCs_signal, min_corr_time, max_corr_time, TCx, TCy, TC_nans = [1,12] , numberof_TCs = 12, subsamp_feq = 10, signal_len_s = 1 ):
    """
    Calculates wind speed, wind direction from TC array based on signal correlation 
    
    Parameters
    ----------
    base_TC: int
        The base TC which the signal correlation is calculated on, refers to the position TCs_signal list. base_tc given the TCs number-1 is the place in the TCs_signal list
    TCs_signal: list 
        List of all signals available in the TC array  
    min_corr_time: int
        The minimal travel time in between the TCs given in the units of the TCs_signal. If your subsamp_feq frequency is 10Hz then 10 will allow 1 s minimal travel time.
    max_corr_time: int
        The maximal travel time in between the TCs given in the units of the TCs_signal.
    TC_nans : lst
        A list of TCs which should be excluded
    numberof_TCs : int
        The number of all available TCs
    subsamp_feq : int
        the subsampling frequency of the TCs
    signal_len_s : int
        the signal length in seconds to find the correlation based on. signal_len_s * subsamp_feq = the number of measurements which are correlated         
    Returns
    -------
    recommended_interval : float
        The found most powerful interval in float. Needs rounding to the next int.
    
    """

    #base_TC=4
    #TCs_signal=TCs_signal
    #min_corr_time=3
    #max_corr_time=20
    #TC_nans = [1,12] 
    #numberof_TCs = 12

    
    signal_len_multiplier = int(max_corr_time/10)
    
    WS_lst = []
    WD_lst = []
    picked_tc_lst = []
    corr_time_lst = []
    base_TC_signal=TCs_signal[base_TC-1]
    
    full_corr_lst_TC1 = []
    full_corr_lst_TC2 = []
    full_corr_lst_TC3 = []
    full_corr_lst_TC4 = []
    full_corr_lst_TC5 = []

    full_corr_lst_TC6 = []
    full_corr_lst_TC7 = []
    full_corr_lst_TC8 = []
    full_corr_lst_TC9 = []
    full_corr_lst_TC10 = []
    full_corr_lst_TC11 = []
    full_corr_lst_TC12 = []
    full_corr_max_lst = []
    
    # 0 -> len of signal - 40, step = 20
    for j in range(0,len(base_TC_signal)-2*signal_len_multiplier*(subsamp_feq*signal_len_s),(subsamp_feq*signal_len_s)):
        # defines the beginnings of the signals correlations
        
        corr_lst_TC1 = []
        corr_lst_TC2 = []
        corr_lst_TC3 = []
        corr_lst_TC4 = []
        corr_lst_TC5 = []
        corr_lst_TC6 = []
        corr_lst_TC7 = []
        corr_lst_TC8 = []
        corr_lst_TC9 = []
        corr_lst_TC10 = []
        corr_lst_TC11 = []
        corr_lst_TC12 = []


        for i in range(0,(subsamp_feq*signal_len_s)*signal_len_multiplier):
            
            corr_TC1 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[0][j:j+(subsamp_feq*signal_len_s)])
            corr_TC2 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[1][j:j+(subsamp_feq*signal_len_s)])
            corr_TC3 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[2][j:j+(subsamp_feq*signal_len_s)])
            corr_TC4 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[3][j:j+(subsamp_feq*signal_len_s)])
            corr_TC5 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[4][j:j+(subsamp_feq*signal_len_s)])
            corr_TC6 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[5][j:j+(subsamp_feq*signal_len_s)])
            corr_TC7 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[6][j:j+(subsamp_feq*signal_len_s)])
            corr_TC8 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[7][j:j+(subsamp_feq*signal_len_s)])
            corr_TC9 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[8][j:j+(subsamp_feq*signal_len_s)])
            corr_TC10 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[9][j:j+(subsamp_feq*signal_len_s)])
            corr_TC11 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[10][j:j+(subsamp_feq*signal_len_s)])
            corr_TC12 = np.corrcoef(base_TC_signal[j+i:j+(subsamp_feq*signal_len_s)+i],TCs_signal[11][j:j+(subsamp_feq*signal_len_s)])

            corr_lst_TC1.append(corr_TC1[0,1])
            corr_lst_TC2.append(corr_TC2[0,1])
            corr_lst_TC3.append(corr_TC3[0,1])
            corr_lst_TC4.append(corr_TC4[0,1])
            corr_lst_TC5.append(corr_TC5[0,1])
            corr_lst_TC6.append(corr_TC6[0,1])
            corr_lst_TC7.append(corr_TC7[0,1])
            corr_lst_TC8.append(corr_TC8[0,1])
            corr_lst_TC9.append(corr_TC9[0,1])
            corr_lst_TC10.append(corr_TC10[0,1])
            corr_lst_TC11.append(corr_TC11[0,1])
            corr_lst_TC12.append(corr_TC12[0,1])

            full_corr_lst_TC1.append(corr_TC1[0,1])
            full_corr_lst_TC2.append(corr_TC2[0,1])
            full_corr_lst_TC3.append(corr_TC3[0,1])
            full_corr_lst_TC4.append(corr_TC4[0,1])
            full_corr_lst_TC5.append(corr_TC5[0,1])
            full_corr_lst_TC6.append(corr_TC6[0,1])
            full_corr_lst_TC7.append(corr_TC7[0,1])
            full_corr_lst_TC8.append(corr_TC8[0,1])
            full_corr_lst_TC9.append(corr_TC9[0,1])
            full_corr_lst_TC10.append(corr_TC10[0,1])
            full_corr_lst_TC11.append(corr_TC11[0,1])
            full_corr_lst_TC12.append(corr_TC12[0,1])


        corr_max_lst = []
        
        # append either nan or max(corr_lst_TCX)
        for l in range(1,numberof_TCs):
            if l in TC_nans or l == base_TC:
                corr_max_lst.append(np.nan)
            else:
                if l == 1:
                    corr_max_lst.append(np.max(corr_lst_TC1[min_corr_time:max_corr_time]))
                elif l == 2:

                    corr_max_lst.append(np.max(corr_lst_TC2[min_corr_time:max_corr_time]))   
                elif l == 3:

                    corr_max_lst.append(np.max(corr_lst_TC3[min_corr_time:max_corr_time]))
                elif l == 4:

                    corr_max_lst.append(np.max(corr_lst_TC4[min_corr_time:max_corr_time]))
                elif l == 5:

                    corr_max_lst.append(np.max(corr_lst_TC5[min_corr_time:max_corr_time]))
                elif l == 6:

                    corr_max_lst.append(np.max(corr_lst_TC6[min_corr_time:max_corr_time]))
                elif l == 7:

                    corr_max_lst.append(np.max(corr_lst_TC7[min_corr_time:max_corr_time]))
                elif l == 8:

                    corr_max_lst.append(np.max(corr_lst_TC8[min_corr_time:max_corr_time]))
                elif l == 9:

                    corr_max_lst.append(np.max(corr_lst_TC9[min_corr_time:max_corr_time]))
                elif l == 10:

                    corr_max_lst.append(np.max(corr_lst_TC10[min_corr_time:max_corr_time]))
                elif l == 11:

                    corr_max_lst.append(np.max(corr_lst_TC11[min_corr_time:max_corr_time]))
                elif l == 12:

                    corr_max_lst.append(np.max(corr_lst_TC12[min_corr_time:max_corr_time]))

        picked_TC = np.nanargmax(corr_max_lst)+1
        max_corrval= np.nanmax(corr_max_lst)
        full_corr_max_lst.append(max_corrval)
        if picked_TC == 12 :
            corr_time = corr_lst_TC12.index(max_corrval)
            distance, WD = TCcalcdistangle(12,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        if picked_TC == 11:
            corr_time = corr_lst_TC11.index(max_corrval)
            distance, WD = TCcalcdistangle(11,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 10:
            corr_time = corr_lst_TC10.index(max_corrval)
            distance, WD = TCcalcdistangle(10,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 9:
            corr_time = corr_lst_TC9.index(max_corrval)
            distance, WD = TCcalcdistangle(9,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 8:
            corr_time = corr_lst_TC8.index(max_corrval)
            distance, WD = TCcalcdistangle(8,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 7:
            corr_time = corr_lst_TC7.index(max_corrval)
            distance, WD = TCcalcdistangle(7,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)    
        elif picked_TC == 6:
            corr_time = corr_lst_TC6.index(max_corrval)
            distance, WD = TCcalcdistangle(6,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 5:
            corr_time = corr_lst_TC5.index(max_corrval)
            distance, WD = TCcalcdistangle(5,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 4:
            corr_time = corr_lst_TC4.index(max_corrval)
            distance, WD = TCcalcdistangle(4,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 3:
            corr_time = corr_lst_TC3.index(max_corrval)
            distance, WD = TCcalcdistangle(3,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq) 
        elif picked_TC == 2:
            corr_time = corr_lst_TC2.index(max_corrval)
            distance, WD = TCcalcdistangle(2,base_TC,x=TCx,y=TCy)
            WS = distance/(corr_time/subsamp_feq)
        elif picked_TC == 1:
            corr_time = corr_lst_TC1.index(max_corrval)
            distance, WD = TCcalcdistangle(1,base_TC,x=TCx,y=TCy)
            #print(distance)
            WS = distance/(corr_time/subsamp_feq) 
            
        WS_lst.append(WS)
        WD_lst.append(WD)
        picked_tc_lst.append(picked_TC)
        corr_time_lst.append(corr_time)
    TC_corr_df = pd.DataFrame(list(zip(full_corr_lst_TC1, full_corr_lst_TC2,full_corr_lst_TC3, full_corr_lst_TC4, full_corr_lst_TC5,
                      full_corr_lst_TC6,full_corr_lst_TC7,full_corr_lst_TC8,full_corr_lst_TC9,full_corr_lst_TC10,
                      full_corr_lst_TC11, full_corr_lst_TC12)), 
           columns =['TC1_corr', 'TC2_corr','TC3_corr','TC4_corr','TC5_corr','TC6_corr','TC7_corr','TC8_corr',
                     'TC9_corr','TC10_corr','TC11_corr','TC12_corr']) 

    return([WS_lst,WD_lst,picked_tc_lst,corr_time_lst,full_corr_max_lst,TC_corr_df])
