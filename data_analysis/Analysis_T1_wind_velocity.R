

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}


relERROR <- function(m, o, m1, o1){
  (abs((m - o)/o)*100 + 
     (abs(m1 - o1)/o1)*100)/2
}

Z0 = 0.001
H2 = 0.015



cc_IV_df = read.csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/cc_IV_df.csv", header = TRUE)
cc_IV_df_mean <- data.frame(cc_IV_df$window_size, cc_IV_df$overlap, cc_IV_df$search_area_size, cc_IV_df$u_mean,
                            cc_IV_df$v_mean,cc_IV_df$u_3x3_mean, cc_IV_df$v_3x3_mean, cc_IV_df$u_5x5_mean,cc_IV_df$v_5x5_mean, 
                            cc_IV_df$u_mean_irg1,cc_IV_df$v_mean_irg1,
                            cc_IV_df$u_mean_irg2,cc_IV_df$v_mean_irg2)

colnames(cc_IV_df_mean)<- colnames(cc_IV_df)[c(2:4,25:26,47:48,69:70,91:92,113:114)]

cc_IV_df_mean$log_u_mean_irg2 = cc_IV_df_mean$u_mean_irg2*(log(H2/0.01)/log(1.5/0.01))
cc_IV_df_mean$log_u_mean_irg1 =cc_IV_df_mean$u_mean_irg1*(log(H2/0.01)/log(0.5/0.01))
cc_IV_df_mean$log_v_mean_irg2 = cc_IV_df_mean$v_mean_irg2*(log(H2/0.01)/log(1.5/0.01))
cc_IV_df_mean$log_v_mean_irg1 =cc_IV_df_mean$v_mean_irg1*(log(H2/0.01)/log(0.5/0.01))


cc_IV_error <- data.frame(mean_uv_err = relERROR(cc_IV_df_mean$u_mean, cc_IV_df_mean$log_u_mean_irg2, cc_IV_df_mean$v_mean, cc_IV_df_mean$log_v_mean_irg2))
cc_IV_error$mean_uv_3x3_err <- relERROR(cc_IV_df_mean$u_3x3_mean, cc_IV_df_mean$log_u_mean_irg2, cc_IV_df_mean$v_3x3_mean, cc_IV_df_mean$log_v_mean_irg2)
cc_IV_error$mean_uv_5x5_err <- relERROR(cc_IV_df_mean$u_5x5_mean, cc_IV_df_mean$log_u_mean_irg2, cc_IV_df_mean$v_5x5_mean, cc_IV_df_mean$log_v_mean_irg2)






gs_IV_df = read.csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/greyscale_IV_df.csv", header = TRUE)
gs_IV_df_mean <- data.frame(gs_IV_df$window_size, gs_IV_df$overlap, gs_IV_df$search_area_size, gs_IV_df$u_mean,
                            gs_IV_df$v_mean,gs_IV_df$u_3x3_mean, gs_IV_df$v_3x3_mean, gs_IV_df$u_5x5_mean,gs_IV_df$v_5x5_mean, 
                            gs_IV_df$u_mean_irg1,gs_IV_df$v_mean_irg1,
                            gs_IV_df$u_mean_irg2,gs_IV_df$v_mean_irg2)

colnames(gs_IV_df_mean)<- colnames(gs_IV_df)[c(2:4,25:26,47:48,69:70,91:92,113:114)]
Z0 = 0.01
gs_IV_df_mean$log_u_mean_irg2 = gs_IV_df_mean$u_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
gs_IV_df_mean$log_u_mean_irg1 =gs_IV_df_mean$u_mean_irg1*(log(H2/Z0)/log(0.5/Z0))
gs_IV_df_mean$log_v_mean_irg2 = gs_IV_df_mean$v_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
gs_IV_df_mean$log_v_mean_irg1 =gs_IV_df_mean$v_mean_irg1*(log(H2/Z0)/log(0.5/Z0))

gs_IV_error <- data.frame(mean_uv_err = relERROR(gs_IV_df_mean$u_mean, gs_IV_df_mean$log_u_mean_irg2, gs_IV_df_mean$v_mean, gs_IV_df_mean$log_v_mean_irg2))
gs_IV_error$mean_uv_3x3_err <- relERROR(gs_IV_df_mean$u_3x3_mean, gs_IV_df_mean$log_u_mean_irg2, gs_IV_df_mean$v_3x3_mean, gs_IV_df_mean$log_v_mean_irg2)
gs_IV_error$mean_uv_5x5_err <- relERROR(gs_IV_df_mean$u_5x5_mean, gs_IV_df_mean$log_u_mean_irg2, gs_IV_df_mean$v_5x5_mean, gs_IV_df_mean$log_v_mean_irg2)




rmse_IV_df = read.csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/rmse_IV_df.csv", header = TRUE)

rmse_IV_df_mean <- data.frame(rmse_IV_df$window_size, gs_IV_df$overlap, rmse_IV_df$search_area_size, rmse_IV_df$u_mean,
                              rmse_IV_df$v_mean,rmse_IV_df$u_3x3_mean, rmse_IV_df$v_3x3_mean, rmse_IV_df$u_5x5_mean,rmse_IV_df$v_5x5_mean, 
                              rmse_IV_df$u_mean_irg1,rmse_IV_df$v_mean_irg1,
                              rmse_IV_df$u_mean_irg2,rmse_IV_df$v_mean_irg2)

colnames(rmse_IV_df_mean)<- colnames(gs_IV_df)[c(2:4,25:26,47:48,69:70,91:92,113:114)]

Z0 = 0.01
rmse_IV_df_mean$log_u_mean_irg2 = gs_IV_df_mean$u_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
rmse_IV_df_mean$log_u_mean_irg1 =gs_IV_df_mean$u_mean_irg1*(log(H2/Z0)/log(0.5/Z0))
rmse_IV_df_mean$log_v_mean_irg2 = gs_IV_df_mean$v_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
rmse_IV_df_mean$log_v_mean_irg1 =gs_IV_df_mean$v_mean_irg1*(log(H2/Z0)/log(0.5/Z0))

rmse_IV_error <- data.frame(mean_uv_err = relERROR(rmse_IV_df_mean$u_mean, rmse_IV_df_mean$log_u_mean_irg2, rmse_IV_df_mean$v_mean, rmse_IV_df_mean$log_v_mean_irg2))
rmse_IV_error$mean_uv_3x3_err <- relERROR(rmse_IV_df_mean$u_3x3_mean, rmse_IV_df_mean$log_u_mean_irg2, rmse_IV_df_mean$v_3x3_mean, rmse_IV_df_mean$log_v_mean_irg2)
rmse_IV_error$mean_uv_5x5_err <- relERROR(rmse_IV_df_mean$u_5x5_mean, rmse_IV_df_mean$log_u_mean_irg2, rmse_IV_df_mean$v_5x5_mean, rmse_IV_df_mean$log_v_mean_irg2)





ssim_IV_df = read.csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/ssim_IV_df.csv", header = TRUE)

ssim_IV_df_mean <- data.frame(ssim_IV_df$window_size, ssim_IV_df$overlap, ssim_IV_df$search_area_size, ssim_IV_df$u_mean,
                              ssim_IV_df$v_mean,ssim_IV_df$u_3x3_mean, ssim_IV_df$v_3x3_mean, ssim_IV_df$u_5x5_mean,ssim_IV_df$v_5x5_mean, 
                              ssim_IV_df$u_mean_irg1,ssim_IV_df$v_mean_irg1,
                              ssim_IV_df$u_mean_irg2,ssim_IV_df$v_mean_irg2)

colnames(ssim_IV_df_mean)<- colnames(ssim_IV_df)[c(2:4,25:26,47:48,69:70,91:92,113:114)]


ssim_IV_df_mean$log_u_mean_irg2 = gs_IV_df_mean$u_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
ssim_IV_df_mean$log_u_mean_irg1 =gs_IV_df_mean$u_mean_irg1*(log(H2/Z0)/log(0.5/Z0))
ssim_IV_df_mean$log_v_mean_irg2 = gs_IV_df_mean$v_mean_irg2*(log(H2/Z0)/log(1.5/Z0))
ssim_IV_df_mean$log_v_mean_irg1 =gs_IV_df_mean$v_mean_irg1*(log(H2/Z0)/log(0.5/Z0))

ssim_IV_error <- data.frame(mean_uv_err = relERROR(ssim_IV_df_mean$u_mean, ssim_IV_df_mean$log_u_mean_irg2, ssim_IV_df_mean$v_mean, ssim_IV_df_mean$log_v_mean_irg2))
ssim_IV_error$mean_uv_3x3_err <- relERROR(ssim_IV_df_mean$u_3x3_mean, ssim_IV_df_mean$log_u_mean_irg2, ssim_IV_df_mean$v_3x3_mean, ssim_IV_df_mean$log_v_mean_irg2)
ssim_IV_error$mean_uv_5x5_err <- relERROR(ssim_IV_df_mean$u_5x5_mean, ssim_IV_df_mean$log_u_mean_irg2, ssim_IV_df_mean$v_5x5_mean, ssim_IV_df_mean$log_v_mean_irg2)

ssim_IV_df <- ssim_IV_df[complete.cases(ssim_IV_df),]


ssim_IV_df$wd_irg2 <- uv2wd(ssim_IV_df$u_mean_irg1,ssim_IV_df$v_mean_irg1)

ssim_IV_df$wd_IV_mean <- uv2wd(ssim_IV_df$u_mean,ssim_IV_df$v_mean)
ssim_IV_df$wd_IV_3x3mean <- uv2wd(ssim_IV_df$u_3x3_mean,ssim_IV_df$v_3x3_mean)
ssim_IV_df$wd_IV_5x5mean <- uv2wd(ssim_IV_df$u_5x5_mean,ssim_IV_df$v_5x5_mean)



ssim_IV_df <- ssim_IV_df[ssim_IV_df$wd_IV_mean > ssim_IV_df$wd_irg2-15 | ssim_IV_df$wd_IV_mean < ssim_IV_df$wd_irg2+15 | 
           ssim_IV_df$wd_IV_3x3mean > ssim_IV_df$wd_irg2-15 | ssim_IV_df$wd_IV_3x3mean < ssim_IV_df$wd_irg2+15 |
             ssim_IV_df$wd_IV_5x5mean > ssim_IV_df$wd_irg2-15 | ssim_IV_df$wd_IV_5x5mean < ssim_IV_df$wd_irg2+15,]

ssim_IV_df$wd_IV_mean







uv2wd(ssim_IV_df_mean$log_u_mean_irg2[1],ssim_IV_df_mean$log_v_mean_irg2[1])
uv2ws(ssim_IV_df_mean$log_u_mean_irg2[1],ssim_IV_df_mean$log_v_mean_irg2[1])

uv2wd(ssim_IV_df_mean$u_mean,ssim_IV_df_mean$v_mean)
uv2wd(ssim_IV_df_mean$u_5x5_mean,ssim_IV_df_mean$v_5x5_mean)
uv2ws(ssim_IV_df_mean$u_5x5_mean,ssim_IV_df_mean$v_5x5_mean)


for (i in seq(1,17)){
i <- 15
  a <- c(ssim_IV_df$u_minmean_irg2_1[i], ssim_IV_df$u_minmean_irg2_2[i], ssim_IV_df$u_minmean_irg2_3[i], 
  ssim_IV_df$u_minmean_irg2_4[i],ssim_IV_df$u_minmean_irg2_5[i], ssim_IV_df$u_minmean_irg2_6[i], 
  ssim_IV_df$u_minmean_irg2_7[i], ssim_IV_df$u_minmean_irg2_8[i], ssim_IV_df$u_minmean_irg2_9[i],ssim_IV_df$u_minmean_irg2_10[i])
a = a*(log(H2/0.01)/log(0.5/0.01)) 

b <- c(ssim_IV_df$u_minmean1[i], ssim_IV_df$u_minmean2[i], ssim_IV_df$u_minmean3[i], 
  ssim_IV_df$u_minmean4[i],ssim_IV_df$u_minmean5[i], ssim_IV_df$u_minmean6[i], 
  ssim_IV_df$u_minmean7[i], ssim_IV_df$u_minmean8[i],ssim_IV_df$u_minmean9[i],ssim_IV_df$u_minmeani0[1])
print(a)
print(b)
print(RMSE(b, a))

}






