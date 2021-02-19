
#install.packages("NISTunits", dependencies = TRUE)
library(NISTunits)


create_WS_WD <- function(netcdf_vas_kili,netcdf_uas_kili){
  
  netcdf_WD_kili <- stack()
  netcdf_WS_kili <- stack()
  raster_mask <- netcdf_uas_kili[[1]]
  vals <- 0
  values(raster_mask) <- vals
  act_ws_rst <- raster_mask
  act_wd_rst <- raster_mask
  for (j in seq(1,nlayers(netcdf_uas_kili))){
    print(paste("Layer", j))
    act_netcdf_uas_kili <-  netcdf_uas_kili[[j]]
    act_netcdf_vas_kili <-  netcdf_vas_kili[[j]]
    for (i in (seq(1,ncell(act_netcdf_uas_kili)))){
      #print(i)
      #print(paste("Cell",i))
      act_val_uas <- values(act_netcdf_uas_kili)[i]
      act_val_vas <- values(act_netcdf_vas_kili)[i]
      ws <- sqrt(act_val_uas**2+ act_val_vas**2)
      
      if ( act_val_uas > 0 & act_val_vas > 0){
        wd <- NISTradianTOdeg(atan(abs(act_val_uas/act_val_vas)))
      }
      if (act_val_uas > 0 & act_val_vas < 0){
        wd <- 180 - NISTradianTOdeg(atan(abs(act_val_uas/act_val_vas)))
      }
      if (act_val_uas < 0 & act_val_vas < 0){
        wd <- 180 + NISTradianTOdeg(atan(abs(act_val_uas/act_val_vas)))
      }
      if (act_val_uas < 0 & act_val_vas > 0){
        wd <- 360 - NISTradianTOdeg(atan(abs(act_val_uas/act_val_vas)))
      }
      if (wd>360){
        print("WHAT?!")
        break
      }
      values(act_ws_rst)[i] <- ws
      values(act_wd_rst)[i] <- wd
    }
    if (wd>360){
      print("WHAT?!")
      break
    }
    netcdf_WD_kili <- stack(netcdf_WD_kili, act_wd_rst)
    netcdf_WS_kili <- stack(netcdf_WS_kili,act_ws_rst)
  }
  wd_ws_lst <- list(netcdf_WD_kili, netcdf_WS_kili)
  names(wd_ws_lst) <- c("WD", "WS")
  return(wd_ws_lst)
}
