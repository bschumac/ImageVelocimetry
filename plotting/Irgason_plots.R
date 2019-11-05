library(zoo)
require(dplyr)
datapath <-"/home/benjamin/Met_ParametersTST/T0/data/Irgason_data/Exp01/CConvert/" 
fls <- list.files("/home/benjamin/Met_ParametersTST/T0/data/Irgason_data/Exp01/CConvert/")

degrees <- function(radians) 180 * radians / pi

calc_wd <- function(v, u){
  mathdegs <- degrees(atan2(v, u))
  wd <- ifelse (mathdegs>0, mathdegs, mathdegs+360)
  return(round(wd,2))
}

calc_ws <- function (v,u){
  ws <- sqrt((u*u)+(v*v))
  return(round(ws,2))
}



T0irg_df <- data.frame()
for (i in seq(1,length(fls))){
  print(i)
  T0irg_df_act <- read.csv(paste0(datapath,fls[i]), header = TRUE, skip = 1)
  T0irg_header <- T0irg_df_act[1:2,]
  T0irg_df_act <- T0irg_df_act[3:length(T0irg_df_act[,1]),]
  T0irg_df <- rbind(T0irg_df,T0irg_df_act)
}



T0irg_df$TIMESTAMP <- strptime(T0irg_df$TIMESTAMP, format="%Y-%m-%d %H:%M:%OS")
T0irg_df<- T0irg_df[order(T0irg_df$TIMESTAMP),]
T0irg_df$Ux <- as.numeric(as.character(T0irg_df$Ux))
T0irg_df$Uy <- as.numeric(as.character(T0irg_df$Uy))



#means <- aggregate(T0irg_df$Ux, by = list(T0irg_df$TIMESTAMP), FUN=mean)

T0_irg_means <- aggregate(T0irg_df["Ux"], 
                   list(TIMESTAMP=cut(T0irg_df$TIMESTAMP, "1 mins")),
                   mean)

Uy_means<- aggregate(T0irg_df["Uy"], 
          list(TIMESTAMP=cut(T0irg_df$TIMESTAMP, "1 mins")),
          mean)

T0_irg_means$Uy <- Uy_means$Uy





T0_irg_means$wd_uv <- calc_wd(T0_irg_means$Uy,T0_irg_means$Ux)
T0_irg_means$ws_uv <- calc_ws(T0_irg_means$Uy,T0_irg_means$Ux)

T0irg_df <- T0_irg_means

mean(T0irg_df$ws_uv)

head(T0irg_df)
tail(T0irg_df)
############# LINEPLOTS
par(mfrow = c(1,1))
plot(T0irg_df$TIMESTAMP,T0irg_df$Uy, type="l")
lines(T0irg_df$TIMESTAMP,T0irg_df$Ux, col="red")


plot(T0irg_df$TIMESTAMP,T0irg_df$wd_uv, type="l")


############# HISTOGRAM



par(mfrow=c(1,1))
test_rollmean <-  rollmean(T0irg_df$wd_uv, k=20)
hist(T0irg_df$wd_uv)

# ws
hist(T0irg_df$ws_uv)
mean(T0irg_df$ws_uv)

############## BOXPLOT


datapath_tiv = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/"
filename = "area_wd_TIV0-1600.csv"
tiv_400 <- read.csv(paste0(datapath_tiv,filename), header=FALSE)


par(mfrow=c(1,2))
boxplot(T0irg_df$wd_uv, main="Irgason Unit")
boxplot(tiv_400$V1, main="TIV Area")

filename = "area_ws_TIV0-400.csv"
tiv_400 <- read.csv(paste0(datapath_tiv,filename), header=FALSE)
par(mfrow=c(1,2))
boxplot(T0irg_df$ws_uv, main="Irgason Unit", ylim = c(0,6))
boxplot(tiv_400$V1, main="TIV Area")

###############
mean(T0irg_df$wd_uv)



calc_wd(-0.63,-1.3575)

calc_ws(-0.63,-1.3575)




