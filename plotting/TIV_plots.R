
library(RcppCNPy)
library(raster)
library(RStoolbox)
library(rasterVis)
library(ncdf4)

u_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test.nc"
v_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test.nc"
v_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test_new_biggerIW_0-400.nc"
u_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test_new_biggerIW_0-400.nc"
v_artificial = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test_artifical_0-3.nc"
u_artificial = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test_artifical_0-3.nc"

pertub_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/Tb_pertub_cut_0-400.nc"

filebase_code <- "/home/benjamin/Met_ParametersTST/T0/code/"



source(paste0(filebase_code,"analyse_fun.R"))


netcdf_rst <- stack(pertub_file, varname = "Tb_pertub")


#for (i in seq(1,nlayers(netcdf_rst))){
  #print(i)
  #values(netcdf_rst[[i]]) = values(netcdf_rst[[i]]) / max(values(netcdf_rst[[i]]))
  
#}
#hist(netcdf_rst[[1]], breaks=c(-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5))
#hist(netcdf_rst[[2]], breaks=c(-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5))




e = extent(c(50,125,0,75))

netcdf_rst_c1 <- crop(netcdf_rst[[1]], e)
netcdf_rst_c2 <- crop(netcdf_rst[[2]], e)
par(mfrow=c(1,2))
plot(netcdf_rst_c1)
plot(netcdf_rst_c2)



v_new <- stack(v_new_file, varname="v") 
u_new <- stack(u_new_file, varname="u")
#2,length(names(v_new)))-1
i = 0

for (k in seq(0,400)){
  print(k)
  i = i+1
  if (k < 10){
    png(filename=paste0("/home/benjamin/Met_ParametersTST/T0/data/test_tiv/pngs/000",k,".png"),
        units="cm",
        width=20,
        height=20,
        res=150)
    
  } else if (k < 100){
    png(filename=paste0("/home/benjamin/Met_ParametersTST/T0/data/test_tiv/pngs/00",k,".png"),
        units="cm",
        width=20,
        height=20,
        res=150)
  } else if (k < 1000){
    png(filename=paste0("/home/benjamin/Met_ParametersTST/T0/data/test_tiv/pngs/0",k,".png"),
        units="cm",
        width=20,
        height=20,
        res=150)
    
  } else {
    png(filename=paste0("/home/benjamin/Met_ParametersTST/T0/data/test_tiv/pngs/",k,".png"),
        units="cm",
        width=20,
        height=20,
        res=150)
  }
  r2 <- stack(u_new[[i]],v_new[[i]]*-1)
  r2 = flip(r2,"y")
  
  
  values(r2) = values(r2)*5
  act_plot <- vectorplot(r2, isField = "dXY",par.settings=BuRdTheme(), region = flip(netcdf_rst[[i]],"y"),  narrows = 3000,  at = seq(-4,4, 0.5))
  print(act_plot)
  dev.off()
  
}

#region = flip(netcdf_rst[[k]],"y"),


# artificial data test


v_new <- stack(v_artificial, varname="v") 
u_new <- stack(u_artificial, varname="u")


k=1
r2 <- stack(u_new[[k]], v_new[[k]]*-1)
r2 = flip(r2,"y")

values(r2) = values(r2)*5
act_plot <- vectorplot(r2, isField = "dXY",par.settings=RdBuTheme(),    narrows = 10000,  at = seq(-4,4, 0.5))
print(act_plot)














plot(netcdf_rst_c1)
i_s <- c()
for (i in seq(-5,5)){
  for (k in seq(-5,5)){
    ext = extent(c((84.5+i),(89.5+i),(29.5+k),(34.5+k)))
    netcdf_rst_c2 <- crop(netcdf_rst[[2]], ext)
    res = t.test(values(netcdf_rst_c1), values(netcdf_rst_c2))
    print(res$p.value)
    print(i)
    print(k)
    
    
  } 
}


#e = extent(c(75,100,25,50))
u <- raster(u_file, varname="u") 
v <- raster(v_file, varname="v")
r = stack(v,u)
#r <- crop(r, e)
values(r) = values(r)*5
vectorplot(r, isField = "dXY",par.settings=RdBuTheme,  narrows = 1000)


e = extent(c(84.5,89.5,29.5,34.5))
netcdf_rst_c1 <- crop(netcdf_rst[[1]], e)
plot(netcdf_rst_c1)
ext = extent(c((84.5-5),(89.5-5),(29.5+3),(34.5+3)))
netcdf_rst_c2 <- crop(netcdf_rst[[2]], ext)
plot(netcdf_rst_c2)
hist(values(netcdf_rst_c1))
hist(values(netcdf_rst_c2))


values(v)


a <- c(2,3,4,5)
b <- c(2,3,4,5,6,7,8,9)
print(mean(a))
print(mean(b))
res_test <- t.test(a, b)
print(res_test$p.value)


(vectorplot(r, isField = "dXY", region = topo, margin = FALSE, par.settings = my.settings, 
            narrows = 1000, at = seq(0,5000, 100), main="a)")
  (vectorplot(r, isField = "dXY", margin = FALSE, 
              narrows = 1000, main="a)", par.settings=RdBuTheme))
plot(u)
df <- expand.grid(x=seq(-2, 2, .1), y=seq(-2, 2, .1))
df$z <- with(df, (3*x^2 + y)*exp(-x^2-y^2))