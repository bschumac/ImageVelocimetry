library(raster)
library(R.matlab)
library(ncdf4)
library(RStoolbox)
library(stringr)
library(rgdal)
library(gdalUtils)
gdalbuildvrt("/home/benjamin/Met_ParametersTST/T0/data/DEM_Chch_tif/*.tif", "/home/benjamin/Met_ParametersTST/T0/data/test.vrt")

flslst <- list.files("/home/benjamin/Met_ParametersTST/T0/data/DEM_Chch_tif", pattern=".tif")
flslst = flslst[!grepl(".aux",flslst)]

rasterlst <- list()
length(flslst)/4


for (i in seq(1,200)){
  print(paste0("Reading Tile... ",i))
  rasterlst[[i]] <- raster(paste0("/home/benjamin/Met_ParametersTST/T0/data/DEM_Chch/",flslst[i]))
  
}
rasterlst$fun <- mean
y <- do.call(mosaic,rasterlst)

plot(y)


hum_flux <- hum_flux[[1:12]]
pca <- rasterPCA(hum_flux, nComp = 1)
plot(pca$map)
summary(pca$model)

?

raster(paste0("/home/benjamin/Met_ParametersTST/T0/data/DEM_Chch/",flslst[i]))

