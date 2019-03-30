library(raster)
library(R.matlab)
library(ncdf4)
library(RStoolbox)

T0Temp <- stack("/home/benjamin/Met_ParametersTST/data/Tb_pertub.nc", varname = "Tb_pertub")

#file_nc_obj <- nc_open("/home/benjamin/Met_ParametersTST/data/Tb_pertub.nc", verbose = TRUE)
T0Temp_rot <- t(T0Temp)

# delete first layer because it is all 0
T0Temp_rot<-T0Temp_rot[[3:3058]]
T0Temp_rot_flip <- flip(T0Temp_rot,"x")
writeRaster(T0Temp_rot_flip, filename = "/home/benjamin/Met_ParametersTST/data/Tb_pertub_rotated_flipped.nc", format="CDF", varname = "Tb_pertub")

pca <- rasterPCA(T0Temp_rot_flip, nComp = 6)


T0Temp <- stack("/home/benjamin/Met_ParametersTST/data/Tb_pertub_rotated_flipped.nc", varname = "Tb_pertub")

library(RColorBrewer)

T0TempMean<- mean(T0Temp)
plot(T0TempMean, col = col)
mean(values(T0TempMean))





col <- brewer.pal(11, "Spectral")

pc1 <- pca$map$PC1
pc1_scaled <- scale(pc1)
pc2 <- pca$map$PC2
pc2_scaled <- scale(pc2)
pc3 <- pca$map$PC3
pc3_scaled <- scale(pc3)
pc4 <- pca$map$PC4
pc4_scaled <- scale(pc4)
pc5 <- pca$map$PC5
pc5_scaled <- scale(pc5)
pc6 <- pca$map$PC6
pc6_scaled <- scale(pc6)


pc1_scaled[pc1_scaled< -2] <- NA

pca_scaled <- stack(pc1_scaled,pc2_scaled, pc3_scaled, pc4_scaled,pc5_scaled, pc6_scaled)
plot(pca_scaled, col=rev(col))

writeRaster(pca$map, filename = "/home/benjamin/Met_ParametersTST/data/Tb_pertub_rotated_flipped_pca.tif", format="GTiff")

setOption("max.print",10000)
sumpca <- capture.output(summary(pca$model))
write(sumpca, file="/home/benjamin/Met_ParametersTST/data/summarypca")


str(summarypca)
