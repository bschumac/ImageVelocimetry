
library(openair)



P1vis <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P1_visROSmean.csv", header=FALSE )
P1vis_direc <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P1_visROSmean_direc.csv", header=FALSE )
colnames(P1vis) <- "vis"
colnames(P1vis_direc) <- "vis_direc"
P1TIV <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P1_TIVROSmean.csv", header=FALSE )
colnames(P1TIV) <- "TIV"
P1TIV_direc <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P1_TIVROSmean_direc.csv", header=FALSE )
colnames(P1TIV_direc) <- "TIV_direc"



P2vis <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P2_visROSmean.csv", header=FALSE )
colnames(P1vis) <- "vis"
P2vis_direc <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P2_visROSmean_direc.csv", header=FALSE )
colnames(P2vis_direc) <- "vis_direc"
P2TIV <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P2_TIVROSmean.csv", header=FALSE )
colnames(P2TIV) <- "TIV"
P2TIV_direc <- read.csv("/home/benjamin/Met_ParametersTST/FIRE/Darfield_2019/P2_TIVROSmean_direc.csv", header=FALSE )
colnames(P2TIV_direc) <- "TIV_direc"





P1_vis <- data.frame(ws=P1vis, wd=P1vis_direc-17, nox=P1vis)
colnames(P1_vis) <- c("ws", "wd", "Speed")
P1_tiv <- data.frame(ws=P1TIV, wd=P1TIV_direc+90, nox=P1TIV)
colnames(P1_tiv) <- c("ws", "wd", "Speed")

P2_vis <- data.frame(ws=P2vis, wd=P2vis_direc+90, nox=P2vis)
colnames(P2_vis) <- c("ws", "wd", "Speed")
P2_tiv <- data.frame(ws=P2TIV, wd=P2TIV_direc+23, nox=P2TIV)
colnames(P2_tiv) <- c("ws", "wd", "Speed")

sequen = seq(0,1.5,0.01)
my.settings <- list(
  par.main.text = list(font = 1.5, # make it bold
                       just = "left", 
                       x = grid::unit(5, "mm")))
k = list(at=sequen, col=openColours("default", length(sequen)), labels= list(labels=seq(0,1.5,0.25), 
          at=seq(0,1.5,0.25)))
a <- polarPlot(P1_tiv, pollutant= "Speed",  units= "m/s", key.header= "m/s",key = k, main="a) TIV-FFL Velocity Plot 1", auto.text=F, par.settings=my.settings)


#b <- polarPlot(P1_vis, pollutant= "ws")
b <- polarPlot(P1_vis, pollutant= "Speed", units= "m/s", key.header= "m/s",key = k, main="b) VPFT-FFL Velocity Plot 1", auto.text=F, par.settings=my.settings)#b <- polarPlot(P1_vis, pollutant= "ws")
c <- polarPlot(P2_tiv, pollutant= "Speed", grid.line=c(0,5), units= "m/s", key.header= "m/s",key = k, main="c) TIV-FFL Velocity Plot 2", auto.text=F, par.settings=my.settings)#b <- polarPlot(P1_vis, pollutant= "ws")
d <- polarPlot(P2_vis, pollutant= "Speed", grid.line=c(0,5), units= "m/s", key.header= "m/s",key = k, main="d) VPFT-FFL Velocity Plot 2", auto.text=F, par.settings=my.settings)#b <- polarPlot(P1_vis, pollutant= "ws")



#c <- polarPlot(P2_tiv, pollutant= "ws")
#d <- polarPlot(P2_vis, pollutant= "ws") 
#trace(polarPlot, edit=T) change line ~413 

print(a,split=c(1, 1, 2, 2))
print(b,split=c(2, 1, 2, 2), newpage= FALSE)
print(c,split=c(1, 2, 2, 2), newpage= FALSE)
print(d,split=c(2, 2, 2, 2), newpage= FALSE)
