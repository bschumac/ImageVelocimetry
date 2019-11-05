library(tibble)

senperfdf = read.csv("/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test2/sensitivity_test_performance.txt", header = TRUE)
senperfdf$meanccssim <- rowMeans(cbind(senperfdf$ccussim,senperfdf$ccvssim))
senperfdf$meanccrmse<-  rowMeans(cbind(senperfdf$ccurmse,senperfdf$ccvrmse))
senperfdf$meangsssim<- rowMeans(cbind(senperfdf$gsussim,senperfdf$gsvssim))
senperfdf$meangsrmse<- rowMeans(cbind(senperfdf$gsurmse,senperfdf$gsvrmse))
senperfdf$meanrmsessim<- rowMeans(cbind(senperfdf$rmseussim,senperfdf$rmsevssim))
senperfdf$meanrmsermse<- rowMeans(cbind(senperfdf$rmseurmse,senperfdf$rmsevrmse))
senperfdf$meanssimssim<- rowMeans(cbind(senperfdf$ssimussim,senperfdf$rmsevssim))
senperfdf$meanssimrmse<- rowMeans(cbind(senperfdf$ssimurmse,senperfdf$ssimvrmse))


strongwinddf <- senperfdf[senperfdf$case == "strong_wind",]
no_winddf <- senperfdf[senperfdf$case == "no_wind",]
vertex_winddf <- senperfdf[senperfdf$case == "vertex_shedding",]


# cross-correlation

# case
#strong wind
swbest <- rbind(strongwinddf[which.max(strongwinddf$meanccssim),], strongwinddf[which.min(strongwinddf$meanccrmse),])
#no wind
nwbest<- rbind(no_winddf[which.max(no_winddf$meanccssim),], no_winddf[which.min(no_winddf$meanccrmse),])
#vertex
vsbest<- rbind(vertex_winddf[which.max(vertex_winddf$meanccssim),], vertex_winddf[which.min(vertex_winddf$meanccrmse),])

ccbestperformanceres <-rbind(swbest,nwbest,vsbest) 
ccbestperformanceres<-add_column(ccbestperformanceres, method = c(rep("cross-correlation",6)), .before = 1)


# greyscale
#strong wind
swbest <-rbind(strongwinddf[which.max(strongwinddf$meangsssim),], strongwinddf[which.min(strongwinddf$meangsrmse),])
#no wind
nwbest<-rbind(no_winddf[which.max(no_winddf$meangsssim),], no_winddf[which.min(no_winddf$meangsrmse),])
#vertex
vsbest <- rbind(vertex_winddf[which.max(vertex_winddf$meangsssim),], vertex_winddf[which.min(vertex_winddf$meangsrmse),])

gsbestperformanceres <-rbind(swbest,nwbest,vsbest) 
gsbestperformanceres<- add_column(gsbestperformanceres, method = c(rep("cross-greyscale",6)), .before = 1)


# rmse
#strong wind
swbest <-rbind(strongwinddf[which.max(strongwinddf$meanrmsessim),], strongwinddf[which.min(strongwinddf$meanrmsermse),])
#no wind
nwbest<-rbind(no_winddf[which.max(no_winddf$meanrmsessim),], no_winddf[which.min(no_winddf$meanrmsermse),])
#vertex
vsbest <- rbind(vertex_winddf[which.max(vertex_winddf$meanrmsessim),], vertex_winddf[which.min(vertex_winddf$meanrmsermse),])

rmsebestperformanceres <-rbind(swbest,nwbest,vsbest) 
rmsebestperformanceres<- add_column(rmsebestperformanceres, method = c(rep("rmse",6)), .before = 1)


# ssim
#strong wind
swbest <-rbind(strongwinddf[which.max(strongwinddf$meanssimssim),], strongwinddf[which.min(strongwinddf$meanssimrmse),])
#no wind
nwbest<-rbind(no_winddf[which.max(no_winddf$meanssimssim),], no_winddf[which.min(no_winddf$meanssimrmse),])
#vertex
vsbest <- rbind(vertex_winddf[which.max(vertex_winddf$meanssimssim),], vertex_winddf[which.min(vertex_winddf$meanssimrmse),])

ssimbestperformanceres <-rbind(swbest,nwbest,vsbest) 
ssimbestperformanceres<- add_column(ssimbestperformanceres, method = c(rep("ssim",6)), .before = 1)



total_bestperformances <- rbind(ccbestperformanceres,gsbestperformanceres,rmsebestperformanceres,ssimbestperformanceres)
write.csv(total_bestperformances, file="/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test2/analysis_sensitivity_test.csv")

