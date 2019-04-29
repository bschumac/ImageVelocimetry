uv2wd <- function(u,v) {
  
  degrees <- function(radians) 180 * radians / pi
  
  mathdegs <- degrees(atan2(u, v))
  wdcalc <- ifelse (mathdegs>0, mathdegs, mathdegs+360) 
  #
  wd <- wdcalc

  wd <- round(wd,2) 
  
  return(wd)
  
}


uv2ws <- function(u,v) {
  
  ws <- sqrt(u^2 + v^2)
  
  ws <- round(ws, 2)
  
  return(ws)
  
}