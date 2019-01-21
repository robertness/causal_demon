p_PKC <- Vectorize(function(){
  p <- .5804
  rbinom(1, 1, p)
})

p_PKA  <- Vectorize(function(PKC){
  p = ifelse(PKC == 1.0, .9521, .5423)
  rbinom(1, 1, p)
})

p_Jnk <- Vectorize(function(PKC, PKA){
  if(PKC == 1.0){
    p = .46
  }else{
    p = ifelse(PKA == 1.0, .2696, .7155)
  }
  rbinom(1, 1, p)
})

p_P38 <- Vectorize(function(PKC, PKA){
  if(PKC == 1.0){
    p = ifelse(PKA == 1.0, .1946, .3263)
  }else{
    p = ifelse(PKA == 1.0, .1245, .7025)
  }
  rbinom(1, 1, p)
})

p_Raf <- Vectorize(function(PKC, PKA){
  if(PKA == 1.0){
    p = .39
  }else{
    p = ifelse(PKC == 1.0, .6180, .9379)
  }
  rbinom(1, 1, p)
})

p_Mek <- Vectorize(function(PKC, PKA, Raf){
  if(PKC == 1.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .6822, .7848)
  }else if(PKC == 1.0 && Raf == 0.0){
    p = .2342
  }else if(PKC == 0.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .4311, .8869)
  } else if(PKC == 0.0 && Raf == 0.0){
    p = ifelse(PKA == 1.0, .1030, .3750)
  }
  rbinom(1, 1, p)
})

p_Erk <- Vectorize(function(PKA, Mek){
  if(Mek == 1.0){
    p = .95;
  } else {
    p = ifelse(PKA == 1.0, .8909, .1565)
  }
  rbinom(1, 1, p)
})

p_Akt <- Vectorize(function(Erk, PKA){
  if(Erk == 1.0){
    p <- ifelse(PKA == 1, .3277, .8073)
  } else {
    p <- 0.3107
  }
  h(N, p)
})
