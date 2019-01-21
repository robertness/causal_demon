# Forward simulation of endogenous variables
h <- function(N, p) ifelse(N < p, 1, 0)

f_PKC <- Vectorize(function(N){
  p = .5804
  h(N, .5804)
})

f_PKA  <- Vectorize(function(PKC, N){
  p = ifelse(PKC == 1.0, .9521, .5423)
  h(N, p)
})

f_Jnk <- Vectorize(function(PKC, PKA, N){
  if(PKC == 1.0){
    p = .46
  }else{
    p = ifelse(PKA == 1.0, .2696, .7155)
  }
  h(N, p)
})

f_P38 <- Vectorize(function(PKC, PKA, N){
  if(PKC == 1.0){
    p = ifelse(PKA == 1.0, .1946, .3263)
  }else{
    p = ifelse(PKA == 1.0, .1245, .7025)
  }
  h(N, p)
})

f_Raf <- Vectorize(function(PKC, PKA, N){
  if(PKA == 1.0){
    p = .39
  }else{
    p = ifelse(PKC == 1.0, .6180, .9379)
  }
  h(N, p)
})

f_Mek <- Vectorize(function(PKC, PKA, Raf, N){
  if(PKC == 1.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .6822, .7848)
  }else if(PKC == 1.0 && Raf == 0.0){
    p = .2342
  }else if(PKC == 0.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .4311, .8869)
  } else if(PKC == 0.0 && Raf == 0.0){
    p = ifelse(PKA == 1.0, .1030, .3750)
  }
  h(N, p)
})

f_Erk <- Vectorize(function(PKA, Mek, N){
  if(Mek == 1.0){
    p = .95;
  } else {
    p = ifelse(PKA == 1.0, .8909, .1565)
  }
  h(N, p);
})

f_Akt <- Vectorize(function(Erk, PKA, N){
  if(Erk == 1.0){
    p <- ifelse(PKA == 1, .3277, .8073)
  } else {
    p <- 0.3107
  }
  h(N, p)
})

# Backward inference of exogenous noise variables
g <- function(X, p, m = 100) ifelse(X == 1, list(runif(m, 0, p)), list(runif(m, p, 1)))

r_N_PKC <- Vectorize(function(PKC){
  p <- .5804
  g(PKC, p)
})

r_N_PKA  <- Vectorize(function(PKC, PKA){
  p = ifelse(PKC == 1.0, .9521, .5423)
  g(PKA, p)
})

r_N_Jnk <- Vectorize(function(PKC, PKA, Jnk){
  if(PKC == 1.0){
    p = .46
  }else{
    p = ifelse(PKA == 1.0, .2696, .7155)
  }
  g(Jnk, p)
})

r_N_P38 <- Vectorize(function(PKC, PKA, P38){
  if(PKC == 1.0){
    p = ifelse(PKA == 1.0, .1946, .3263)
  }else{
    p = ifelse(PKA == 1.0, .1245, .7025)
  }
  g(P38, p)
})

r_N_Raf <- Vectorize(function(PKC, PKA, Raf){
  if(PKA == 1.0){
    p = .39
  }else{
    p = ifelse(PKC == 1.0, .6180, .9379)
  }
  g(Raf, p)
})

r_N_Mek <- Vectorize(function(PKC, PKA, Raf, Mek){
  if(PKC == 1.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .6822, .7848)
  }else if(PKC == 1.0 && Raf == 0.0){
    p = .2342
  }else if(PKC == 0.0 && Raf == 1.0){
    p = ifelse(PKA == 1.0, .4311, .8869)
  } else if(PKC == 0.0 && Raf == 0.0){
    p = ifelse(PKA == 1.0, .1030, .3750)
  }
  g(Mek, p)
})

r_N_Erk <- Vectorize(function(PKA, Mek, Erk){
  if(Mek == 1.0){
    p = .95;
  } else {
    p = ifelse(PKA == 1.0, .8909, .1565)
  }
  g(Erk, p);
})

r_N_Akt <- Vectorize(function(Erk, PKA, Akt){
  if(Erk == 1.0){
    p <- ifelse(PKA == 1, .3277, .8073)
  } else {
    p <- 0.3107
  }
  g(Akt, p)
})
