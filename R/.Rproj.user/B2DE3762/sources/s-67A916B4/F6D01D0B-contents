functions {
  real h(real N, real p){
    if(N < p){
      return 1.0;
    }else{
      return 0.0;
    }
  }
  real f_PKC(real N){
     real p;
     p = .5804;
     return h(N, p);
  }
  real f_PKA(real PKC, real N){
    real p;
    p = PKC == 1.0 ? .9521 : .5423;
    return h(N, p);
  }
  real f_Jnk(real PKC, real PKA, real N){
    real p;
    if(PKC == 1.0){
      p = .46;
    }else{
      p = PKA == 1.0 ? .2696 : .7155;
    }
    return h(N, p);
  }
  real f_P38(real PKC, real PKA, real N){
    real p;
    if(PKC == 1.0){
      p = PKA == 1.0 ? .1946 : .3263;
    }else{
      p = PKA == 1.0 ? .1245 : .7025;
    }
    return h(N, p);
  }
  real f_Raf(real PKC, real PKA, real N){
    real p;
    if(PKA == 1.0){
      p = .39;
    }else{
      p = PKC == 1.0 ? .6180: .9379;
    }
    return h(N, p);
  }
  real f_Mek(real PKC, real PKA, real Raf, real N){
    real p;
    if(PKC == 1.0 && Raf == 1.0){
      p = PKA == 1.0 ? .6822 : .7848;
    }else if(PKC == 1.0 && Raf == 0.0){
      p = .2342;
    }else if(PKC == 0.0 && Raf == 1.0){
      p = PKA == 1.0 ? .4311 : .8869;
    } else if(PKC == 0.0 && Raf == 0.0){
      p = PKA == 1.0 ? .1030 : .3750;
    }
    return h(N, p);
  }
  real f_Erk(real PKA, real Mek, real N){
    real p;
    if(Mek == 1.0){
      p = .95;
    } else {
      p = PKA == 1.0 ? .8909 : .1565;
    }
    return h(N, p);
  }
  real f_Akt(real Erk, real PKA, real N){
    real p;
    if(Erk == 1.0){
      p = PKA == 1.0 ? .3277 : .8073;
    } else {
      p = 0.3107;
    }
    return h(N, p);
  }
}
data {
  real<lower=0.0, upper=1.0> PKC;
  real<lower=0.0, upper=1.0> PKA;
  real<lower=0.0, upper=1.0> Jnk;
  real<lower=0.0, upper=1.0> P38;
  real<lower=0.0, upper=1.0> Raf;
  real<lower=0.0, upper=1.0> Mek;
  real<lower=0.0, upper=1.0> Erk;
  real<lower=0.0, upper=1.0> Akt;
}
parameters {
  real<lower=0.0, upper=1.0> N_PKC;
  real<lower=0.0, upper=1.0> N_PKA;
  real<lower=0.0, upper=1.0> N_Jnk;
  real<lower=0.0, upper=1.0> N_P38;
  real<lower=0.0, upper=1.0> N_Raf;
  real<lower=0.0, upper=1.0> N_Mek;
  real<lower=0.0, upper=1.0> N_Erk;
  real<lower=0.0, upper=1.0> N_Akt;
}
transformed parameters {
  real PKC_mu;
  real PKA_mu;
  real Jnk_mu;
  real P38_mu;
  real Raf_mu;
  real Mek_mu;
  real Erk_mu;
  real Akt_mu;

  PKC_mu = f_PKC(N_PKC);
  PKA_mu = f_PKA(PKC, N_PKA);
  Jnk_mu = f_Jnk(PKC, PKA, N_Jnk);
  P38_mu = f_P38(PKC, PKA, N_P38);
  Raf_mu = f_Raf(PKC, PKA, N_Raf);
  Mek_mu = f_Mek(PKC, PKA, Raf, N_Mek);
  Erk_mu = f_Erk(PKA, Mek, N_Erk);
  Akt_mu = f_Akt(Erk, PKA, N_Akt);
}
model {
  target += uniform_lpdf(N_PKC | 0, 1);
  target += uniform_lpdf(N_PKA | 0, 1);
  target += uniform_lpdf(N_Jnk | 0, 1);
  target += uniform_lpdf(N_P38 | 0, 1);
  target += uniform_lpdf(N_Raf | 0, 1);
  target += uniform_lpdf(N_Mek | 0, 1);
  target += uniform_lpdf(N_Erk | 0, 1);
  target += uniform_lpdf(N_Akt | 0, 1);

  target += normal_lpdf(PKC | PKC_mu, 1e-6);
  target += normal_lpdf(PKA | PKA_mu, 1e-6);
  target += normal_lpdf(Jnk | Jnk_mu, 1e-6);
  target += normal_lpdf(P38 | P38_mu, 1e-6);
  target += normal_lpdf(Raf | Raf_mu, 1e-6);
  target += normal_lpdf(Mek | Mek_mu, 1e-6);
  target += normal_lpdf(Erk | Erk_mu, 1e-6);
  target += normal_lpdf(Akt | Akt_mu, 1e-6);
}
