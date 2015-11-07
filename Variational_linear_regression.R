library(mvtnorm)

lower_bound <- function(N, D, a_N, b_N, y, X, w_N, temp, V_N_inv, a0, b0, a0_alpha, b0_alpha, a_N_alpha, b_N_alpha) {
  return(-N/2*log(2*pi) - 1/2*(a_N/b_N*t(y - X%*%w_N)%*%(y - X%*%w_N) + N*temp) + 1/2*log(det(solve(V_N_inv))) + D/2 - lgamma(a0) + a0*log(b0) - b0*a_N/b_N + lgamma(a_N) - a_N*log(b_N) + a_N - lgamma(a0_alpha) + a0_alpha*log(b0_alpha) + lgamma(a_N_alpha) - a_N_alpha*log(b_N_alpha))
}

variational_inference <- function(y, X, a0, b0, a0_alpha, b0_alpha, max_iter) {
  # initial values
  N <- as.numeric(dim(X)[1]) # number of observations
  D <- as.numeric(dim(X)[2]) # number of features
  w_N <- rmvnorm(N, mean = rep(0, N), sigma = diag(1/a0_alpha, N), method = 'chol') # initial weights
  b_N_alpha <- b0_alpha # initial prior
  
  # values that don't change
  a_N <- a0 + N/2 
  a_N_alpha <- a0_alpha + D/2
  lower_old <- -10000
  step <- 0
  for (j in 1:max_iter) {
    V_N_inv <- diag(as.numeric(a_N_alpha/b_N_alpha), D) + t(X)%*%X
    w_N <- solve(V_N_inv)%*%t(X)%*%y
    b_N <- b0 + 1/2*( t(y-X%*%w_N)%*%(y-X%*%w_N) + t(w_N)%*%diag(as.numeric(a_N_alpha/b_N_alpha), D)%*%w_N )
    b_N_alpha <- b0_alpha + 1/2*( a_N/b_N*(t(w_N)%*%w_N) + sum(diag(solve(V_N_inv))) )

    temp <- 0
    for (i in 1:N) {
      temp <- temp + t(X[i,])%*%solve(V_N_inv)%*%X[i,]
    }
    step <- step + 1
    lower_new <- lower_bound(N, D, a_N, b_N, y, X, w_N, temp, V_N_inv, a0, b0, a0_alpha, b0_alpha, a_N_alpha, b_N_alpha)
    if (abs(lower_new - lower_old) < 0.0001) break
    else lower_old <- lower_new
  }

  return(list("V_N_inv" = V_N_inv, 'w_N' = w_N, 'a_N' = a_N, 'b_N' = b_N, 'a_N_alpha' = a_N_alpha, 'b_N_alpha' = b_N_alpha, 'step' = step))
}


data(iris)
load(iris)

X <- as.matrix(iris[2:4])
y <- as.matrix(iris[1])
variational_inference(y, X, 2, 2, 2, 2, 40)