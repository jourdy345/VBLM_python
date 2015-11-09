## In this logistic regression, the training set has to be either 1 or -1. Note that it is not 0 or 1.
## A simple way to modify the training set is to switch 0 and 1 with (2y-1).


variational_inference <- function(y, X, a0, b0, max_iter) {
  
  # functions required for lower bound
  sigmoid <- function(x) 1/(1+exp(-x))
  lambda_xi <- function(x) 1/(2*x) * ( sigmoid(x) - 1/2)

  # lower bound function
  lower_bound <- function(w_N, V_N_inv, xi, a0, b0, a_N, b_N) {
    return(1/2*t(w_N)%*%V_N_inv%*%w_N + 1/2*log(det(solve(V_N_inv))) + sum(log(sigmoid(xi)) - xi/2 + lambda_xi(xi)*xi) - lgamma(a0) + a0*log(b0) - b0*a_N/b_N - a_N*log(b_N) + lgamma(a_N) + a_N)
  }
  
  # Setting the necessary initial values
  N <- as.numeric(dim(X)[1])
  D <- as.numeric(dim(X)[2])
  a_N <- a0 + D/2
  b_N <- b0
  xi <- rep(1, N)
  lower_old <- -Inf
  step <- 0

  for (j in 1:max_iter) {
    
    # Updating V_N
    temp_V <- matrix(0, nrow = D, ncol = D)
    for (i in 1:N) {
      temp_V <- temp_V + xi[i]*X[i,]%*%t(X[i,])
    }
    V_N_inv <- diag(as.numeric(a_N/b_N), D) + 2*temp_V
    
    # Updating w_N
    temp_w <- rep(0, D)
    for (i in 1:N) {
      temp_w <- temp_w + y[i]/2 * X[i,]
    }
    w_N <- solve(V_N_inv)%*%temp_w
    
    # Updating b_N
    b_N <- b0 + 1/2*( t(w_N)%*%w_N + sum(diag(solve(V_N_inv))) )

    # Updating xi
    xi_new <- rep(0, N)
    for (k in 1:N) {
      xi_new[k] <- t(X[k,])%*%(solve(V_N_inv) + w_N%*%t(w_N))%*%X[k,]
    }

    # Computing the lower bound
    step <- step + 1
    lower_new <- lower_bound(w_N, V_N_inv, xi_new, a0, b0, a_N, b_N)

    if (abs(lower_old - lower_new) < .Machine$double.eps) break
    else lower_old <- lower_new
    
  }

  return(list('V_N_inv' = V_N_inv, 'w_N' = w_N, 'a_N' = a_N, 'b_N' = b_N, 'lower bound' = lower_new, 'step' = step))
}



my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
my_data <- as.matrix(my_data)
y <- 2*(unname(my_data[, 1])) - 1
X <- unname(my_data[, 2:4])

variational_inference(y, X, 2, 2, 40)