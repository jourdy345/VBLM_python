library(mvtnorm)
linear_MCMC <- function(y = NULL, X = NULL, a0_lambda = NULL, b0_lambda = NULL, a0_alpha = NULL, b0_alpha = NULL, max_iter = NULL) {
  if (is.null(y) || is.null(X)) stop("The model requires both y and X.")
  if (!is.matrix(y) || !is.matrix(X)) stop("y and X both need to be matrices.")
  if (is.null(a0_lambda)) a0_lambda <- 100
  if (is.null(b0_lambda)) b0_lambda <- 100
  if (is.null(a0_alpha)) b0_alpha <- 100
  if (is.null(b0_alpha)) b0_alpha <- 100
  if (is.null(max_iter)) {
    max_iter <- 10000
    print("The default number of iteration is 10,000.")
  }
  N <- as.numeric(dim(X)[1])
  D <- as.numeric(dim(X)[2])
  
  w_list <- matrix(0, ncol = D, nrow = (max_iter - 5000) / 5)
  alpha_list <- c(0)
  lambda_list <- c(0)

  alpha <- 100
  lambda <- 100
  for (i in 1:max_iter) {
    Sigma <- solve(lambda*(t(X)%*%X + diag(alpha, D)))
    mu <- lambda*(Sigma%*%t(X)%*%y)
    w <- rmvnorm(1, mean = mu, sigma = Sigma, method = 'chol')
    w <- t(w)

    a_N_lambda <- a0_lambda + N/2
    b_N_lambda <- 1/2 * sum((y - X%*%w)^2) + b0_lambda
    lambda <- rgamma(1, a_N_lambda, b_N_lambda)

    a_N_alpha <- a0_alpha + D/2
    b_N_alpha <- 1/2 * t(w)%*%w + b0_alpha
    alpha <- rgamma(1, a_N_alpha, b_N_alpha)


    if (i > 5000 && i %% 5 == 0) {
      j <- (i - 5000) / 5
      w_list[j,] <- w
      lambda_list[j] <- lambda
      alpha_list[j] <- alpha
    }
  }

  return(list('w' = w_list, 'lambda' = lambda_list, 'alpha' = alpha_list))
}

data(iris)
head(iris, 2)

X <- as.matrix(unname(iris[2:4]))
y <- as.matrix(unname(iris[1]))

start_time <- Sys.time()
linear_model <- linear_MCMC(y, X, a0_lambda = 2, b0_lambda = 2, a0_alpha = 2, b0_alpha = 2, max_iter = 30000)
w_list_mix <- linear_model$w
apply(w_list_mix, 2, mean)
end_time <- Sys.time()
time_taken <- end_time - start_time
print(time_taken)