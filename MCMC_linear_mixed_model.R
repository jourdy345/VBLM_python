
MCMC_linear_mixed <- function(y, X, Z, K, A_ul, B_ul, A_eps, B_eps, sigma_beta, max_iter) {
  if (!require('mvtnorm')) {
    stop('{mvtnorm} package is required.')
  } else {
    library(mvtnorm)
    library(Matrix)
  }
  if (max_iter < 5000) {
    stop('The number of iteration should be greater than 5000 for the chain to converge.')
  }
  X <- as.matrix(X)
  Z <- as.matrix(Z)

  n <- dim(X)[1]
  p <- dim(X)[2]
  s <- dim(Z)[2]
  r <- length(K)


  # Initialize the sampling parameters
  sigma_ul <- rep(0, r)
  for (l in 1:r) {
    sigma_ul[l] <- 1 / rgamma(1, A_ul[l], rate = B_ul[l])
    A_ul[l] <- A_ul[l] + 0.5 * K[l]
  }
  sigma_eps <- 1 / rgamma(1, A_eps, rate = B_eps)
  u <- as.vector(rmvnorm(1, mean = rep(0, s), sigma = diag(1, s)))
  B_ul_prior <- B_ul

  # Storage
  beta_list <- matrix(0, ncol = p, nrow = (max_iter - 5000) / 5)
  u_list <- matrix(0, ncol = s, nrow = (max_iter - 5000) / 5)
  sigma_ul_list <- matrix(0, ncol = r, nrow = (max_iter - 5000) / 5)
  sigma_eps_list <- c(0)


  # Initialize the Markov chain
  for (i in 1:max_iter) {
    # Sampling beta
    beta_covariance <- solve((1 / sigma_eps * t(X) %*% X) + diag(1 / sigma_beta, p))
    beta_mean <- as.vector(beta_covariance %*% (1 / sigma_eps * (t(X) %*% y - t(X) %*% Z %*% u)))
    beta <- as.vector(rmvnorm(1, mean = beta_mean, sigma = beta_covariance))

    # Sampling u
    lst <- vector('list', length = r)
    for (k in 1:r) {
      lst[[k]] <- diag(1 / sigma_ul[k], K[k])
    }
    G_inverse <- bdiag(lst)

    u_covariance <- as.matrix(solve( 1/ sigma_eps * t(Z) %*% Z + G_inverse))
    u_mean <- as.vector(u_covariance %*% (1 / sigma_eps * (t(Z) %*% y - t(Z) %*% X %*% beta)))
    u <- as.vector(rmvnorm(1, mean = u_mean, sigma = u_covariance))

    # Sampling sigma_ul
    for (j in 1:r) {
      B_ul[j] <- B_ul_prior[j] + 0.5 * (u[j])^2
      sigma_ul[j] <- 1 / rgamma(1, A_ul[j] + K[j] * 0.5, rate = B_ul[j])
    }
    
    # Sampling sigma_eps
    sigma_eps <- 1 / rgamma(1, A_eps + 0.5 * n, rate = B_eps + 0.5 * sum((y - X %*% beta - Z %*% u)^2))

    if (i > 5000 && i %% 5 == 0) {
      j <- (i - 5000) / 5
      beta_list[j, ] <- beta
      u_list[j, ] <- u
      sigma_ul_list[j, ] <- sigma_ul
      sigma_eps_list[j] <- sigma_eps
    }
  }
  list('beta_list' = beta_list, 'u_list' = u_list, 'sigma_ul_list' = sigma_ul_list, 'sigma_eps_list' = sigma_eps_list)
}

library(nlme)
library(Matrix)
data(Orthodont)
head(Orthodont)
y <- as.vector(Orthodont$distance)
cat_to_num <- function(x) {
  if (x == 'Male') x <- 0
  else x <- 1
}
temp <- sapply(Orthodont$Sex, cat_to_num)
X <- as.matrix(cbind(as.vector(rep(1, length(y))), Orthodont$age, temp))
Z <- kronecker(diag(1, 27), rep(1, 4))
sigma_beta <- 10^8
A_eps <- B_eps <- 0.01
A_ul <- rep(0.01, 4)
B_ul <- rep(0.01, 4)
K <- c(4, 6, 10, 7)

result <- MCMC_linear_mixed(y, X, Z, K, A_ul, B_ul, A_eps, B_eps, sigma_beta, 10000)
apply(result$beta_list, 2, mean)
apply(result$u_list, 2, mean)
apply(result$sigma_ul_list, 2, mean)
mean(result&sigma_eps_list)

