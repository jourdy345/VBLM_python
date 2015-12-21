# Optimization of the S matrix is going to be impractical since the Hessian matrix will be (p * p * p * p) 4th-order tensor.
# We instead apply Barzilai-Borwein algorithm (a special type of gradient-based optimization method).
variational_inference <- function(y, X, mu, Sigma, max_iter) {
  # mu and Sigma are the mean vector and covariance matrix of the prior distribution of beta
  n <- dim(X)[1]
  p <- dim(X)[2]
  # y <- as.vector(y)
  # X <- unname(as.matrix(X))
  # mu <- as.vector(mu)
  # Sigma <- as.matrix(Sigma)

  lower_bound <- function(w, S, y, X, mu, Sigma) {
    -sum(y * (X%*%w)) - sum(exp(X%*%w + 0.5 * diag(X%*%S%*%t(X)))) - 0.5 * sum((w - mu) * solve(Sigma, w - mu)) - 0.5 * sum(diag(solve(Sigma)%*%S)) + 0.5 * log(det(S)) - 0.5 * log(det(Sigma)) + 0.5 * p - sum(log(factorial(y)))
  }

  w_gradient <- function(w, S, y, X, Sigma) {
    as.vector(t(y)%*%X - t(w)%*%solve(Sigma) - t(mu)%*%solve(Sigma) - t(exp(X%*%w + 0.5 * diag(X%*%S%*%t(X))))%*%X)
  }

  S_gradient <- function(w, S, X, Sigma) {
    -0.5 * (t(X) %*% diag(as.vector(exp(X%*%w + 0.5 * diag(X%*%S%*%t(X))))) %*% X + solve(S) - solve(Sigma))
  }

  step <- 0

  # initialise w for Newton-Raphson method
  w <- rep(0, p)

  # initialise 2 S matrices for Barzilai-Borwein method
  library(dlm)
  # S_old <- rwishart(df = n - p, p = p)
  # S_new <- rwishart(df = n - p, p = p)
  temp1 <- matrix(runif(p * p), nrow = p)
  temp2 <- matrix(runif(p * p), nrow = p)
  S_old <- t(temp1) %*% temp1
  S_new <- t(temp2) %*% temp2

  # initial lower bound
  lower_old <- lower_bound(w, S_new, y, X, mu, Sigma)
  initial_lower_bound <- lower_old

  for (i in 1:max_iter) {
    step <- step + 1
    # optimising S (covariance matrix)
    dS <- S_new - S_old
    print(solve(S_new))
    temp <- exp(X%*%w + 0.5 * diag(X%*%S_new%*%t(X)))
    G_old <- S_gradient(w, S_old, X, Sigma)
    G_new <- S_gradient(w, S_new, X, Sigma)
    dG <- G_new - G_old
    lambda <- sum(dS * dG) / sum(dG * dG)
    S_old <- S_new
    S_new <- S_new - lambda * G_new
    print(G_new)
    print(step)
    lower_new <- lower_bound(w, S_new, y, X, mu, Sigma)

    if (abs(lower_new - lower_old) < .Machine$double.eps) break
    else lower_old <- lower_new
  }

  for (i in 1:max_iter) {
    step <- step + 1
    # optimising w (mean vector) ... Newton-Raphson method
    e <- as.vector(X %*% w + 0.5 * diag(X %*% S_new %*% t(X)))
    hessian_w <- t(X) %*% diag(e) %*% X - solve(Sigma)
    print(hessian_w)
    w <- w - solve(hessian_w, w_gradient(w, S_new, y, X, Sigma))
    lower_new <- lower_bound(w, S_new, y, X, mu, Sigma)
    if (abs(lower_new - lower_old) < .Machine$double.eps) break
    else lower_old <- lower_new
  }
  print(paste0('final w: ', w))

  return(list('m' = w, 'S' = S_new, 'step' = step, 'lower_bound' = lower_new, 'initial_lower_bound' = initial_lower_bound))
}

setwd('~/Desktop/Computer_related/Variational/linearmodel/VBLM_python/')

library(mvtnorm)
crab <- read.table('crab.txt')
crab <- crab[ ,-1]
crab <- unname(as.matrix(crab))
y <- as.vector(crab[ ,5])
X <- cbind(rep(1, nrow(crab)), crab[ , 1:4])
mu <- as.vector(rmvnorm(1, mean = rep(0, ncol(crab)), sigma = diag(rep(1000, ncol(crab)))))
Sigma <- diag(rep(1000, ncol(crab)))
variational_inference(y, X, mu, Sigma, 40)
