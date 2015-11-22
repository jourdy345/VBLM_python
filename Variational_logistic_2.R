library(mvtnorm)
variational_inference <- function(y, X) {
  A_xi <- function(xi) -tanh(0.5 * xi) / (4 * xi)
  C_xi <- function(xi) 0.5 * xi - log(1 + exp(xi)) + xi * tanh(0.5 * xi) * 0.25

  lower_bound <- function(xi, mu_xi, Sig_xi, mu_beta, Sig_beta) {
    temp <- 0
    for (i in 1:n) {
      temp <- temp + C_xi(xi[i])
    }
    return(0.5 * (log(det(Sig_xi)) - log(det(Sig_beta)) + as.numeric(t(mu_xi)%*%solve(Sig_xi)%*%mu_xi) - as.numeric(t(mu_beta)%*%solve(Sig_beta)%*%mu_beta) + temp))
  }

  n <- dim(X)[1]
  p <- dim(X)[2]

  Sig_xi <- diag(1, n)
  mu_xi <- rep(1, n)
  xi <- drop(rmvnorm(1, rep(1, n), diag(1, n)))
  Sig_beta <- diag(p, p)
  mu_beta <- drop(rmvnorm(n = 1, mean = rep(0.1, p), sigma = Sig_beta))
  lower_old <- lower_bound(xi, mu_xi, Sig_xi, mu_beta, Sig_beta)

  step <- 0
  while (TRUE) {
    step <- step + 1

    Sig_xi <- solve(solve(Sig_beta) - 2 * t(X)%*%diag(A_xi(xi))%*%X)
    mu_xi <- Sig_xi %*% (t(X) %*% (y - 0.5 * rep(1, n)) + solve(Sig_beta, mu_beta))
    pre_xi <- X %*% (Sig_xi + mu_xi %*% t(mu_xi)) %*% t(X)
    xi <- sqrt(diag(pre_xi))
    lower_new <- lower_bound(xi, mu_xi, Sig_xi, mu_beta, Sig_beta)
    if (abs(lower_new - lower_old) < .Machine$double.eps) break
    else lower_old <- lower_new
  }
  return(list('mu_xi' = mu_xi, 'step' = step))
}

my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
my_data <- as.matrix(my_data)
y <- unname(my_data[, 1])
X <- unname(my_data[, 2:4])

variational_inference(y, X)