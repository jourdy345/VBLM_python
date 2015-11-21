variational_inference <- function(y, X, mu_prior, sigma_prior, max_iter = 40) {
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  lower_bound <- function(mu_b, mu_prior, sigma_prior) {
    return(sum(y * log(pnorm(X%*%mu_b))) + sum((rep(1, n) - y) * log(rep(1, n) - pnorm(X %*% mu_b))) - 0.5 * t(mu_b - mu_prior) %*% solve(sigma_prior) %*% (mu_b - mu_prior) - 0.5 * log(det(sigma_prior %*% t(X) %*% X + diag(1, p))))
  }

  # initialize the auxiliary variable 'a'
  mu_a <- rep(1, n)
  mu_b <- rep(0, p)
  step <- 0
  lower_old <- lower_bound(mu_b, mu_prior, sigma_prior)
  for (i in 1:max_iter) {
    step <- step + 1
    mu_b <- solve(t(X) %*% X + solve(sigma_prior)) %*% (t(X) %*% mu_a + solve(sigma_prior) %*% mu_prior)
    mu_a <- X %*% mu_b + dnorm(X %*% mu_b) / (pnorm(X %*% mu_b)^y * pnorm(X %*% mu_b - rep(1, n))^(rep(1, n) - y))

    lower_new <- lower_bound(mu_b, mu_prior, sigma_prior)
    if (abs(lower_new - lower_old) < .Machine$double.eps) break
    else lower_old <- lower_new
  }
  return(list('mu_b' = mu_b, 'step' = step))
}

my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
my_data <- as.matrix(my_data)
y <- unname(my_data[, 1])
X <- unname(my_data[, 2:4])

variational_inference(y, X, rep(2, dim(X)[2]), diag(2, dim(X)[2]))