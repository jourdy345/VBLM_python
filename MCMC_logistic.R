library(mvtnorm)
logistic_MCMC <- function(y, X, burnin, sample_size) {
  polya_gamma_generate <- function(z) {
    # CDF of inverse Gaussian
    inverse_gaussian_cdf <- function(q, lambda, mu) {
      return(pnorm(sqrt(lambda/q) * (q/mu - 1)) + exp(2*lambda/mu) * pnorm(-sqrt(lambda/q) * (q/mu + 1)))
    }

    # probability integral transform of inverse Gaussian
    inverse_gaussian_generate <- function(n, lambda, mu){
      temp <- c()
      while (length(temp) < n) {
        nu <- rnorm(1)
        y <- nu^2
        x <- mu + 0.5 * mu^2 * y / lambda - 0.5 * mu / lambda * sqrt(4 * mu * lambda * y * (mu * y)^2)
        if (runif(1) > mu / (mu + x)) {
          x <- mu^2 / x
          temp <- c(temp, x)
        }
      }
      return(temp)
    }

    trunc_inv_gaussian_generate <- function(z, t = 2/pi) {
      z <- abs(z)
      mu <- 1/z
      X <- t + 1
      if (mu > t) {
        alpha <- 0
        while (runif(1) > alpha) {
          E <- rexp(2)
          while (E[1]^2 > 2 * E[2] / t) {
            E <- rexp(2)
          }
          X <- t / (1 + t * E[1])^2
          alpha <- exp(-0.5 * z^2 * X)
        }
      } else {
        while(X > t) {
          X <- inverse_gaussian_generate(1, lambda = 1.0, mu = mu)
        }
      }
      return(X)
    }

    mass.texpon <- function(z) {
      x <- 2/pi
      fz <- pi^2 / 8 + z^2 / 2
      b <- sqrt(1/x) * (x * z - 1)
      a <- -sqrt(1/x) * (x * z + 1)

      x0 <- log(fz) + fz * 2/pi
      xb <- x0 - z + pnorm(b, log.p = TRUE)
      xa <- x0 + z + pnorm(a, log.p = TRUE)

      qdivp <- 4 / pi * (exp(xb) + exp(xa))
      return(1 / (1 + qdivp))
    }

    # define piecewise coefficient
    an <- function(x, n, t = 2/pi) {
      if (t <= 0) stop('t should be greater than 0')
      if (x > t) {
        return(pi*(n + 1/2)* exp(-(n + 1/2)^2 * pi^2 * x / 2))
      } else {
        return(pi*(n + 1/2)*(2/(pi*x))^(3/2) * exp(-(2*(n + 1/2)/x)))
      }
    }

    # generate Polya-Gamma variate
    z <- abs(z) / 2
    t <- 2 / pi
    K <- pi^2 / 8 + z^2 / 2
    # p <- pi / (2 * K) * exp(-K * t)
    # q <- 2 * exp(-z) * inverse_gaussian_cdf(q = t, lambda = 1, mu = 1/z)
    num.trials <- 0
    total.iter <- 0

    while (TRUE) {
      num.trials <- num.trials + 1

      if (runif(1) < mass.texpon(z)) {
        # Truncated Exponential
        X <- t + rexp(1) / K
      } else {
        # Truncated Inverse Gaussian
        X <- trunc_inv_gaussian_generate(z)
      }

      S <- an(X, 0)
      print(S)
      Y <- runif(1) * S
      print('Y')
      print(Y)
      n <- 0

      while (TRUE) {
        n <- n + 1
        total.iter <- total.iter + 1
        if (n %% 2 == 1) {
          S <- S - an(X, n)
          if (Y <= S) break
        } else {
          S <- S + an(X, n)
          if (Y > S) break
        }
      }
      if (Y <= S) break
    }
    list('x' = 0.25 * X, 'n' = num.trials, 'total.iter' = total.iter)
  }

  # MCMC
  y <- as.numeric(y) 
  X <- as.numeric(X)
  X <- as.matrix(X)

  w <- c()
  n <- nrow(X)
  p <- ncol(X)
  m <- rep(0.1, p)
  Sig <- diag(length(m), p)
  beta <- rmvnorm(n = p, mean = m, sigma = Sig)
  beta_list <- matrix(0, ncol = p)

  for (j in 1:sample_size) {
    for (i in 1:n) {
      b <- abs(sum(X[i, ] * beta)) # dot product
      w[i] <- polya_gamma_generate(b)$x
    }
    Sigma <- solve(t(X)%*%diag(w)%*%X + solve(Sig))
    Mean <- Sigma%*%(t(X)%*%(y - 0.5*rep(1, n)) + solve(Sig)%*%m)
    beta <- rmvnorm(n = p, mean = Mean, sigma = Sigma)
    if (j > burnin & j && 5 == 0) {
      beta_list <- rbind(beta_list, beta)
    }
  }
  beta_list <- beta_list[-1, ]
  return(beta_list)
}


my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
my_data <- as.matrix(my_data)
y <- unname(my_data[, 1])
X <- unname(my_data[, 2:4])
result <- logistic_MCMC(y, X, 500, 1000)
print('result')
print(result)
# apply(result, 2, mean)
