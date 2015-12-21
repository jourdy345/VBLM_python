## Linear mixed model

## Model specification
# Likelihood
# y|β, u, G, R ~ N(Xβ + Zu, R)
# u|G ~ N(0, G)
# G = blockdiag(σ1^2*Ι_{Κ1}, σ2^2*Ι_{Κ2}, ... , σr^2*I_{Kr})
# R = σε^2*Ι

## Prior
# β ~ N(0, σβ^2*Ι) -> sig_beta
# σl^2 ~ IG(Al, Bl), 1 <= l <= r
# σε^2 ~ IG(Aε, Bε)

## Dimensions
# X = (n x p) design matrix
# β = (n x 1) parameter vector
# Z = (n x s) design matrix
# u = (s x 1) random effect vector
# R = (p x p) covariance matrix
# G = (s x s) covariance matrix




variational_inference <- function() {
  if (!require('Matrix')) {
    stop("You need to install {Matrix} package. Try install.packages('Matrix').")
  } else {
    library(Matrix)
  }
  X <- as.matrix(X)
  Z <- as.matrix(Z)


  lower_bound <- function(X, Z, K, A_l, B_l, sig_beta, sigma_bu, sigma_beta, B_opt_sigl, B_opt_sigeps) {
    n <- dim(X)[1]
    p <- dim(X)[2]
    s <- dim(Z)[2]
    r <- length(K)
    temp <- 0
    for (i in r) {
      temp <- temp + (A_l[i] * log(B_l[i]) - (A_l[i] + 0.5 * K[i]) * log(B_opt_sigl[i]) + lgamma(A_l[i] + 0.5 * K[i]) - lgamma(A_l[i]))
    }

    return(0.5 * (p + s) - 0.5 * n * log(2 * pi) - 0.5 * p * log(sig_beta) + 0.5 * log(det(sigma_bu)) - 1 / (2 * sig_beta) * (sum((mu_beta^2)) + sum(diag(sigma_beta))) + A_eps * log(B_eps) - (A_eps + 0.5 * n) * log(B_opt_sigeps) + lgamma(A_eps + 0.5 * n) - lgamma(A_eps) + temp)
  }

  mu






  C <- cbind(X, Z)
  block_temp <- 1 / sig_beta * diag(1, p)
  lst <- vector('list', length = (r + 1))
  lst[[1]] <- block_temp
  for (i in 2:(s+1)) {
    lst[[i]] <- A_l[i-1] + 0.5 * K[i-1]) / B_opt_sigl[i-1]) * diag(1, K[i-1])
  }
  block_temp2 <- bdiag(lst)
  sigma_bu <- (A_eps + 0.5 * n) / B_opt_sigeps * t(C)%*%C + block_temp2
  mu_beta <- ((A_eps + 0.5 * n) / B_opt_sigeps) * sigma_bu %*% t(C) %*% y
  B_opt_sigeps <- B_eps + 0.5 * (sum((y - C %*% mu_beta)^2) + sum(diag(t(C) %*% C %*% sigma_bu)))
  for (j in 1:r) {
    B_opt_sigl[j] <- B_l[j] + 0.5 * (mu_)
    
  }
}


