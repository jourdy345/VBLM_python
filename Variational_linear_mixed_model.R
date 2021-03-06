variational_inference <- function(y, X, Z, K, A_ul_prior, B_ul_prior, A_eps_prior, B_eps_prior, sigma_beta_prior, max_iter) {
  # if (!require('Matrix')) {
  #   stop("You need to install {Matrix} package. Try install.packages('Matrix').")
  # } else {
  #   library(Matrix)
  # }
  X <- as.matrix(X)
  Z <- as.matrix(Z)
  
  n <- dim(X)[1]
  p <- dim(X)[2]
  s <- dim(Z)[2]
  r <- length(K)
  lower_bound <- function(K, A_ul_opt, B_ul_opt, A_eps_opt, B_eps_opt, A_ul_prior, B_ul_prior, A_eps_prior, B_eps_prior, Sigma_u_opt, Sigma_beta_opt, mu_beta_opt, mu_u_opt, sigma_beta_prior) {
    # Computing E(G^{-1})
    lst <- vector('list', length = r)
    for (j in 1:r) {
      lst[[j]] <- (A_ul_opt[j] / B_ul_opt[j]) * diag(1, K[j])
    }
    block_temp <- as.matrix(bdiag(lst))
    
    # Computing Σ {K_{l} / 2 * E(ln(σ_{ul}^{-2}))
    temp <- 0
    for (i in 1:r) {
      temp <- temp + 0.5 * K[i] * (digamma(A_ul_opt[i]) - log(B_ul_opt[i]))
    }
    
    # Computing Σ (A_ul_prior - 1) * E(ln(σ_{ul}^{-2})) - B_ul_prior * E(σ_{ul}^{-2})
    temp2 <- 0
    for (k in 1:r) {
      temp2 <- temp2 + (A_ul_prior[k] - 1) * (digamma(A_ul_opt[k]) - log(B_ul_opt[k])) - B_ul_prior[k] * A_ul_opt[k] / B_ul_opt[k]
    }
    
    # Computing Σ (A_ul_opt - 1) * E(ln(σ_{ul}^{-2})) - B_ul_opt * E(σ_{ul}^{-2})
    temp3 <- 0
    for (l in 1:r) {
      temp3 <- temp3 + (A_ul_opt[l] - 1) * (digamma(A_ul_opt[l]) - log(B_ul_opt[l])) - B_ul_opt[l] * B_ul_opt[l] / A_ul_opt[l] + A_ul_opt[l] * log(B_ul_opt[l]) - lgamma(B_ul_opt[l])
    }
    
    0.5 * n * (digamma(A_eps_opt) - log(B_eps_opt)) - 0.5 * A_eps_opt / B_eps_opt * (sum(y - (X %*% mu_beta_opt + Z %*% mu_u_opt))^2 + sum(diag(t(X) %*% X %*% Sigma_beta_opt)) + sum(diag(t(Z) %*% Z %*% Sigma_u_opt))) + temp - 0.5 * sum(diag(block_temp %*% (Sigma_u_opt + mu_u_opt %*% t(mu_u_opt)))) - 0.5 * sum(mu_beta_opt^2) + temp2 + (A_eps_prior - 1) * (digamma(A_eps_opt) - log(B_eps_opt)) - B_eps_prior * A_eps_opt / B_eps_opt + 0.5 * (log(det(Sigma_beta_opt)) + log(det(Sigma_u_opt))) - temp3  - A_eps_opt * log(B_eps_opt) + lgamma(A_eps_opt) - (A_eps_opt - 1) * (digamma(A_eps_opt) - log(B_eps_opt)) + B_eps_opt * A_eps_opt / B_eps_opt
  }
  
  # Initialize variational parameters for update
  lower_old <- -Inf
  B_ul_opt <- rep(0.01, r)
  B_eps_opt <- 0.01
  mu_u_opt <- rep(0, s)
  A_ul_opt <- rep(0, r)
  
  # Updating variational parameters
  for (v in 1:r) {
    A_ul_opt[v] <- A_ul_prior[v] + K[v] * 0.5
  }
  A_eps_opt <- A_eps_prior + 0.5 * n
  step <- 0
  for (i in 1:max_iter) {
    step <- step + 1
    
    # beta_opt ~ N(mu_beta_opt, Sigma_beta_opt)
    Sigma_beta_opt <- solve((A_eps_opt / B_eps_opt) * t(X) %*% X + diag(sigma_beta_prior, p))
    mu_beta_opt <- Sigma_beta_opt %*% ((A_eps_opt / B_eps_opt) * (t(X) %*% y - t(X) %*% Z %*% mu_u_opt))
    
    # u_opt ~ N(mu_u_opt, Sigma_u_opt)
    # Computing E(G^{-1})
    lst2 <- vector('list', length = r)
    for (j in 1:r) {
      lst2[[j]] <- (A_ul_opt[j] / B_ul_opt[j]) * diag(1, K[j])
    }
    block_temp2 <- as.matrix(bdiag(lst2))
    
    Sigma_u_opt <- solve((A_eps_opt / B_eps_opt) * t(Z) %*% Z + block_temp2)
    mu_u_opt <- Sigma_u_opt %*% ((A_eps_opt / B_eps_opt) * (t(Z) %*% y - t(Z) %*% X %*% mu_beta_opt))
    
    # sigma_ul ~ IG(A_ul + Kl, B_ul + 0.5 * (E(u)_l)^2)
    for (v in 1:r) {
      B_ul_opt[v] <- B_ul_prior[v] + 0.5 * (diag(Sigma_u_opt)[v] + (mu_u_opt[v])^2)
    }
    
    # sigma_eps ~ IG(A_eps_prior + 0.5 * n, B_eps_prior + 0.5 * (t(y - Xmu_beta_opt - Zmu_u_opt) %*% (y - Xmu_beta_opt - Zmu_u_opt) + tr(t(X)XSigma_beta_opt + t(Z)ZSigma_u_opt)))
    B_eps_opt <- B_eps_prior + 0.5 * (sum((y - X %*% mu_beta_opt - Z %*% mu_u_opt)^2) + sum(diag(t(X) %*% X %*% Sigma_beta_opt)) + sum(diag(t(Z) %*% Z %*% Sigma_u_opt)))
    
    # Check convergence
    lower_new <- lower_bound(K, A_ul_opt, B_ul_opt, A_eps_opt, B_eps_opt, A_ul_prior, B_ul_prior, A_eps_prior, B_eps_prior, Sigma_u_opt, Sigma_beta_opt, mu_beta_opt, mu_u_opt, sigma_beta_prior)
    if (abs(lower_new - lower_old) < .Machine$double.eps) break
    else lower_old <- lower_new
  } 
  
  list('mu_beta' = mu_beta_opt, 'Sigma_beta' = Sigma_beta_opt, 'mu_u' = mu_u_opt,'Sigma_u' = Sigma_u_opt, 'A_ul' = A_ul_opt, 'B_ul' = B_ul_opt, 'A_eps' = A_eps_opt, 'B_eps' = B_eps_opt, 'step' = step)  
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
sigma_beta_prior <- 10^8
A_eps_prior <- B_eps_prior <- 0.01
A_ul_prior <- rep(0.01, 4)
B_ul_prior <- rep(0.01, 4)
K <- c(4, 6, 10, 7)

variational_inference(y, X, Z, K, A_ul_prior, B_ul_prior, A_eps_prior, B_eps_prior, sigma_beta_prior, 40)

