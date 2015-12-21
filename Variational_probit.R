# variational_inference <- function(y, X, mu_prior, sigma_prior, max_iter = 40) {
#   n <- dim(X)[1]
#   p <- dim(X)[2]
  
#   lower_bound <- function(mu_b, mu_prior, sigma_prior) {
#     return(sum(y * log(pnorm(X%*%mu_b))) + sum((rep(1, n) - y) * log(rep(1, n) - pnorm(X %*% mu_b))) - 0.5 * t(mu_b - mu_prior) %*% solve(sigma_prior) %*% (mu_b - mu_prior) - 0.5 * log(det(sigma_prior %*% t(X) %*% X + diag(1, p))))
#   }

#   # initialize the auxiliary variable 'a'
#   mu_a <- rep(1, n)
#   mu_b <- rep(0, p)
#   step <- 0
#   lower_old <- lower_bound(mu_b, mu_prior, sigma_prior)
#   for (i in 1:max_iter) {
#     step <- step + 1
#     mu_b <- solve(t(X) %*% X + solve(sigma_prior)) %*% (t(X) %*% mu_a + solve(sigma_prior) %*% mu_prior)
#     mu_a <- X %*% mu_b + dnorm(X %*% mu_b) / (pnorm(X %*% mu_b)^y * pnorm(X %*% mu_b - rep(1, n))^(rep(1, n) - y))

#     lower_new <- lower_bound(mu_b, mu_prior, sigma_prior)
#     if (abs(lower_new - lower_old) < .Machine$double.eps) break
#     else lower_old <- lower_new
#   }
#   return(list('mu_b' = mu_b, 'step' = step))
# }







variational_inference <- function(X, draws){
  lower.bound2 <- function(betas, X, stars, sigs, draws){
    ##the code is a straightforward implementation
    ##of the lower bound, divided into parts
    ab<- t(X)%*%X%*%(betas%*%t(betas) + sigs)
    part1<- sum(diag(ab))/2
    part2<- t(betas)%*%betas + sum(diag(sigs))
    part2<- part2*(beta.mat.prior[1,1])
    part2<- part2/2 + (1/2)*log(det(solve(beta.mat.prior))) + (length(betas)/2)*log(2*pi)
    part3<- t(stars)%*%stars
    part4<- length(betas)/2 + (1/2)*log(det(sigs)) + (length(betas)/2)*log(2*pi)
    bounds<- part1 + part2 + part3/2 + part4
    parts<- c(-part1, -part2, part3/2, part4)
    ##we will return the lower bound and the constituent
    ##parts, useful for monitoring convergence of the algorithm
    bounds<- list(bounds, parts)
    names(bounds)<- c("bounds", "parts")
    return(bounds)
  }
  
  ##the variance covariance for the beta priors,
  ##can easily be added as an argument
  beta.mat.prior<- diag(1/100, ncol(X))
  ##for teaching purposes, it is useful to store each updated
  ##mean vector for the beta approximating distribution
  beta.VA<- matrix(NA, nrow=1000, ncol=ncol(X))
  ##we begin with random values for the augmented data
  require(msm)
  ystars<- rep(0, nrow(X))
  for(j in 1:nrow(X)){
    ystars[j]<- ifelse(draws[j]==1, rtnorm(1, mean=0.5, sd=1, lower=0, upper=Inf),
    rtnorm(1, mean=-0.5, sd=1, lower=-Inf, upper=0) )
  }
  ##we will store the progress of the lower bound on the model
  bounds<- c()
  zz<- 0
  ##this stores the parts of the lower bound
  parts<- matrix(NA, nrow=1000,ncol=4)
  j<- 0
  ##creating a while loop
  while(zz==0){
    j<- j + 1

    ##updating the beta parameters
    beta.VA[j,]<- solve(t(X)%*%X + beta.mat.prior)%*%t(X)%*%ystars
    ##this does not need to be in the loop
    ##(it doesn"t change over observations)
    ##but is placed here for teaching purposes
    sigs<- solve(t(X)%*%X + beta.mat.prior)
    ##computing the inner product of the
    ##covariates current estimates of the coefficients
    stars<- X%*%beta.VA[j,]
    denom1<- pnorm(-stars)
    num1<- dnorm(-stars)
    ##now, computing the expected value for each
    ##individual"s augmented data, given
    ##current estimates of the approximating
    ##distribution on the coefficients
    ystars[which(draws==0)]<- stars[draws==0] - num1[which(draws==0)]/denom1[which(draws==0)]
    ystars[which(draws==1)]<- stars[draws==1] + num1[which(draws==1)]/(1 - denom1[which(draws==1)])
    ##calculating the lower bound
    trial <- lower.bound2(beta.VA[j,], X, ystars, sigs, draws)
    bounds[j] <- trial$bounds
    parts[j,] <- trial$parts
    if(j>1){
      ##observing convergence
      ab<-abs(bounds[j]- bounds[j-1])
      if(ab<1e-8){
        zz<-1
      }
    }
  }
  ##the information to be returned, after convergence
  stuff<- list(bounds, beta.VA[j,], sigs)
  names(stuff)<- c("bound","betas", "sigma")
  return(stuff)
}


my_data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
my_data <- as.matrix(my_data)
y <- unname(my_data[, 1])
X <- unname(my_data[, 2:4])

variational_inference(y, X)
