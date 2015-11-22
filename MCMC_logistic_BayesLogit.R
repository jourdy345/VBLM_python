library(mvtnorm)
logistic_MCMC <- function(y, X, burnin, sample_size) {
  TRUNC = 0.64
  cutoff = 1 / TRUNC;

  ## pigauss - cumulative distribution function for Inv-Gauss(mu, lambda).
  ##------------------------------------------------------------------------------
  pigauss <- function(x, mu, lambda)
  {
    Z = 1.0 / mu;
    b = sqrt(lambda / x) * (x * Z - 1);
    a = -1.0 * sqrt(lambda / x) * (x * Z + 1);
    y = exp(pnorm(b, log.p=TRUE)) + exp(2 * lambda * Z + pnorm(a, log.p=TRUE));
                                          # y2 = 2 * pnorm(-1.0 / sqrt(x));
    y
  }

  q.and.p <- function(Z)
  {
    fz = pi^2 / 8 + Z^2 / 2;
    p = (0.5 * pi) * exp( -1.0 * fz * TRUNC) / fz;
    q = 2 * exp(-1.0 * Z) * pigauss(TRUNC, 1.0/Z, 1.0);

    list("q"=q, "p"=p, "qdivp"=q/p);
  }

  mass.texpon <- function(Z)
  {
    x = TRUNC;
    fz = pi^2 / 8 + Z^2 / 2;
    b = sqrt(1.0 / x) * (x * Z - 1);
    a = -1.0 * sqrt(1.0 / x) * (x * Z + 1);

    x0 = log(fz) + fz * TRUNC;
    xb = x0 - Z + pnorm(b, log.p=TRUE);
    xa = x0 + Z + pnorm(a, log.p=TRUE);

    qdivp = 4 / pi * ( exp(xb) + exp(xa) );

    1.0 / (1.0 + qdivp);
  }

  mass.detail <- function(Z)
  {
    x = TRUNC;
    fz = pi^2 / 8 + Z^2 / 2;
    b = sqrt(1.0 / x) * (x * Z - 1);
    a = -1.0 * sqrt(1.0 / x) * (x * Z + 1);

    x0 = log(fz) + fz * TRUNC;
    xb = x0 - Z + pnorm(b, log.p=TRUE);
    xa = x0 + Z + pnorm(a, log.p=TRUE);

    qdivp = 4 / pi * ( exp(xb) + exp(xa) );

    m = 1.0 / (1.0 + qdivp);
    p = cosh(Z) * 0.5 * pi * exp(-x0);
    q = p * (1/m - 1);
    
    out = list("qdivp"=qdivp, "m"=m, "p"=p, "q"=q, "c"=p+q, "qdivp2"=q/p);

    out
  }

  ## rtigauss - sample from truncated Inv-Gauss(1/abs(Z), 1.0) 1_{(0, TRUNC)}.
  ##------------------------------------------------------------------------------
  rtigauss <- function(Z, R=TRUNC)
  {
    Z = abs(Z);
    mu = 1/Z;
    X = R + 1;
    if (mu > R) {
      alpha = 0.0;
      while (runif(1) > alpha) {
        ## X = R + 1
        ## while (X > R) {
        ##     X = 1.0 / rgamma(1, 0.5, rate=0.5);
        ## }
        E = rexp(2)
        while ( E[1]^2 > 2 * E[2] / R) {
          E = rexp(2)
        }
        X = R / (1 + R*E[1])^2
        alpha = exp(-0.5 * Z^2 * X);
      }
    }
    else {
      while (X > R) {
        lambda = 1.0;
        Y = rnorm(1)^2;
        X = mu + 0.5 * mu^2 / lambda * Y -
          0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2);
        if ( runif(1) > mu / (mu + X) ) {
          X = mu^2 / X;
        }
      }
    }
    X;
  }

  ## rigauss - sample from Inv-Gauss(mu, lambda).
  ##------------------------------------------------------------------------------
  rigauss <- function(mu, lambda)
  {
    nu = rnorm(1);
    y  = nu^2;
    x  = mu + 0.5 * mu^2 * y / lambda -
      0.5 * mu / lambda * sqrt(4 * mu * lambda * y + (mu*y)^2);
    if (runif(1) > mu / (mu + x)) {
      x = mu^2 / x;
    }
    x
  }

  ## Calculate coefficient n in density of PG(1.0, 0.0), i.e. J* from Devroye.
  ##------------------------------------------------------------------------------
  a.coef <- function(n,x)
  {
    if ( x>TRUNC )
      pi * (n+0.5) * exp( -(n+0.5)^2*pi^2*x/2 )
    else
      (2/pi/x)^1.5 * pi * (n+0.5) * exp( -2*(n+0.5)^2/x )
  }

  ## Samples from PG(n=1.0, psi=Z)
  ## ------------------------------------------------------------------------------
  rpg.devroye.1 <- function(Z)
  {
    Z = abs(Z) * 0.5;

    ## PG(1,z) = 1/4 J*(1,Z/2)
    fz = pi^2 / 8 + Z^2 / 2;
    ## p = (0.5 * pi) * exp( -1.0 * fz * TRUNC) / fz;
    ## q = 2 * exp(-1.0 * Z) * pigauss(TRUNC, 1.0/Z, 1.0);

    num.trials = 0;
    total.iter = 0;

    while (TRUE)
      {
        num.trials = num.trials + 1;

        if ( runif(1) < mass.texpon(Z) ) {
          ## Truncated Exponential
          X = TRUNC + rexp(1) / fz
        }
        else {
          ## Truncated Inverse Normal
          X = rtigauss(Z)
        }

        ## C = cosh(Z) * exp( -0.5 * Z^2 * X )

        ## Don't need to multiply everything by C, since it cancels in inequality.
        S = a.coef(0,X)
        Y = runif(1)*S
        n = 0

        while (TRUE)
          {
            n = n + 1
            total.iter = total.iter + 1;
            if ( n %% 2 == 1 )
              {
                S = S - a.coef(n,X)
                if ( Y<=S ) break
              }
            else
              {
                S = S + a.coef(n,X)
                if ( Y>S ) break
              }
          }

        if ( Y<=S ) break
      }

    ## 0.25 * X
    list("x"=0.25 * X, "n"=num.trials, "total.iter"=total.iter)
  }
  
  # MCMC
  y <- as.numeric(y) 
  X <- as.matrix(X)

  n <- dim(X)[1]
  p <- dim(X)[2]
  m <- rep(0.1, p)
  Sig <- diag(length(m), p)
  beta <- drop(rmvnorm(n = 1, mean = m, sigma = Sig))
  beta_list <- matrix(0, ncol = p)

  for (j in 1:sample_size) {
    w <- c()
    for (i in 1:n) {
      b <- abs(sum(X[i, ] * beta)) # dot product
      w[i] <- rpg.devroye.1(b)$x
    }
    Sigma <- solve(t(X)%*%diag(w)%*%X + solve(Sig))
    Mean <- Sigma%*%(t(X)%*%(y - 0.5*rep(1, n)) + solve(Sig)%*%m)
    beta <- rmvnorm(n = p, mean = Mean, sigma = Sigma)
    if (j > burnin & j %% 5 == 0) {
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
result <- logistic_MCMC(y, X, 500, 2000)
print('result')
print(result)
print(str(result))
apply(result, 2, mean)