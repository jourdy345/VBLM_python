// [[Rcpp:: depends ( RcppArmadillo )]]
#include <RcppArmadillo.h>
#include <cmath>
using namespace Rcpp; using namespace arma;

// [[Rcpp::export]]
double A_xi (double xi) {
  return(-tanh(0.5 * xi) / (4 * xi));
}

double C_xi (double xi) {
  return(0.5 * xi - log(1 + exp(xi)) + xi * tanh(0.5 * xi) * 0.25);
}

double lower_bound (arma::vec xi, arma::vec mu_xi, arma::mat Sig_xi, arma::vec mu_beta, arma::mat Sigma_beta) {
  double temp = 0;
  int n = mu_xi.size();
  for (int i = 0; i<n; ++i) {
    temp = temp + C_xi(xi(i));
  }
  return(0.5 * (log(det(Sig_xi)) - log(det(Sig_beta)) + (trans(mu_xi) * solve(Sig_xi) * mu_xi) - (trans(mu_beta) * solve(Sig_beta)%*%mu_beta) + temp))
}

arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

List VB_logisticCpp (arma::vec y, arma::mat X, double a0, double b0, int max_iter) {
  int n = X.nrow(), p = X.ncol();
  arma::mat Sig_xi = arma::eye(n, n);
  arma::vec temp_mean = 
  arma::vec xi = mvrnormArma

  return Rcpp::List::create(Rcpp::Named("mu_xi") = mu_xi, Rcpp::Named("step") = step)
}