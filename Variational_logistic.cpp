// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
using namespace Rcpp; using namespace arma;

// [[Rcpp::export]]
double A_xi (double xi) {
  return(-tanh(0.5 * xi) / (4 * xi));
}

// [[Rcpp::export]]
double C_xi (double xi) {
  return(0.5 * xi - log(1 + exp(xi)) + xi * tanh(0.5 * xi) * 0.25);
}

// [[Rcpp::export]]
double lower_bound (arma::vec xi, arma::vec mu_xi, arma::mat Sig_xi, arma::vec mu_beta, arma::mat Sig_beta) {
  double temp = 0;
  int n = mu_xi.size();
  for (int i = 0; i<n; ++i) {
    temp = temp + C_xi(xi(i));
  }
  return(arma::as_scalar(0.5 * (log(det(Sig_xi)) - log(det(Sig_beta)) + (trans(mu_xi) * arma::inv_sympd(Sig_xi) * mu_xi) - (trans(mu_beta) * arma::inv_sympd(Sig_beta) * mu_beta) + temp)));
}

// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
   int ncols = sigma.n_cols;
   arma::mat Y = arma::randn(n, ncols);
   return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

// [[Rcpp::export]]
Rcpp::List VB_logisticCpp (arma::vec y, arma::mat X) {
  int n = X.n_rows, p = X.n_cols;
  arma::mat Sig_xi = arma::eye<mat>(n, n);
  arma::colvec temp_mean = arma::ones<colvec>(n);
  arma::mat temp_sigma = arma::eye<mat>(n, n);
  arma::colvec mu_xi = arma::ones<colvec>(n);
  arma::colvec xi = arma::vectorise(mvrnormArma(1, temp_mean, temp_sigma));
  arma::mat Sig_beta = arma::eye<mat>(p, p) * p ;
  arma::colvec temp_mean2 = arma::ones<colvec>(p) * 0.1;
  arma::colvec mu_beta = arma::vectorise(mvrnormArma(1, temp_mean2, Sig_beta));
  double lower_old = -10000;

  int step = 0;
  while (TRUE) {
    step += 1;

    int t = xi.size();
    for (int i=0; i<t; ++i) {
      xi(i) = A_xi(xi(i));
    }
    // std::for_each(xi.begin(), xi.end(), A_xi);
    Sig_xi = (Sig_beta.i() - 2 * X.t() * arma::diagmat(xi) * X).i();
    mu_xi = Sig_xi * (X.t() * (y - 0.5 * arma::ones<vec>(n)) + arma::solve(Sig_beta, mu_beta));
    arma::mat pre_xi = X * (Sig_xi + mu_xi * mu_xi.t()) * X.t();
    xi = sqrt(pre_xi.diag());
    double lower_new = lower_bound(xi, mu_xi, Sig_xi, mu_beta, Sig_beta);
    if (std::abs(lower_new - lower_old) < 0.00001) break;
    else lower_old = lower_new;
  }


  return Rcpp::List::create(Rcpp::Named("mu_xi") = Rcpp::wrap(mu_xi), Rcpp::Named("step") = Rcpp::wrap(step));
}