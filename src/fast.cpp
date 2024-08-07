#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

//' @name .wcov
//'
//' @title Weighted Covariance Matrix
//'
//' @description \code{wcov} computes the estimates of the weighted covariance
//'  matrix and the weighted mean of the data.
//'
//' @param x A matrix with \eqn{m} columns and \eqn{n} rows, where each column
//'  represents a different variable and each row a different observation.
//'
//' @param w A non-negative and non-zero vector of weights for each observation.
//'  Its length must equal the number of rows of x.
//'
//' @return A list with two components:
//'  \itemize{
//'   \item \code{center}: an estimate for the center (mean) of the data.
//'   \item \code{cov}: the estimated (weighted) covariance matrix.
//'  }
//'
//' @author Simon Garnier, \email{garnier@@njit.edu}
//'
//' @examples
//' m <- matrix(c(rnorm(500, 6), rnorm(500, 11, 3)), ncol = 2)
//' w <- runif(500)
//' gravitree:::.wcov(m, w)
//'
// [[Rcpp::export(.wcov)]]
Rcpp::List wcov(Eigen::MatrixXd &x, Eigen::VectorXd &w) {
  Eigen::VectorXd center(x.cols());
  double ws = w.sum();
  for (int i = 0; i < x.cols(); i++) {
    center(i) = (x.col(i).array() * w.array()).sum() / ws;
  }

  int p = x.cols();
  Eigen::VectorXd sqw = (w.array() / ws).cwiseSqrt();
  Eigen::MatrixXd X = x.array().rowwise() - center.transpose().array();
  Eigen::MatrixXd cov = Eigen::MatrixXd(p, p)
                            .setZero()
                            .selfadjointView<Eigen::Lower>()
                            .rankUpdate(X.transpose() * sqw.asDiagonal());

  return Rcpp::List::create(Rcpp::_["center"] = center, Rcpp::_["cov"] = cov);
}
