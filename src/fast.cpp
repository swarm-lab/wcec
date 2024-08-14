// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

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
//' wcec:::.wcov(m, w)
//'
// [[Rcpp::export(.wcov)]]
Rcpp::List wcov(const Eigen::MatrixXd &x, const Eigen::VectorXd &w) {
  Eigen::VectorXd normW = w / w.sum();
  Eigen::VectorXd center = normW.transpose() * x;
  Eigen::MatrixXd normX = normW.cwiseSqrt().asDiagonal() * (x.rowwise() - center.transpose());
  Eigen::MatrixXd cov = normX.transpose() * normX; // Slow
  return Rcpp::List::create(Rcpp::_["center"] = center, Rcpp::_["cov"] = cov);
}

// [[Rcpp::export(.wmean)]]
Eigen::VectorXd wmean(const Eigen::MatrixXd &x, const Eigen::VectorXd &w) {
  Eigen::VectorXd normW = w / w.sum();
  return normW.transpose() * x;
}

// [[Rcpp::export(.log_mvd)]] 
Eigen::VectorXd log_mvd(const Eigen::MatrixXd &x, const Eigen::VectorXd &mu,
                         const Eigen::MatrixXd &sigma) {
  Eigen::MatrixXd diff = x.rowwise() - mu.transpose();
  Eigen::LLT<Eigen::MatrixXd> llt(sigma);
  Eigen::MatrixXd sigmaInv = llt.solve(Eigen::MatrixXd::Identity(sigma.rows(), sigma.cols()));
  Eigen::VectorXd exponent = -0.5 * (diff * sigmaInv).cwiseProduct(diff).rowwise().sum();
  double constant = 0.5 * (mu.size() * std::log(2 * M_PI) + std::log(sigma.determinant()));
  return exponent.array() - constant;
}
