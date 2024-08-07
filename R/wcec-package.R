#' wcec: ...
#'
#' The \code{wcec} package provides ...
#'
#' @author Simon Garnier <garnier@@njit.edu>
#' @author Jason Graham <jason.graham@@scranton.edu>
#'
#' Maintainer: Simon Garnier <garnier@@njit.edu>
#'
#' @details
#' \tabular{ll}{
#'  Package: \tab wcec\cr
#'  Type: \tab Package\cr
#'  Version: \tab 0.1\cr
#'  Date: \tab 2024-08-07\cr
#'  License: \tab GPL-3\cr
#' }
#'
#' @name wcec
#' @import RcppEigen
#' @importFrom Rcpp evalCpp
#' @useDynLib wcec, .registration = TRUE
"_PACKAGE"


#' @title Four Gaussian Clusters
#'
#' @description Matrix of 2-dimensional points forming four Gaussian clusters.
#'
#' @name four_Gaussians
#'
#' @docType data
#'
#' @keywords datasets
#'
#' @examples
#' data(four_Gaussians)
#' plot(four_Gaussians, pch = 19)
#'
NULL


#' @title Three Gaussian Clusters
#'
#' @description Matrix of 2-dimensional points forming three Gaussian clusters.
#'
#' @name three_Gaussians
#'
#' @docType data
#'
#' @keywords datasets
#'
#' @examples
#' data(three_Gaussians)
#' plot(three_Gaussians, pch = 19)
#'
NULL
