.sub2ind <- function(dims, subs) {
  k <- c(1, cumprod(dims[-length(dims)]))
  Rfast::rowsums((subs - 1) * matrix(k, nrow = nrow(subs), ncol = length(k), byrow = TRUE)) + 1
}

.init_density <- function(x, k, w) {
  nr <- nrow(x)
  nc <- ncol(x)
  n_blocks <- k + 2
  blocks <- matrix(NA_integer_, nr, nc)

  for (i in seq_len(nc)) {
    breaks <- seq(
      from = min(x[, i], na.rm = TRUE),
      to = max(x[, i], na.rm = TRUE),
      length.out = n_blocks
    )
    blocks[, i] <- .bincode(x[, i], breaks, right = FALSE, include.lowest = TRUE)
  }

  fq <- tabulate(.sub2ind(rep(k + 1, nc), blocks), (k + 1)^nc)
  ord <- order(fq, decreasing = TRUE)
  idx <- arrayInd(ord, rep(k + 1, nc))
  init <- matrix(NA, nrow = k, ncol = nc)
  blocks <- Rfast::transpose(blocks)

  for (i in seq_len(k)) {
    ix <- Rfast::colAll(blocks == idx[1, ])
    init[i, ] <- .wmean(x[ix, , drop = FALSE], w[ix])
    d <- Rfast::rowMaxs(Rfast::eachrow(idx, idx[1, ], oper = "-")^2, TRUE)
    idx <- idx[d > sqrt(n_blocks), , drop = FALSE]
  }

  Rfast::rowMins(
    Rfast::dista(init, x, trans = FALSE, square = FALSE)
  )
}

.array_split <- function(x, k) {
  nr <- nrow(x)
  n <- rep(floor(nr / k), k)
  r <- nr - sum(n)

  for (i in seq_len(r)) {
    n[i] <- n[i] + 1
  }

  end <- cumsum(n)
  start <- c(1, end[1:(k - 1)] + 1)

  l <- lapply(1:k, function(i) {
    x[start[i]:end[i], , drop = FALSE]
  })
  names(l) <- paste0(start, "-", end)
  l
}

.init_sharding <- function(x, k) {
  ord <- order(Rfast::rowsums(x))
  xsplit <- .array_split(x[ord, ], k)
  Rfast::rowMins(
    Rfast::dista(x, t(sapply(xsplit, Rfast::colmeans)))
  )
}

.init_clusters <- function(x, k, w, method) {
  n <- nrow(x)

  if (n != length(w)) {
    stop("The number of elements in `w` is not the same as the number of rows in `x`.")
  }

  if (length(k) > 1) {
    if (n != length(k)) {
      stop("The number of elements in `k` is not the same as the number of rows in `x`.")
    }

    if (!all((floor(k) - k) == 0)) {
      stop("Not all values in `k` are integers.")
    }

    suk <- Rfast::sort_unique(k)
    lookup <- cbind(suk, as.numeric(as.factor(suk)))
    match(k, lookup)
  } else if (length(k) == 1) {
    if (k > 1) {
      if (method == "sharding") {
        .init_sharding(x, k)
      } else if (method == "density") {
        .init_density(x, k, w)
      } else if (method == "random") {
        sample(1:k, n, TRUE)
      } else {
        stop("Invalid initialization method.")
      }
    } else {
      rep(1, n)
    }
  } else {
    stop("Invalid 'k'.")
  }
}

.cost_gaussian <- function(p, n, params) {
  s <- 0
  for (i in seq_along(p)) {
    s <- s + (p[i] * n * log(2.0 * pi * exp(1)) + log(det(params[[i]]$cov))) / 2
  }
  s
}

.wcec <- function(x, k, w, iter_max) {
  n <- nrow(x)
  nk <- max(k)
  out <- list(
    clustering = k,
    probability = tabulate(k, nk) / n,
    params = lapply(1:nk, function(j) {
      idx <- k == j
      .wcov(x[idx, , drop = FALSE], w[idx])
    }),
    iter = 0
  )
  out$cost <- .cost_gaussian(out$probability, n, out$params)
  d <- matrix(NA_real_, nrow = n, ncol = length(out$params))

  for (i in 1:iter_max) {
    logP <- log(out$probability)

    for (j in seq_along(out$params)) {
      d[, j] <- -logP[j] - .log_mvd(x, out$params[[j]]$center, out$params[[j]]$cov)
    }

    new_clustering <- Rfast::as_integer(Rfast::rowMins(d), FALSE)
    test <- Rfast::all_equals(new_clustering, out$clustering)
    out$clustering <- new_clustering

    if (test == TRUE) {
      break
    } else {
      for (j in 1:nk) {
        idx <- out$clustering == j
        out$params[[j]] <- .wcov(x[idx, , drop = FALSE], w[idx])
      }

      out$probability <- tabulate(out$clustering) / n
      out$cost <- .cost_gaussian(out$probability, n, out$params)
    }
  }

  out$iter <- i
  out
}

#' @export
wcec <- function(
    x, k, w = rep(1 / nrow(x), nrow(x)), iter_max = 10,
    init_method = "sharding") {
  # CHECKS

  # INITIALIZATION
  k <- .init_clusters(x, k, w, method = init_method)

  # FIRST PASS
  out <- .wcec(x = x, k = k, w = w, iter_max = iter_max)

  out
}
