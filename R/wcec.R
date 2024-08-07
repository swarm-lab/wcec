.init_density <- function(x, k, w) {
  n_blocks <- k + 2
  block_list <- apply(x, 2, function(y) {
    breaks <- seq(
      from = min(y, na.rm = TRUE),
      to = max(y, na.rm = TRUE),
      length.out = n_blocks
    )
    cut(y, breaks, labels = FALSE, include.lowest = TRUE, right = FALSE)
  }, simplify = FALSE)
  fq <- table(block_list)
  ord <- order(fq, decreasing = TRUE)
  idx <- arrayInd(ord, dim(fq), dimnames(fq))
  init <- matrix(NA, nrow = k, ncol = ncol(x))
  blocks <- simplify2array(block_list)

  for (i in seq_len(k)) {
    ix <- Rfast::rowAll(sweep(blocks, 2, idx[1, ], "=="))
    init[i, ] <- apply(x[ix, , drop = FALSE], 2, weighted.mean, w = w[ix])
    d <- Rfast::rowMaxs(sweep(idx, 2, idx[1, ])^2, TRUE)
    idx <- idx[d > sqrt(n_blocks), , drop = FALSE]
  }

  Rfast::rowMins(
    Rfast::dista(x, init)
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

    suk <- sort(unique(k))
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

.ce_gaussian <- function(n, cov) {
  (n * log(2.0 * pi * exp(1)) + log(det(cov))) / 2
}

.cost_gaussian <- function(p, n, params) {
  sum(
    sapply(1:length(p), function(i) {
      .ce_gaussian(p[i] * n, params[[i]]$cov)
    })
  )
}

.dist_gaussian <- function(x, p, mu, sigma) {
  -log(p) - log(mvtnorm::dmvnorm(x, mean = mu, sigma = sigma, checkSymmetry = FALSE))
}

.wcec <- function(x, k, w, iter_max) {
  n <- nrow(x)
  out <- list(
    clustering = k,
    probability = table(k) / n,
    params = lapply(unique(k), function(clust) {
      idx <- k == clust
      wcec:::.wcov(
        x[idx, , drop = FALSE],
        w[idx]
      )
    }),
    iter = 0
  )
  d <- matrix(Inf, nrow = n, ncol = length(out$params))
  out$cost <- wcec:::.cost_gaussian(out$probability, n, out$params)

  for (i in 1:iter_max) {
    for (j in seq_along(out$params)) {
      d[, j] <- wcec:::.dist_gaussian(x, out$probability[j], out$params[[j]]$center, out$params[[j]]$cov)
    }

    new_clustering <- Rfast::rowMins(d)
    new_clustering <- match(new_clustering, sort(unique(new_clustering)))
    test <- all(new_clustering == out$clustering)
    out$clustering <- new_clustering

    if (test == TRUE) {
      break
    } else {
      out$params <- lapply(unique(out$clustering), function(clust) {
        idx <- out$clustering == clust
        wcec:::.wcov(
          x[idx, , drop = FALSE],
          w[idx]
        )
      })

      d <- d * Inf
      out$probability <- table(out$clustering) / n
      out$cost <- wcec:::.cost_gaussian(out$probability, n, out$params)
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
