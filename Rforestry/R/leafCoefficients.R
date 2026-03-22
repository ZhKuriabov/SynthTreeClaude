#' Extract ridge‐regression coefficients for every leaf
#'
#' @param object   A fitted \code{forestry} model with \code{linear = TRUE}.
#' @param x_train  The training feature matrix or data.frame used to fit
#'                 \code{object}.  If \code{NULL}, the function will try to
#'                 read it from \code{object@processed_dta}.
#' @param y_train  The training response vector.  If \code{NULL}, the function
#'                 will try to read it from \code{object@y}.
#' @param lambda   Optional ridge penalty.  The default (\code{NULL}) uses
#'                 the value stored in the model object.
#'
#' @return A list with
#'   \describe{
#'     \item{coefficients}{A \code{data.frame}; one row per leaf, columns are
#'       \code{(Intercept)} followed by feature names.}
#'     \item{leafSizes}{The number of training samples in every leaf.}
#'   }
#' @examples
#' \dontrun{
#'   cal  <- rpart::fetch_california_housing()
#'   fit  <- forestry(cal$data, cal$target, ntree = 1, linear = TRUE)
#'   leafCoefficients(fit)
#' }
#' @export
leafCoefficients <- function(object,
                             x_train = NULL,
                             y_train = NULL,
                             lambda  = NULL) {

  ## ------------------------------------------------------------
  ## 0.  Basic checks
  stopifnot(inherits(object, "forestry"))
  if (isFALSE(object@linear)) {
    stop("Model was not trained with linear = TRUE.")
  }

  ## ------------------------------------------------------------
  ## 1.  Get training data --------------------------------------
  if (is.null(x_train)) x_train <- object@processed_dta
  if (is.null(y_train)) y_train <- object@y
  x_train <- as.matrix(x_train)
  y_train <- as.numeric(y_train)

  ## ------------------------------------------------------------
  ## 2.  Ask the model for its adaptive NN weights --------------
  pred <- predict(object, x_train,
                  aggregation = "weightMatrix")
  W    <- pred$weightMatrix            # n × n  dense matrix
  nodes<- pred$terminalNodes           # length n

  ## ------------------------------------------------------------
  ## 3.  Ridge coefficients per leaf ----------------------------
  if (is.null(lambda)) lambda <- object@linearLambda
  p        <- ncol(x_train)
  I_p      <- diag(p + 1)              # +1 for intercept
  I_p[1,1] <- 0                        # do not penalise intercept

  ##   helper to build [1, x] row with intercept
  add_1 <- function(X) cbind(1, X)

  leaves     <- split(seq_len(nrow(x_train)), nodes)
  coef_list  <- lapply(leaves, function(idx) {
    Xi  <- add_1(x_train[idx, , drop = FALSE])
    Wi  <- diag(W[idx[1], ])           # same weights for all idx in leaf
    XtWX <- t(Xi) %*% Wi %*% Xi + lambda * I_p
    XtWy <- t(Xi) %*% Wi %*% y_train[idx]
    solve(XtWX, XtWy)                  # (p+1) × 1
  })

  coefs_df <- do.call(rbind, coef_list)
  colnames(coefs_df) <- c("(Intercept)", colnames(x_train))

  list(
    coefficients = as.data.frame(coefs_df),
    leafSizes    = vapply(leaves, length, integer(1))
  )
}