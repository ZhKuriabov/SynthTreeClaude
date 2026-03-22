required_packages <- c(
  "devtools",
  "Rcpp",
  "RcppArmadillo",
  "RcppThread",
  "glmnet",
  "onehot",
  "visNetwork"
)

installed <- rownames(installed.packages())
missing <- setdiff(required_packages, installed)

if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

message("R dependencies installed.")
message("Run Python scripts from CleanCode/SynthTree so ../Rforestry resolves correctly.")
