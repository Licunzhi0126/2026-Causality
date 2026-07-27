#!/usr/bin/env Rscript

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) {
    dirname(normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE))
  } else {
    getwd()
  }
}

script <- file.path(get_script_dir(), "03_run_seurat_k40.R")
user_args <- commandArgs(trailingOnly = TRUE)
status <- system2(file.path(R.home("bin"), "Rscript"), c(shQuote(script), shQuote(user_args)))
quit(status = status)
