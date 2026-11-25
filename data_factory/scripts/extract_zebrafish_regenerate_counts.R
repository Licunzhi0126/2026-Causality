# Export the Spatial assay as little-endian sparse arrays for the H5AD converter.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript extract_zebrafish_regenerate_counts.R INPUT_RDS TEMP_DIR")
}

rds_path <- normalizePath(args[[1L]], mustWork = TRUE)
out_dir <- normalizePath(args[[2L]], mustWork = TRUE)
object <- readRDS(rds_path)
spatial <- attr(object, "assays")[["Spatial"]]
counts <- attr(spatial, "counts")
if (!inherits(counts, "dgCMatrix")) {
  stop("Expected a Spatial assay with dgCMatrix raw counts")
}

ids <- attr(counts, "Dimnames")
if (length(ids) != 2L || !identical(lengths(ids), attr(counts, "Dim"))) {
  stop("Raw counts must include gene and spot IDs")
}
writeLines(enc2utf8(ids[[1L]]), file.path(out_dir, "genes.txt"), useBytes = TRUE)
writeLines(enc2utf8(ids[[2L]]), file.path(out_dir, "spots.txt"), useBytes = TRUE)

# Columns are spots. The same sparse arrays describe spot-by-gene CSR in Python.
writeBin(attr(counts, "p"), file.path(out_dir, "indptr.i32"), size = 4L, endian = "little")
writeBin(attr(counts, "i"), file.path(out_dir, "indices.i32"), size = 4L, endian = "little")

write_integer_counts <- function(values, path) {
  connection <- file(path, open = "wb")
  on.exit(close(connection))
  chunk_size <- 1000000L
  for (start in seq.int(1L, length(values), by = chunk_size)) {
    block <- values[start:min(start + chunk_size - 1L, length(values))]
    if (any(!is.finite(block) | block <= 0 | block > .Machine$integer.max | block != trunc(block))) {
      stop("Stored counts must be positive 32-bit integers")
    }
    writeBin(as.integer(block), connection, size = 4L, endian = "little")
  }
}

values <- attr(counts, "x")
if (!length(values)) {
  stop("Spatial counts matrix is empty")
}
write_integer_counts(values, file.path(out_dir, "counts.i32"))
cat("Extracted", length(ids[[2L]]), "spots and", length(ids[[1L]]), "genes\n")
