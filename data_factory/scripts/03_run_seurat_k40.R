#!/usr/bin/env Rscript

## This script copies the core Seurat K-domain logic from:
## 2025 Causality/CCI_GRN_creater/seurat_20n40.r
## Changes are limited to argument parsing, recursive spot input discovery,
## and factory output naming.

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) {
    normalizePath(sub("^--file=", "", m[1]), mustWork = FALSE) |> dirname()
  } else {
    getwd()
  }
}
.script_dir <- get_script_dir()
.log_path <- file.path(.script_dir, sprintf("seurat_k40_factory_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
zz <- file(.log_path, open = "wt")
sink(zz, type = "output", split = TRUE)
.msg_split <- TRUE
tryCatch({
  sink(zz, type = "message", split = TRUE)
}, error = function(e) {
  .msg_split <<- FALSE
  sink(zz, type = "message")
})
on.exit({
  sink(type = "message")
  sink()
  close(zz)
}, add = TRUE)

options(warn = 1)
logi <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%F %T"), paste0(..., collapse = " ")))
logi("Log file: ", .log_path)

options(repos = c(CRAN = "https://cloud.r-project.org"))
ensure_pkg <- function(p, bioc = FALSE) {
  if (!requireNamespace(p, quietly = TRUE)) {
    logi("[Setup] Installing: ", p)
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(p, update = FALSE, ask = FALSE, quiet = TRUE)
    } else {
      install.packages(p, quiet = TRUE)
    }
  }
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  logi("[Setup] Installing BiocManager from CRAN")
  install.packages("BiocManager")
}
for (p in c("Seurat", "SeuratObject", "Matrix", "ggplot2", "grid")) ensure_pkg(p, bioc = FALSE)
for (p in c("zellkonverter", "SingleCellExperiment", "SummarizedExperiment",
            "S4Vectors", "IRanges", "BiocGenerics", "MatrixGenerics", "DelayedArray")) {
  ensure_pkg(p, bioc = TRUE)
}
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(ggplot2)
  library(grid)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
})

argv <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  key <- paste0("--", name)
  hit <- which(argv == key)
  if (!length(hit)) return(default)
  if (hit[1] == length(argv)) stop("Missing value for ", key)
  argv[[hit[1] + 1]]
}
arg_values <- function(name, default = character()) {
  key <- paste0("--", name)
  hit <- which(argv == key)
  if (!length(hit)) return(default)
  start <- hit[1] + 1
  if (start > length(argv)) stop("Missing value for ", key)
  end <- start
  while (end <= length(argv) && !startsWith(argv[[end]], "--")) {
    end <- end + 1
  }
  if (end == start) stop("Missing value for ", key)
  values <- argv[start:(end - 1)]
  if (!length(values)) default else values
}

SPOT_ROOT <- arg_value("spot-root", "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/spot")
OUT_DIR_BASE <- arg_value("output-root", "/home/jovyan/public/datasets/Mouse-embryo/E1S1_domain_factory/seurat_k40")
DOMAIN_K <- as.integer(arg_value("k", "40"))
PCA_DIMENSIONS <- as.integer(arg_value("pca-dimensions", "30"))
VARIABLE_FEATURES <- as.integer(arg_value("variable-features", "3000"))
SAMPLE_NAMES <- arg_values("sample-names", character())
MANIFEST_NAME <- arg_value("manifest-name", "domain_manifest_seurat_k40.csv")

`%||%` <- function(a, b) if (is.null(a)) b else a

pick_counts_assay <- function(sce) {
  an <- assayNames(sce)
  if ("counts" %in% an) return("counts")
  if ("X" %in% an) return("X")
  an[1]
}

get_spatial_coords <- function(sce) {
  if ("spatial" %in% reducedDimNames(sce)) {
    coords <- reducedDim(sce, "spatial")
    if (is.null(colnames(coords))) colnames(coords) <- c("x", "y")[seq_len(ncol(coords))]
    return(as.matrix(coords))
  }
  cn <- colnames(colData(sce))
  cand <- list(
    c("imagecol", "imagerow"),
    c("x", "y"),
    c("X", "Y"),
    c("pxl_col_in_fullres", "pxl_row_in_fullres")
  )
  for (pair in cand) {
    if (all(pair %in% cn)) {
      return(as.matrix(cbind(colData(sce)[[pair[1]]], colData(sce)[[pair[2]]])))
    }
  }
  NULL
}

normalize_annotation <- function(x) {
  x <- as.character(x)
  x[x == "Lung primordium"] <- "Lung"
  x
}

aggregate_by_cluster <- function(seu_counts, clusters, genes, spatial = NULL) {
  labels <- as.character(clusters)
  levels_keep <- unique(labels)
  j_idx <- match(labels, levels_keep)
  mm <- sparseMatrix(i = seq_along(j_idx), j = j_idx, x = 1,
                     dims = c(length(j_idx), length(levels_keep)))
  colnames(mm) <- levels_keep
  agg <- seu_counts %*% mm
  spot_count <- Matrix::colSums(mm)
  domain_ids <- sprintf("domain_%03d", seq_along(levels_keep))
  colnames(agg) <- domain_ids
  rownames(agg) <- genes

  dom_coords <- NULL
  if (!is.null(spatial) && nrow(spatial) == length(labels)) {
    sums <- t(mm) %*% spatial
    dom_coords <- sweep(as.matrix(sums), 1, as.numeric(spot_count), "/")
    colnames(dom_coords) <- colnames(spatial) %||% c("x", "y")[seq_len(ncol(dom_coords))]
    rownames(dom_coords) <- domain_ids
  }

  obs <- DataFrame(
    domain_id = domain_ids,
    domain_label = levels_keep,
    spot_count = as.numeric(spot_count),
    row.names = domain_ids
  )

  sce_dom <- SingleCellExperiment(assays = list(counts = agg), colData = obs)
  if (!is.null(dom_coords)) reducedDim(sce_dom, "spatial") <- dom_coords
  metadata(sce_dom)$domain_label <- levels_keep
  sce_dom
}

plot_spot_clusters <- function(spatial, clusters, out_png, title = "") {
  if (is.null(spatial)) {
    logi("[Visualize] no spatial coords; skip plot.")
    return(invisible())
  }
  df <- data.frame(x = spatial[, 1], y = spatial[, 2], cluster = as.factor(clusters))
  p <- ggplot(df, aes(x = x, y = y, color = cluster)) +
    geom_point(size = 0.8, stroke = 0, alpha = 0.9) +
    coord_equal() + scale_y_reverse() +
    labs(title = paste0(title, " spot domains"), color = "Domain") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "right", legend.key.height = unit(0.5, "cm"))
  ggsave(out_png, p, width = 8, height = 6, dpi = 200)
  logi("[Visualize] saved: ", out_png)
}

write_h5ad <- function(sce, path) {
  zellkonverter::writeH5AD(sce, path)
  logi("[Write] h5ad saved: ", path)
}

merge_to_k <- function(seu, K, dims_use) {
  emb <- Embeddings(seu, "pca")[, dims_use, drop = FALSE]
  centers <- t(sapply(levels(Idents(seu)), function(cl) {
    colMeans(emb[Idents(seu) == cl, , drop = FALSE])
  }))
  hc <- hclust(dist(centers), method = "average")
  merged <- cutree(hc, k = K)
  map <- setNames(merged, rownames(centers))
  new <- paste0("d", sprintf("%03d", map[as.character(Idents(seu))]))
  Idents(seu) <- factor(new, levels = unique(new))
  seu
}

assign_split_labels <- function(seu, cells, parent_label, sub_labels) {
  tmp <- as.character(Idents(seu))
  names(tmp) <- colnames(seu)
  idx <- match(cells, names(tmp))
  if (anyNA(idx)) {
    stop("Internal split assignment failed: subset cells are missing from the parent Seurat object.")
  }
  tmp[idx] <- paste0(parent_label, "_", as.character(sub_labels))
  Idents(seu) <- factor(tmp)
  seu
}

fallback_split_labels <- function(seu, cells, target_groups, dims_use) {
  target_groups <- min(as.integer(target_groups), length(cells))
  if (target_groups < 2 || length(cells) < 2) return(NULL)

  emb <- Embeddings(seu, "pca")[cells, dims_use, drop = FALSE]
  emb <- as.matrix(emb)
  unique_rows <- unique(as.data.frame(emb))
  if (nrow(unique_rows) >= target_groups) {
    km <- tryCatch(
      kmeans(emb, centers = target_groups, nstart = 20, iter.max = 100, algorithm = "Lloyd"),
      error = function(e) NULL
    )
    if (!is.null(km) && length(unique(km$cluster)) >= 2) {
      return(paste0("km", sprintf("%03d", km$cluster)))
    }
  }

  ord <- order(emb[, 1], seq_along(cells))
  labels <- integer(length(cells))
  labels[ord] <- cut(seq_along(cells), breaks = target_groups, labels = FALSE)
  paste0("ord", sprintf("%03d", labels))
}

ensure_exact_k_by_merge <- function(seu, K, dims_use = 1:30,
                                    algo = 3L,
                                    allow_singletons = FALSE,
                                    res_grid = c(seq(0.2, 2, 0.2), seq(2.5, 12, 0.5))) {
  stopifnot(K >= 2)
  set.seed(42)
  if (is.null(seu@graphs$SCT_snn) && is.null(seu@graphs$RNA_snn)) {
    seu <- FindNeighbors(seu, dims = dims_use, verbose = FALSE)
  }
  k_of_res <- sapply(res_grid, function(r) {
    obj <- FindClusters(seu, resolution = r, algorithm = algo,
                        group.singletons = allow_singletons, verbose = FALSE)
    length(unique(Idents(obj)))
  })
  if (all(k_of_res < K)) {
    res_star <- tail(res_grid, 1)
    logi(sprintf("[EnsureK] grid no K; use res=%.3f (k=%d)", res_star, tail(k_of_res, 1)))
  } else {
    res_star <- res_grid[which(k_of_res >= K)[1]]
    logi(sprintf("[EnsureK] res=%.3f (k=%d)", res_star, k_of_res[which(k_of_res >= K)[1]]))
  }
  seu <- FindClusters(seu, resolution = res_star, algorithm = algo,
                      group.singletons = allow_singletons, verbose = FALSE)
  cur_ids <- Idents(seu)
  cur_k <- nlevels(cur_ids)

  if (cur_k > K) {
    seu <- merge_to_k(seu, K, dims_use)
    logi(sprintf("[EnsureK] merge %d -> %d", cur_k, K))
    return(list(seu = seu, info = list(res_star = res_star, algo = algo, final_k = K)))
  }

  while (nlevels(Idents(seu)) < K) {
    cur_ids <- Idents(seu)
    prev_k <- nlevels(cur_ids)
    tab <- sort(table(cur_ids), decreasing = TRUE)
    split_done <- FALSE

    for (cl_big in names(tab)) {
      parent_cells <- colnames(seu)[as.character(cur_ids) == cl_big]
      if (length(parent_cells) < 2) next

      split_labels <- NULL
      split_method <- "seurat"
      seurat_split <- tryCatch({
        sub <- subset(seu, cells = parent_cells)
        sub <- FindNeighbors(sub, dims = dims_use, verbose = FALSE)
        out <- NULL
        for (r in c(seq(0.4, 6, by = 0.4), seq(7, 20, by = 1))) {
          sub <- FindClusters(sub, resolution = r, algorithm = algo,
                              group.singletons = allow_singletons, verbose = FALSE)
          if (nlevels(Idents(sub)) >= 2) {
            out <- list(cells = colnames(sub), labels = as.character(Idents(sub)))
            break
          }
        }
        out
      }, error = function(e) {
        logi("[EnsureK] Seurat split failed for ", cl_big, ": ", conditionMessage(e))
        NULL
      })
      if (!is.null(seurat_split)) {
        parent_cells <- seurat_split$cells
        split_labels <- seurat_split$labels
      }

      if (is.null(split_labels)) {
        split_method <- "pca_kmeans_fallback"
        target_groups <- min(length(parent_cells), K - prev_k + 1)
        split_labels <- fallback_split_labels(seu, parent_cells, target_groups, dims_use)
      }
      if (is.null(split_labels) || length(unique(split_labels)) < 2) next

      seu <- assign_split_labels(seu, parent_cells, cl_big, split_labels)
      if (nlevels(Idents(seu)) > prev_k) {
        logi(sprintf(
          "[EnsureK] split %s via %s: %d -> %d",
          cl_big, split_method, prev_k, nlevels(Idents(seu))
        ))
        split_done <- TRUE
        break
      }
    }

    if (!split_done) {
      stop(sprintf("Unable to split any cluster while enforcing K=%d; current_k=%d", K, prev_k))
    }

    cur_k <- nlevels(Idents(seu))
    if (cur_k >= K) {
      if (cur_k > K) {
        seu <- merge_to_k(seu, K, dims_use)
      }
      break
    }
  }

  list(seu = seu, info = list(res_star = res_star, algo = algo, final_k = nlevels(Idents(seu))))
}

parse_sample <- function(path) {
  stem <- sub("\\.h5ad$", "", basename(path))
  m <- regexec("^spot_([A-Za-z]+)_([0-9.]+)$", stem)
  parts <- regmatches(stem, m)[[1]]
  if (length(parts) == 3) return(list(organ = tolower(parts[2]), stage = parts[3]))
  parent <- basename(dirname(path))
  stage <- sub(".*?([0-9]+\\.[0-9]+).*", "\\1", stem)
  list(organ = tolower(parent), stage = stage)
}

run_one <- function(in_path, out_path, target_clusters) {
  logi("[Domain] Processing ", in_path, " -> ", out_path)
  set.seed(42)

  sce <- zellkonverter::readH5AD(in_path)
  counts_assay <- pick_counts_assay(sce)
  counts <- assay(sce, counts_assay)
  if (!inherits(counts, "dgCMatrix")) counts <- as(counts, "dgCMatrix")

  genes <- rownames(counts)
  cells <- colnames(counts)
  spatial <- get_spatial_coords(sce)

  anno <- NULL
  if ("annotation" %in% colnames(colData(sce))) {
    anno <- normalize_annotation(colData(sce)$annotation)
  } else {
    anno <- rep("", length(cells))
  }

  seu <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
  if (!is.null(spatial) && nrow(spatial) == ncol(seu)) {
    seu@meta.data$.__x__ <- spatial[, 1]
    seu@meta.data$.__y__ <- spatial[, 2]
  }
  seu@meta.data$annotation <- anno

  seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE)
  seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = VARIABLE_FEATURES, verbose = FALSE)
  hvgs <- VariableFeatures(seu)
  seu <- ScaleData(seu, features = hvgs, verbose = FALSE)
  seu <- RunPCA(seu, features = hvgs, npcs = PCA_DIMENSIONS, verbose = FALSE)
  seu <- FindNeighbors(seu, dims = 1:PCA_DIMENSIONS, verbose = FALSE)

  ensure_out <- ensure_exact_k_by_merge(
    seu, K = as.integer(target_clusters), dims_use = 1:PCA_DIMENSIONS,
    algo = 3L, allow_singletons = FALSE
  )
  seu <- ensure_out$seu
  clusters <- as.character(Idents(seu))

  dom_sce <- aggregate_by_cluster(
    seu_counts = GetAssayData(seu, slot = "counts"),
    clusters = clusters,
    genes = genes,
    spatial = spatial
  )

  label_levels <- metadata(dom_sce)$domain_label
  label2domain <- setNames(colnames(dom_sce), label_levels)
  spot_assign <- data.frame(
    spot_id = cells,
    domain_label = clusters,
    domain_id = unname(label2domain[clusters]),
    annotation = anno,
    stringsAsFactors = FALSE
  )
  if (!is.null(spatial) && nrow(spatial) == nrow(spot_assign)) {
    spot_assign$x <- spatial[, 1]
    spot_assign$y <- spatial[, 2]
  }

  map_csv <- sub("\\.h5ad$", "_spot_domain_map.csv", out_path)
  write.csv(spot_assign, map_csv, row.names = FALSE)
  logi("[Write] spot-domain map: ", map_csv)

  comp <- as.data.frame.matrix(table(spot_assign$domain_id, spot_assign$annotation))
  comp$domain_id <- rownames(comp)
  comp <- comp[, c("domain_id", setdiff(colnames(comp), "domain_id")), drop = FALSE]
  comp_csv <- sub("\\.h5ad$", "_domain_organ_counts.csv", out_path)
  write.csv(comp, comp_csv, row.names = FALSE)
  logi("[Write] domain organ counts: ", comp_csv)

  colData(sce)$domain_label <- clusters
  colData(sce)$domain_id <- unname(label2domain[clusters])
  spot_h5ad <- sub("\\.h5ad$", "_spots_with_domain.h5ad", out_path)
  write_h5ad(sce, spot_h5ad)

  png_path <- sub("\\.h5ad$", "_spot_domains.png", out_path)
  plot_spot_clusters(spatial, clusters, png_path, title = basename(in_path))
  png_path2 <- sub("\\.h5ad$", "_spot_organs.png", out_path)
  plot_spot_clusters(spatial, anno, png_path2, title = paste0(basename(in_path), " organs"))

  write_h5ad(dom_sce, out_path)
  list(file = out_path, n_spots = ncol(seu), n_domains = ncol(dom_sce), target = target_clusters)
}

main <- function() {
  logi("R.version: ", as.character(getRversion()))
  logi("[Config] spot-root=", SPOT_ROOT)
  logi("[Config] output-root=", OUT_DIR_BASE)
  logi("[Config] k=", DOMAIN_K)
  if (length(SAMPLE_NAMES)) logi("[Config] sample-names=", paste(SAMPLE_NAMES, collapse = ","))
  logi("[Config] manifest-name=", MANIFEST_NAME)

  files <- list.files(SPOT_ROOT, pattern = "\\.h5ad$", recursive = TRUE, full.names = TRUE)
  files <- files[!grepl("_spots_with_domain\\.h5ad$|_COMMOT\\.h5ad$", files)]
  if (length(SAMPLE_NAMES)) {
    stems <- sub("\\.h5ad$", "", basename(files))
    files <- files[stems %in% SAMPLE_NAMES]
  }
  if (!length(files)) stop("No spot h5ad files found under: ", SPOT_ROOT)

  rows <- list()
  for (in_path in sort(files)) {
    sample <- parse_sample(in_path)
    out_dir <- file.path(OUT_DIR_BASE, sample$organ)
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    out_path <- file.path(out_dir, sprintf("seurat_%s_%s.h5ad", sample$organ, sample$stage))
    row <- data.frame(
      input_file = in_path,
      output_file = out_path,
      organ = sample$organ,
      stage = sample$stage,
      k = DOMAIN_K,
      n_spots = NA_integer_,
      n_domains = NA_integer_,
      status = "planned",
      reason = "",
      stringsAsFactors = FALSE
    )
    row <- tryCatch({
      result <- run_one(in_path, out_path, DOMAIN_K)
      row$status <- "written"
      row$n_spots <- result$n_spots
      row$n_domains <- result$n_domains
      row
    }, error = function(e) {
      row$status <- "error"
      row$reason <- paste0(class(e)[1], ": ", conditionMessage(e))
      logi("[Error] ", in_path, ": ", row$reason)
      row
    })
    rows[[length(rows) + 1]] <- row
  }
  manifest_dir <- file.path(dirname(OUT_DIR_BASE), "manifests")
  dir.create(manifest_dir, showWarnings = FALSE, recursive = TRUE)
  manifest <- file.path(manifest_dir, MANIFEST_NAME)
  write.csv(do.call(rbind, rows), manifest, row.names = FALSE)
  logi("[Write] manifest: ", manifest)
  logi("All done.")
}

if (sys.nframe() == 0L) main()
