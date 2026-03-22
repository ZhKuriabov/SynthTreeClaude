#' @include forestry.R

# ---plots a tree ----------------------------------------------------------
#' plot
#' @name plot-forestry
#' @title visualize a tree
#' @rdname plot-forestry
#' @description plots a tree in the forest.
#' @param x A forestry x.
#' @param tree.id Specifies the tree number that should be visulaized.
#' @param print.meta_dta Should the data for the plot be printed?
#' @param beta.char.len The length of the beta values in leaf node
#' representation.
#' @param ... additional arguments that are not used.
#' @import glmnet
#' @examples
#' set.seed(292315)
#' rf <- forestry(x = iris[,-1],
#'                y = iris[, 1])
#'
#' plot(x = rf)
#' plot(x = rf, tree.id = 2)
#' plot(x = rf, tree.id = 500)
#'
#' ridge_rf <- forestry(
#'   x = iris[,-1],
#'   y = iris[, 1],
#'   replace = FALSE,
#'   nodesizeStrictSpl = 10,
#'   mtry = 4,
#'   ntree = 1000,
#'   minSplitGain = .004,
#'   linear = TRUE,
#'   overfitPenalty = 1.65,
#'   linFeats = 1:2)
#'
#' plot(x = ridge_rf)
#' plot(x = ridge_rf, tree.id = 2)
#' plot(x = ridge_rf, tree.id = 1000)
#'
#' @export
#' @import visNetwork
plot.forestry <- function(x, tree.id = 1, print.meta_dta = FALSE,
                          beta.char.len = 30, ...) {
  if (x@ntree < tree.id | 1 > tree.id) {
    stop("tree.id is too large or too small.")
  }

  forestry_tree <- make_savable(x)

  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val

  # get info for the first node ------------------------------------------------
  root_is_leaf <- split_feat[1] < 0
  node_info <- data.frame(
    node_id = 1,
    is_leaf = root_is_leaf,
    parent = NA,
    left_child = ifelse(root_is_leaf, NA, 2),
    right_child = NA,
    split_feat = ifelse(root_is_leaf, NA, split_feat[1]),
    split_val = ifelse(root_is_leaf, NA, split_val[1]),
    num_splitting = ifelse(root_is_leaf, -split_feat[2], 0),
    num_averaging = ifelse(root_is_leaf, -split_feat[1], 0),
    level = 1)
  if (root_is_leaf) {
    split_feat <- split_feat[-(1:2)]
    split_val <- split_val[-1]
  } else {
    split_feat <- split_feat[-1]
    split_val <- split_val[-1]
  }

  # loop through split feat to get all the information -------------------------
  while (length(split_feat) != 0) {
    if (!node_info$is_leaf[nrow(node_info)]) {
      # previous node is not leaf => left child of previous node
      parent <- nrow(node_info)
      node_info$left_child[parent] <- nrow(node_info) + 1
    } else {
      # previous node is leaf => right child of last unfilled right
      parent <-
        max(node_info$node_id[!node_info$is_leaf &
                                is.na(node_info$right_child)])
      node_info$right_child[parent] <- nrow(node_info) + 1
    }
    if (split_feat[1] > 0) {
      # it is not a leaf
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = FALSE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = split_feat[1],
          split_val = split_val[1],
          num_splitting = 0,
          num_averaging = 0,
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-1]
      split_val <- split_val[-1]
    } else {
      #split_feat[1] < 0
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = TRUE,
          parent = parent,
          left_child = NA,
          right_child = NA,
          split_feat = NA,
          split_val = NA,
          num_splitting = -split_feat[2],
          num_averaging = -split_feat[1],
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-(1:2)]
      split_val <- split_val[-1]
    }
  }

  for (i in nrow(node_info):1) {
    if (node_info$num_splitting[i] != 0) {
      node_info$num_splitting[node_info$parent[i]] <-
        node_info$num_splitting[node_info$parent[i]] + node_info$num_splitting[i]

      node_info$num_averaging[node_info$parent[i]] <-
        node_info$num_averaging[node_info$parent[i]] + node_info$num_averaging[i]
    }
  }


  # Save more information about the splits -------------------------------------
  feat_names <- colnames(forestry_tree@processed_dta$processed_x)
  node_info$feat_nm <- NA
  node_info$is_cat_feat <- NA
  node_info$cat_split_value <- NA
  if (sum(!is.na(node_info$split_feat)) != 0) {
    node_info$feat_nm <- feat_names[node_info$split_feat]
    cat_feats <- names(forestry_tree@processed_dta$categoricalFeatureCols_cpp)
    node_info$is_cat_feat <- node_info$feat_nm %in% cat_feats
    node_info$cat_split_value <- as.character(node_info$split_val)

    cat_feat_map <- forestry_tree@categoricalFeatureMapping
    if (length(cat_feat_map) > 0) {
      for (i in 1:length(cat_feat_map)) {
        # i = 1
        nodes_with_this_split <-
          node_info$split_feat == cat_feat_map[[i]]$categoricalFeatureCol &
          (!is.na(node_info$split_feat))

        node_info$cat_split_value[nodes_with_this_split] <-
          as.character(cat_feat_map[[i]]$uniqueFeatureValues[
            node_info$split_val[nodes_with_this_split]])
      }
    }
  }


  # get the observation ids per leaf -------------------------------------------
  ave_idx <- forestry_tree@R_forest[[tree.id]]$leafAveidx

  leaf_idx <- list()
  for (leaf_id in node_info$node_id[node_info$is_leaf]) {
    # leaf_id <- 4

    these_idces <- 1:(node_info$num_averaging[node_info$node_id == leaf_id])

    leaf_idx[[leaf_id]] <- ave_idx[these_idces]
    ave_idx <- ave_idx[-these_idces]
  }

  # Prepare data for VisNetwork ------------------------------------------------

  nodes <- data.frame(
    id = node_info$node_id,
    shape = ifelse(node_info$is_leaf,
                   "box", "circle"),
    label = ifelse(
      node_info$is_leaf,
      paste0(node_info$num_averaging, " Obs"),
      paste0(feat_names[node_info$split_feat])
    ),
    level = node_info$level
  )

  edges <- data.frame(
    from = node_info$parent,
    to = node_info$node_id,
    smooth = list(
      enabled = TRUE,
      type = "cubicBezier",
      roundness = .5
    )
  )
  edges <- edges[-1, ]


  edges$label =
    ifelse(
      floor(node_info$split_val[edges$from]) == node_info$split_val[edges$from],
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" = ", node_info$cat_split_value[edges$from]),
        paste0(" != ", node_info$cat_split_value[edges$from])
      ),
      ifelse(
        node_info$left_child[edges$from] == edges$to,
        paste0(" < ", round(node_info$split_val[edges$from], digits = 2)),
        paste0(" >= ", round(node_info$split_val[edges$from], digits = 2))
      )
    )

  edges$width = node_info$num_averaging[edges$to] /
    (node_info$num_averaging[1] / 4)

  # collect data for leaves ----------------------------------------------------
  nodes$label <- as.character(nodes$label)
  nodes$title <- as.character(nodes$label)


  dta_x <- forestry_tree@processed_dta$processed_x
  dta_y <- forestry_tree@processed_dta$y


  if (forestry_tree@linear) {
    # ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      # leaf_id = 5
      ###
      this_ds <- dta_x[leaf_idx[[leaf_id]],
                       forestry_tree@linFeats + 1]
      encoder <- onehot::onehot(this_ds)
      remat <- predict(encoder, this_ds)
      ###
      y_leaf <- dta_y[leaf_idx[[leaf_id]]]
      plm <- glmnet::glmnet(x = remat,
                            y = y_leaf,
                            lambda = forestry_tree@overfitPenalty * sd(y_leaf)/nrow(remat),
                            alpha	= 0)

      plm_pred <- predict(plm, type = "coef")
      plm_pred_names <- c("interc", colnames(remat))

      return_char <- character()
      for (i in 1:length(plm_pred)) {
        return_char <- paste0(return_char,
                              substr(plm_pred_names[i], 1, beta.char.len), " ",
                              round(plm_pred[i], 2), "<br>")
      }
      nodes$title[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "<br> R2 = ",
                                     plm$dev.ratio,
                                     "<br>========<br>",
                                     return_char)
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id],
                                     "\n R2 = ",
                                     round(plm$dev.ratio, 3),
                                     "\n=======\nm = ",
                                     round(mean(dta_y[leaf_idx[[leaf_id]]]), 5))
    }
  } else {
    # not ridge forest
    for (leaf_id in node_info$node_id[node_info$is_leaf]) {
      nodes$label[leaf_id] <- paste0(nodes$label[leaf_id], "\n=======\nm = ",
                                     round(mean(dta_y[leaf_idx[[leaf_id]]]), 5))
    }
  }

  # defines a colors -----------------------------------------------------------
  split_vals <- node_info$split_feat
  split_vals <- ifelse(is.na(split_vals), 0, split_vals)
  split_vals <- factor(split_vals)
  color_code <- grDevices::terrain.colors(n = length(feat_names) + 1,
                                          alpha = .7)
  names(color_code) <- as.character(0:(length(feat_names)))
  nodes$color <- color_code[split_vals]

  # Plot the actual node -------------------------------------------------------
  (p1 <-
    visNetwork(
      nodes,
      edges,
      width = "100%",
      height = "800px",
      main = paste("Tree", tree.id)
    ) %>%
    visEdges(arrows = "to") %>%
    visHierarchicalLayout() %>% visExport(type = "pdf", name = "ridge_tree"))

  if (print.meta_dta) print(node_info)
  return(p1)
}



get_leaf_info <- function(model, tree.id = 1) {
  if (!"forestry" %in% class(model)) {
    stop("Input must be a trained `forestry` model, not the constructor function.")
  }

  forestry_tree <- make_savable(model)

  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val

  root_is_leaf <- split_feat[1] < 0
  node_info <- data.frame(
    node_id = 1,
    is_leaf = root_is_leaf,
    parent = NA,
    left_child = ifelse(root_is_leaf, NA, 2),
    right_child = NA,
    split_feat = ifelse(root_is_leaf, NA, split_feat[1]),
    split_val = ifelse(root_is_leaf, NA, split_val[1]),
    num_splitting = ifelse(root_is_leaf, -split_feat[2], 0),
    num_averaging = ifelse(root_is_leaf, -split_feat[1], 0),
    level = 1
  )

  if (root_is_leaf) {
    split_feat <- split_feat[-(1:2)]
    split_val <- split_val[-1]
  } else {
    split_feat <- split_feat[-1]
    split_val <- split_val[-1]
  }

  while (length(split_feat) != 0) {
    if (!node_info$is_leaf[nrow(node_info)]) {
      parent <- nrow(node_info)
      node_info$left_child[parent] <- nrow(node_info) + 1
    } else {
      parent <- max(node_info$node_id[!node_info$is_leaf & is.na(node_info$right_child)])
      node_info$right_child[parent] <- nrow(node_info) + 1
    }

    if (split_feat[1] > 0) {
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = FALSE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = split_feat[1],
          split_val = split_val[1],
          num_splitting = 0,
          num_averaging = 0,
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-1]
      split_val <- split_val[-1]
    } else {
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = TRUE,
          parent = parent,
          left_child = NA,
          right_child = NA,
          split_feat = NA,
          split_val = NA,
          num_splitting = -split_feat[2],
          num_averaging = -split_feat[1],
          level = node_info$level[parent] + 1
        )
      )
      split_feat <- split_feat[-(1:2)]
      split_val <- split_val[-1]
    }
  }

  for (i in nrow(node_info):1) {
    if (node_info$num_splitting[i] != 0) {
      parent <- node_info$parent[i]
      node_info$num_splitting[parent] <- node_info$num_splitting[parent] + node_info$num_splitting[i]
      node_info$num_averaging[parent] <- node_info$num_averaging[parent] + node_info$num_averaging[i]
    }
  }

  leaf_info <- node_info[node_info$is_leaf, c("node_id", "level", "num_averaging")]
  colnames(leaf_info) <- c("leaf_id", "depth", "num_observations")

  # === Compute interpretability metric ===
  N <- sum(leaf_info$num_observations)
  raw_interpretability <- sum(leaf_info$num_observations * leaf_info$depth) / N

#   cat("Leaf summary:\n")
#   print(leaf_info)
  cat(sprintf("\nRaw Interpretability Score: %.4f\n", raw_interpretability))

#   return(list(leaf_info = leaf_info, interpretability_raw = raw_interpretability))
  return(list(leaf_info = leaf_info, interpretability_raw = raw_interpretability))
}





#' Get number of leaves and max depth of a tree
#' @param model A forestry object
#' @param tree.id Integer specifying the tree index
#' @return Named vector with leaf count and maximum depth
#' @export
get_tree_leaf_count_and_depth <- function(model, tree.id = 1) {
  # Utility to safely extract first element or return NA
  get_safe <- function(vec) {
    if (length(vec) == 0) return(NA)
    return(vec[1])
  }

  forestry_tree <- make_savable(model)
  split_feat <- forestry_tree@R_forest[[tree.id]]$var_id
  split_val <- forestry_tree@R_forest[[tree.id]]$split_val
  node_values <- forestry_tree@R_forest[[tree.id]]$weights
  avg_counts <- forestry_tree@R_forest[[tree.id]]$average_count
  split_counts <- forestry_tree@R_forest[[tree.id]]$split_count

  root_is_leaf <- split_feat[1] < 0
  first_val <- split_val[1]
  node_info <- data.frame(
    node_id = 1,
    is_leaf = root_is_leaf,
    parent = NA,
    left_child = ifelse(root_is_leaf, NA, 2),
    right_child = NA,
    split_feat = ifelse(root_is_leaf, NA, split_feat[1]),
    split_val = ifelse(root_is_leaf, NA, first_val),
    num_splitting = ifelse(root_is_leaf, get_safe(split_counts), 0),
    num_averaging = ifelse(root_is_leaf, get_safe(avg_counts), 0),
    value = 0.0,
    level = 1
  )

  split_feat <- split_feat[-1]
  split_val <- split_val[-1]
  node_values <- node_values[-1]
  avg_counts <- avg_counts[-1]
  split_counts <- split_counts[-1]

  while (length(split_feat) != 0) {
    if (!node_info$is_leaf[nrow(node_info)]) {
      parent <- nrow(node_info)
      node_info$left_child[parent] <- nrow(node_info) + 1
    } else {
      parent <- max(node_info$node_id[!node_info$is_leaf & is.na(node_info$right_child)])
      node_info$right_child[parent] <- nrow(node_info) + 1
    }

    if (split_feat[1] > 0) {
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = FALSE,
          parent = parent,
          left_child = nrow(node_info) + 2,
          right_child = NA,
          split_feat = split_feat[1],
          split_val = split_val[1],
          num_splitting = 0,
          num_averaging = 0,
          value = 0.0,
          level = node_info$level[parent] + 1
        )
      )
    } else {
      node_info <- rbind(
        node_info,
        data.frame(
          node_id = nrow(node_info) + 1,
          is_leaf = TRUE,
          parent = parent,
          left_child = NA,
          right_child = NA,
          split_feat = NA,
          split_val = NA,
          num_splitting = get_safe(split_counts),
          num_averaging = get_safe(avg_counts),
          value = get_safe(node_values),
          level = node_info$level[parent] + 1
        )
      )
    }

    split_feat <- if (length(split_feat) > 1) split_feat[-1] else numeric()
    split_val <- if (length(split_val) > 1) split_val[-1] else numeric()
    node_values <- if (length(node_values) > 1) node_values[-1] else numeric()
    avg_counts <- if (length(avg_counts) > 1) avg_counts[-1] else numeric()
    split_counts <- if (length(split_counts) > 1) split_counts[-1] else numeric()
  }

  leaf_count <- sum(node_info$is_leaf)
  max_depth <- max(node_info$level)

  return(c(leaf_count = leaf_count, max_depth = max_depth))
}

get_leaf_assignments_and_depth <- function(model, X_test) {
  tree <- model@R_forest[[1]]

  split_feat <- tree$var_id
  split_val <- tree$split_val
  left_idx <- tree$leftChild
  right_idx <- tree$rightChild
  n_nodes <- length(split_feat)

  if (n_nodes == 0 || is.null(split_feat)) {
    return(data.frame(
      leaf_node = 1L,
      count = nrow(X_test),
      depth = 0L
    ))
  }

  # Compute depth of each node
  depth <- rep(NA_integer_, n_nodes)
  depth[1] <- 0
  for (i in seq_len(n_nodes)) {
    if (!is.na(left_idx[i]) && left_idx[i] + 1 <= n_nodes && !is.na(depth[i])) {
      depth[left_idx[i] + 1] <- depth[i] + 1
    }
    if (!is.na(right_idx[i]) && right_idx[i] + 1 <= n_nodes && !is.na(depth[i])) {
      depth[right_idx[i] + 1] <- depth[i] + 1
    }
  }

  # Defensive leaf assignment
  assign_leaf <- function(x) {
    node <- 0
    repeat {
      idx <- node + 1
      if (idx > length(split_feat) || is.na(split_feat[idx])) {
        return(idx)  # node index out of bounds — fallback
      }
      if (split_feat[idx] < 0) {
        return(idx)  # leaf node reached
      }
      f <- split_feat[idx] + 1
      v <- split_val[idx]
      if (f > length(x)) {
        stop(sprintf("Feature index %d out of bounds for test point.", f))
      }
      node <- if (x[f] <= v) left_idx[idx] else right_idx[idx]
      if (is.na(node)) return(idx)  # hit dead end
    }
  }

  # Apply to all test points
  leaf_assignments <- apply(X_test, 1, assign_leaf)

  # Create summary table
  assignment_table <- as.data.frame(table(leaf_assignments), stringsAsFactors = FALSE)
  colnames(assignment_table) <- c("leaf_node", "count")
  assignment_table$leaf_node <- as.integer(assignment_table$leaf_node)
  assignment_table$depth <- depth[assignment_table$leaf_node]

  return(assignment_table)
}