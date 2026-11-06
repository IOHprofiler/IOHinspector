#!/usr/bin/env Rscript
# Golden reference outputs from IOHanalyzer (WP I03).
#
#   Rscript generate.R <fixtures dir> <output dir>
#
# Runs IOHanalyzer's own R functions on every fixture and writes one Parquet file per metric and
# fixture to <output dir>/<fixture>/<metric>.parquet, plus <output dir>/versions.json.
# Everything is deterministic: random steps use set.seed, Parquet is written with zstd, and
# nothing records a time. The metric list follows the parity matrix of docs/definitions/README.md.

suppressPackageStartupMessages({
  library(IOHanalyzer)
  library(data.table)
  library(arrow)
  library(jsonlite)
})
options(warn = -1)

args <- commandArgs(trailingOnly = TRUE)
fix_dir <- if (length(args) >= 1) args[[1]] else "/work/fixtures"
out_dir <- if (length(args) >= 2) args[[2]] else "/work/iohanalyzer"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

SEED <- 20240423L
index <- list()      # file -> description of the call
skipped <- list()    # file -> reason

# ---- helpers -----------------------------------------------------------------------------------
to_frame <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.matrix(x) || is.array(x)) {
    x <- as.data.frame(as.table(x), stringsAsFactors = FALSE) |> setDT()
  } else {
    x <- as.data.table(x)
  }
  for (nm in names(x)) {
    col <- x[[nm]]
    if (is.factor(col)) set(x, j = nm, value = as.character(col))
    else if (is.list(col)) set(x, j = nm, value = vapply(col, function(v) paste(format(v, digits = 17), collapse = ","), ""))
  }
  as.data.frame(x)
}

emit <- function(set, name, value, call) {
  key <- paste0(set, "/", name, ".parquet")
  df <- to_frame(value)
  if (is.null(df) || nrow(df) == 0) {
    skipped[[key]] <<- "empty result"
    return(invisible(NULL))
  }
  d <- file.path(out_dir, set)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  write_parquet(df, file.path(d, paste0(name, ".parquet")), compression = "zstd", compression_level = 9,
                use_dictionary = FALSE, version = "2.6")
  index[[key]] <<- call
  invisible(NULL)
}

# Run a metric; an error becomes a recorded skip, never a silent success.
attempt <- function(set, name, call, expr) {
  res <- tryCatch(expr, error = function(e) e)
  if (inherits(res, "error")) {
    skipped[[paste0(set, "/", name, ".parquet")]] <<- paste("error:", conditionMessage(res))
    return(invisible(NULL))
  }
  emit(set, name, res, call)
}

tag <- function(dt, ...) {  # prepend identifying columns
  dt <- as.data.table(dt)
  extra <- list(...)
  for (nm in rev(names(extra))) set(dt, j = nm, value = extra[[nm]])
  setcolorder(dt, c(names(extra), setdiff(names(dt), names(extra))))
  dt
}

# a matrix with dimnames as a long table
long <- function(m, row = "row", col = "col", value = "value") {
  r <- rownames(m); if (is.null(r)) r <- seq_len(nrow(m))
  cn <- colnames(m); if (is.null(cn)) cn <- seq_len(ncol(m))
  z <- data.table(r = rep(as.character(r), times = ncol(m)), c = rep(as.character(cn), each = nrow(m)), v = as.vector(m))
  setnames(z, c(row, col, value))
  z[]
}

# one call per target; targets IOHanalyzer cannot handle are recorded in `skipped`, not hidden
per_target <- function(set, name, f, d, targets, fun) {
  failed <- c()
  out <- bind(lapply(targets, function(t) {
    r <- tryCatch(fun(t), error = function(e) NULL)
    if (is.null(r)) { failed <<- c(failed, format(t, digits = 17)); return(NULL) }
    tag(r, pair_f = f, pair_d = d)
  }))
  if (length(failed) > 0) {
    key <- sprintf("%s/%s.parquet#f%s_d%s", set, name, f, d)
    skipped[[key]] <<- paste("IOHanalyzer errors (subscript out of bounds) at targets", paste(failed, collapse = ", "))
  }
  out
}

bind <- function(lst) rbindlist(lst, fill = TRUE, use.names = TRUE)

# ---- per fixture -------------------------------------------------------------------------------
run_fixture <- function(set, dsl, par_pen = c(1, 10)) {
  ids <- get_id(dsl)
  pairs <- unique(rbindlist(lapply(dsl, function(ds) data.table(f = attr(ds, "funcId"), d = attr(ds, "DIM")))))
  setorder(pairs, f, d)
  maximize <- isTRUE(attr(dsl, "maximization"))
  emit(set, "meta", data.table(
    ID = ids, maximization = maximize,
    n_datasets = length(dsl)), "DataSetList(<fixture>): ids and direction")

  # target and budget grids, per (function, dimension) pair
  for (type in c("log-linear", "linear", "bbob")) {
    nm <- paste0("targets_", gsub("-", "_", type))
    attempt(set, nm, sprintf("get_ECDF_targets(dsl, type='%s', number_targets=%d)", type, ifelse(type == "bbob", 51, 10)),
            get_ECDF_targets(dsl, type = type, number_targets = 10))
  }
  per_pair <- function(fun) bind(lapply(seq_len(nrow(pairs)), function(i) fun(pairs$f[[i]], pairs$d[[i]],
                                  subset(dsl, funcId == pairs$f[[i]] & DIM == pairs$d[[i]]))))

  attempt(set, "budgets_log", "seq_RT(get_runtimes(sub), length.out=10, scale='log')",
          per_pair(function(f, d, sub) {
            rt <- get_runtimes(sub)
            tag(data.table(budget = seq_RT(rt, length.out = 10, scale = "log")), funcId = f, DIM = d)
          }))
  attempt(set, "budgets_linear", "seq_RT(get_runtimes(sub), length.out=10, scale='linear')",
          per_pair(function(f, d, sub) {
            rt <- get_runtimes(sub)
            tag(data.table(budget = seq_RT(rt, length.out = 10, scale = "linear")), funcId = f, DIM = d)
          }))
  attempt(set, "target_dt_by_FV", "get_target_dt(dsl, 'by_FV')", get_target_dt(dsl, "by_FV"))
  attempt(set, "target_dt_by_RT", "get_target_dt(dsl, 'by_RT')", get_target_dt(dsl, "by_RT"))
  attempt(set, "funvals", "get_funvals(sub)", per_pair(function(f, d, sub)
    tag(data.table(f = get_funvals(sub)), funcId = f, DIM = d)))
  attempt(set, "runtimes", "get_runtimes(sub)", per_pair(function(f, d, sub)
    tag(data.table(rt = get_runtimes(sub)), funcId = f, DIM = d)))

  tgt <- function(f, d, sub, k = 5) {
    get_ECDF_targets(sub, "log-linear", k)$target
  }
  bud <- function(f, d, sub, k = 5) unique(seq_RT(get_runtimes(sub), length.out = k, scale = "log"))

  # runtime and value samples
  attempt(set, "RT_sample", "get_RT_sample(sub, ftarget=<5 log targets>, output='long')",
          per_pair(function(f, d, sub) tag(get_RT_sample(sub, tgt(f, d, sub), output = "long"), pair_f = f, pair_d = d)))
  attempt(set, "FV_sample", "get_FV_sample(sub, runtime=<5 log budgets>, output='long')",
          per_pair(function(f, d, sub) tag(get_FV_sample(sub, bud(f, d, sub), output = "long"), pair_f = f, pair_d = d)))
  attempt(set, "maxRT", "get_maxRT(dsl, output='long')", get_maxRT(dsl, output = "long"))

  # ERT, runtime summary (including PAR-k), fixed-budget summary, overviews
  for (pen in par_pen) {
    options(IOHanalyzer.PAR_penalty = pen)
    attempt(set, sprintf("RT_summary_par%d", pen),
            sprintf("get_RT_summary(sub, ftarget) with option IOHanalyzer.PAR_penalty=%d", pen),
            per_pair(function(f, d, sub) tag(get_RT_summary(sub, tgt(f, d, sub)), pair_f = f, pair_d = d)))
    attempt(set, sprintf("RT_overview_par%d", pen),
            sprintf("get_RT_overview(sub) with option IOHanalyzer.PAR_penalty=%d", pen),
            per_pair(function(f, d, sub) tag(get_RT_overview(sub), pair_f = f, pair_d = d)))
    attempt(set, sprintf("overview_par%d", pen),
            sprintf("get_overview(sub) with option IOHanalyzer.PAR_penalty=%d", pen),
            per_pair(function(f, d, sub) tag(get_overview(sub), pair_f = f, pair_d = d)))
  }
  options(IOHanalyzer.PAR_penalty = 1)
  attempt(set, "ERT", "get_ERT(sub, ftarget=<5 log targets>)",
          per_pair(function(f, d, sub) {
            tg <- tgt(f, d, sub)
            bind(lapply(tg, function(t) tag(get_ERT(sub, t), pair_f = f, pair_d = d, target = t)))
          }))
  attempt(set, "FV_summary", "get_FV_summary(sub, runtime=<5 log budgets>)",
          per_pair(function(f, d, sub) tag(get_FV_summary(sub, bud(f, d, sub)), pair_f = f, pair_d = d)))
  attempt(set, "FV_summary_geom", "get_FV_summary(sub, runtime=<5 log budgets>, include_geom=TRUE)",
          per_pair(function(f, d, sub) tag(get_FV_summary(sub, bud(f, d, sub), include_geom = TRUE), pair_f = f, pair_d = d)))
  attempt(set, "FV_overview", "get_FV_overview(sub)",
          per_pair(function(f, d, sub) tag(get_FV_overview(sub), pair_f = f, pair_d = d)))

  # ECDF and AUC
  attempt(set, "ECDF_by_RT", "generate_data.ECDF(sub, targets=<5 log targets>, which='by_RT')",
          per_pair(function(f, d, sub) tag(generate_data.ECDF(sub, tgt(f, d, sub), which = "by_RT"), pair_f = f, pair_d = d)))
  attempt(set, "ECDF_by_FV", "generate_data.ECDF(sub, targets=<5 log budgets>, which='by_FV')",
          per_pair(function(f, d, sub) tag(generate_data.ECDF(sub, bud(f, d, sub), which = "by_FV"), pair_f = f, pair_d = d)))
  attempt(set, "AUC_by_RT", "generate_data.AUC(sub, targets=<5 log targets>, which='by_RT')",
          per_pair(function(f, d, sub) tag(generate_data.AUC(sub, tgt(f, d, sub), which = "by_RT"), pair_f = f, pair_d = d)))
  attempt(set, "AUC_by_RT_raw", "generate_data.AUC(..., normalize=FALSE)",
          per_pair(function(f, d, sub) tag(generate_data.AUC(sub, tgt(f, d, sub), which = "by_RT", normalize = FALSE), pair_f = f, pair_d = d)))
  attempt(set, "AUC_by_FV", "generate_data.AUC(sub, <5 log budgets>, which='by_FV')",
          per_pair(function(f, d, sub) tag(generate_data.AUC(sub, bud(f, d, sub), which = "by_FV"), pair_f = f, pair_d = d)))
  attempt(set, "ECDF_raw", "generate_data.ECDF_raw(dsl, targets=get_ECDF_targets(dsl, 'log-linear', 5))",
          generate_data.ECDF_raw(dsl, get_ECDF_targets(dsl, "log-linear", 5)))

  # multi-function aggregation and ranks
  if (nrow(pairs) > 1) {
    tg5 <- get_ECDF_targets(dsl, "log-linear", 5)
    attempt(set, "ECDF_default_targets", "generate_data.ECDF(dsl, targets=get_ECDF_targets(dsl,'log-linear',5)) (all pairs pooled)",
            generate_data.ECDF(dsl, tg5, which = "by_RT"))
    for (on in c("funcId", "DIM")) {
      if (length(unique(if (on == "funcId") pairs$f else pairs$d)) > 1) {
        attempt(set, paste0("Aggr_", on), sprintf("generate_data.Aggr(dsl, aggr_on='%s', targets=<5 log targets>, which='by_RT')", on),
                generate_data.Aggr(dsl, aggr_on = on, targets = tg5, which = "by_RT"))
      }
    }
  }

  # histograms and violin data
  attempt(set, "hist_by_RT", "generate_data.hist(sub, target=<3rd of 5 log targets>, which='by_RT')",
          per_pair(function(f, d, sub) tag(generate_data.hist(sub, tgt(f, d, sub)[[3]], which = "by_RT"), pair_f = f, pair_d = d)))
  attempt(set, "hist_by_FV", "generate_data.hist(sub, target=<3rd of 5 log budgets>, which='by_FV')",
          per_pair(function(f, d, sub) tag(generate_data.hist(sub, bud(f, d, sub)[[3]], which = "by_FV"), pair_f = f, pair_d = d)))
  attempt(set, "PMF", "generate_data.PMF(sub, target=<3rd of 5 log targets>, which='by_RT')",
          per_pair(function(f, d, sub) tag(generate_data.PMF(sub, tgt(f, d, sub)[[3]], which = "by_RT"), pair_f = f, pair_d = d)))
  attempt(set, "single_function_by_RT", "generate_data.Single_Function(sub, which='by_RT', include_opts=TRUE)",
          per_pair(function(f, d, sub) tag(generate_data.Single_Function(sub, which = "by_RT", include_opts = TRUE), pair_f = f, pair_d = d)))
  attempt(set, "single_function_by_FV", "generate_data.Single_Function(sub, which='by_FV', include_geom_mean=TRUE)",
          per_pair(function(f, d, sub) tag(generate_data.Single_Function(sub, which = "by_FV", include_geom_mean = TRUE), pair_f = f, pair_d = d)))

  # parameters and correlations
  pars <- unique(unlist(lapply(dsl, function(ds) get_PAR_name(ds, "by_RT"))))
  pars_all <- setdiff(pars, c("violation", "penalty"))
  pars <- setdiff(pars_all, c("x0", "x1", "x2"))
  full_fv <- isTRUE(attr(dsl[[1]], "contains_full_FV"))
  corr1 <- if (full_fv) "f(x)" else pars_all[[1]]
  corr2 <- pars_all[[min(2, length(pars_all))]]
  if (length(pars) > 0) {
    with_par <- function(sub) {
      keep <- vapply(sub, function(ds) length(setdiff(get_PAR_name(ds, "by_RT"), c("violation", "penalty"))) > 0, TRUE)
      subset(sub, algId %in% unique(vapply(sub[keep], function(ds) attr(ds, "algId"), "")))
    }
    attempt(set, "PAR_summary_by_FV", "get_PAR_summary(sub, idxValue=<5 log targets>, which='by_FV')",
            per_pair(function(f, d, sub) per_target(set, "PAR_summary_by_FV", f, d, tgt(f, d, sub), function(t) get_PAR_summary(with_par(sub), t, which = "by_FV"))))
    attempt(set, "PAR_summary_by_RT", "get_PAR_summary(sub, idxValue=<5 log budgets>, which='by_RT')",
            per_pair(function(f, d, sub) tag(get_PAR_summary(with_par(sub), bud(f, d, sub), which = "by_RT"), pair_f = f, pair_d = d)))
    attempt(set, "PAR_sample_by_FV", "get_PAR_sample(sub, idxValue=<5 log targets>, which='by_FV')",
            per_pair(function(f, d, sub) per_target(set, "PAR_sample_by_FV", f, d, tgt(f, d, sub), function(t) get_PAR_sample(with_par(sub), t, which = "by_FV"))))
    attempt(set, "PAR_sample_by_RT", "get_PAR_sample(sub, idxValue=<5 log budgets>, which='by_RT')",
            per_pair(function(f, d, sub) tag(get_PAR_sample(with_par(sub), bud(f, d, sub), which = "by_RT"), pair_f = f, pair_d = d)))
    attempt(set, "Parameters_by_RT", "generate_data.Parameters(sub, which='by_RT')",
            per_pair(function(f, d, sub) tag(generate_data.Parameters(with_par(sub), which = "by_RT"), pair_f = f, pair_d = d)))
    attempt(set, "Parameters_by_FV", "generate_data.Parameters(sub, which='by_FV')",
            per_pair(function(f, d, sub) tag(generate_data.Parameters(with_par(sub), which = "by_FV"), pair_f = f, pair_d = d)))
    attempt(set, "Parameter_correlation", "generate_data.Parameter_correlation(sub, par1, par2); par1 = 'f(x)' when the fixture logs every evaluation, else the first parameter; par2 = the second parameter (or the first)",
            per_pair(function(f, d, sub) tag(generate_data.Parameter_correlation(with_par(sub), corr1, corr2), pair_f = f, pair_d = d)))
  }

  # positions, where the fixture logs them as parameters x0..x(d-1)
  if (set == "ioh040_pos") {
    attempt(set, "positions", "get_position_dsl(sub, iid) after setting attr(ds,'contains_position') <- TRUE",
            per_pair(function(f, d, sub) {
              for (i in seq_along(sub)) attr(sub[[i]], "contains_position") <- TRUE
              tag(get_position_dsl(sub, 1:5), pair_f = f, pair_d = d)
            }))
    attempt(set, "positions_without_flag", "get_position_dsl(sub, iid), attribute left as the reader sets it",
            per_pair(function(f, d, sub) tag(get_position_dsl(sub, 1:5), pair_f = f, pair_d = d)))
  }

  # comparisons: KS tests with Holm correction, win-fraction heatmap, Glicko-2
  if (length(ids) > 1) {
    attempt(set, "KS_by_FV", "pairwise.test(sub, ftarget=<3rd of 5 log targets>, which='by_FV')",
            per_pair(function(f, d, sub) tag(pairwise.test(sub, tgt(f, d, sub)[[3]], which = "by_FV"), pair_f = f, pair_d = d)))
    attempt(set, "KS_by_RT", "pairwise.test(sub, ftarget=<3rd of 5 log budgets>, which='by_RT')",
            per_pair(function(f, d, sub) tag(pairwise.test(sub, bud(f, d, sub)[[3]], which = "by_RT"), pair_f = f, pair_d = d)))
    for (w in c("by_FV", "by_RT")) {
      attempt(set, paste0("Heatmaps_", w), sprintf("generate_data.Heatmaps(dsl, which='%s')", w),
              generate_data.Heatmaps(dsl, which = w))
      attempt(set, paste0("Heatmaps_", w, "_5targets"),
              sprintf("generate_data.Heatmaps(dsl, which='%s', target_dt=<get_ECDF_targets(dsl,'log-linear',5) median target per pair>)", w),
              {
                tdt <- get_ECDF_targets(dsl, "log-linear", 5)[, .(target = target[[3]]), by = .(funcId, DIM)]
                if (w == "by_RT") {
                  tdt <- bind(lapply(seq_len(nrow(pairs)), function(i) {
                    sub <- subset(dsl, funcId == pairs$f[[i]] & DIM == pairs$d[[i]])
                    data.table(funcId = pairs$f[[i]], DIM = pairs$d[[i]], target = bud(pairs$f[[i]], pairs$d[[i]], sub)[[3]])
                  }))
                }
                generate_data.Heatmaps(dsl, which = w, target_dt = tdt)
              })
    }
    # Glicko-2: record the game list passed to PlayerRatings::glicko2 together with the ratings
    for (w in c("by_FV", "by_RT")) {
      games <- NULL
      suppressMessages(trace(PlayerRatings::glicko2, tracer = quote(.golden_games <<- x), print = FALSE, where = asNamespace("PlayerRatings")))
      assign(".golden_games", NULL, envir = globalenv())
      attempt(set, paste0("glicko2_", w), sprintf("set.seed(%d); glicko2_ranking(dsl, nr_rounds=50, which='%s')", SEED, w), {
        set.seed(SEED)
        res <- glicko2_ranking(dsl, nr_rounds = 50, which = w)
        r <- as.data.table(res$ratings)
        r[, c("Rating", "Deviation", "Volatility", "Games", "Win", "Draw", "Loss", "Lag") :=
            lapply(c("Rating", "Deviation", "Volatility", "Games", "Win", "Draw", "Loss", "Lag"), function(n) as.numeric(r[[n]]))]
        r
      })
      g <- get(".golden_games", envir = globalenv())
      if (!is.null(g)) {
        # glicko2 receives the games with player indices; keep the names so the list is self-contained
        emit(set, paste0("glicko2_games_", w), as.data.table(g),
             "games data.frame passed to PlayerRatings::glicko2 (Weeks, Player1, Player2, Scores; players are 1-based indices into the sorted algorithm ids of the first sample table)")
      }
      suppressMessages(untrace(PlayerRatings::glicko2, where = asNamespace("PlayerRatings")))
    }
  }

  # Shapley portfolio values
  if (length(ids) > 1) {
    tg5 <- get_ECDF_targets(dsl, "log-linear", 5)
    for (lg in c(TRUE, FALSE)) {
      attempt(set, paste0("shapley_", ifelse(lg, "log", "lin")),
              sprintf("set.seed(%d); get_shapley_values(dsl, targets=<5 log targets>, scale.log=%s, group_size=5, max_perm_size=%d)", SEED, lg, length(ids)),
              {
                set.seed(SEED)
                s <- get_shapley_values(dsl, tg5, scale.log = lg, group_size = 5, max_perm_size = length(ids))
                data.table(ID = s$ID, shapley = unlist(s$shapley))
              })
    }
    attempt(set, "marg_contrib_ecdf", "get_marg_contrib_ecdf(id, perm=sorted ids, j=1.., dt=generate_data.ECDF_raw(dsl, targets))",
            {
              dt <- generate_data.ECDF_raw(dsl, tg5, scale_log = TRUE)
              perm <- sort(ids)
              bind(lapply(seq_along(perm), function(j) data.table(ID = perm[[j]], j = j,
                           contrib = get_marg_contrib_ecdf(perm[[j]], perm, j, dt))))
            })
  }

  # cumulative difference plot (RVCompare bootstrap)
  if (length(ids) > 1 && nrow(pairs) >= 1) {
    # a target that every run of the two algorithms reaches (the fixed-target CDP needs all runs to hit)
    easy_target <- function(two) {
      fin <- unlist(lapply(two, function(ds) apply(get_FV(ds), 2, function(col) if (maximize) max(col, na.rm = TRUE) else min(col, na.rm = TRUE))))
      if (maximize) min(fin) else max(fin)
    }
    attempt(set, "CDP_by_FV", sprintf("set.seed(%d); generate_data.CDP(sub[2 algorithms], <target reached by every run>, isFixedBudget=FALSE, nOfBootstrapSamples=200)", SEED),
            per_pair(function(f, d, sub) {
              two <- subset(sub, algId %in% sort(ids)[1:2])
              set.seed(SEED)
              tag(generate_data.CDP(two, easy_target(two), FALSE, nOfBootstrapSamples = 200), pair_f = f, pair_d = d)
            }))
    attempt(set, "CDP_by_FV_with_failures", sprintf("set.seed(%d); generate_data.CDP(sub[2 algorithms], <3rd of 5 log targets>, isFixedBudget=FALSE, nOfBootstrapSamples=200); IOHanalyzer fails when a run does not reach the target (I02 decision CD-2)", SEED),
            per_pair(function(f, d, sub) {
              two <- subset(sub, algId %in% sort(ids)[1:2])
              set.seed(SEED)
              tag(generate_data.CDP(two, tgt(f, d, sub)[[3]], FALSE, nOfBootstrapSamples = 200), pair_f = f, pair_d = d)
            }))
    attempt(set, "CDP_by_RT", sprintf("set.seed(%d); generate_data.CDP(sub[2 algorithms], <3rd of 5 log budgets>, isFixedBudget=TRUE, nOfBootstrapSamples=200)", SEED),
            per_pair(function(f, d, sub) {
              two <- subset(sub, algId %in% sort(ids)[1:2])
              set.seed(SEED)
              tag(generate_data.CDP(two, bud(f, d, sub)[[3]], TRUE, nOfBootstrapSamples = 200), pair_f = f, pair_d = d)
            }))
  }

  # EAF and EAF differences
  attempt(set, "EAF", "generate_data.EAF(sub, n_sets=11, subsampling=20)",
          per_pair(function(f, d, sub) tag(generate_data.EAF(sub, n_sets = 11, subsampling = 20), pair_f = f, pair_d = d)))
  attempt(set, "EAF_logx", "generate_data.EAF(sub, n_sets=11, subsampling=20, scale_xlog=TRUE)",
          per_pair(function(f, d, sub) tag(generate_data.EAF(sub, n_sets = 11, subsampling = 20, scale_xlog = TRUE), pair_f = f, pair_d = d)))
  attempt(set, "ECDF_From_EAF", "generate_data.ECDF_From_EAF(generate_data.EAF(sub, 11, 20), '', '', maximization)",
          per_pair(function(f, d, sub) {
            e <- generate_data.EAF(sub, n_sets = 11, subsampling = 20)
            tag(generate_data.ECDF_From_EAF(e, "", "", maximization = maximize), pair_f = f, pair_d = d)
          }))
  if (length(ids) > 1) {
    attempt(set, "EAF_Difference", "generate_data.EAF_Difference(sub[alg 1], sub[alg 2]) (eaf::eafdiff, rectangles=TRUE)",
            per_pair(function(f, d, sub) {
              a <- sort(ids)
              x <- subset(sub, algId == a[[1]]); y <- subset(sub, algId == a[[2]])
              tag(as.data.table(generate_data.EAF_Difference(x, y)), pair_f = f, pair_d = d)
            }))
    attempt(set, "EAF_diff_Approximate", "generate_data.EAF_diff_Approximate(sub, xmin, xmax, ymin, ymax) over the full ranges",
            per_pair(function(f, d, sub) {
              m <- generate_data.EAF_diff_Approximate(sub, 0, Inf, -Inf, Inf)
              bind(lapply(names(m), function(a) {
                z <- long(m[[a]], "fvalue", "runtime", "diff")
                tag(z, pair_f = f, pair_d = d, algId = a)
              }))
            }))
  }

  # constraint penalties (only meaningful on the constrained fixture)
  if (isTRUE(attr(dsl[[1]], "constrained"))) {
    viol <- list(
      none = NULL,
      add = function(x, y) x + 10 * y,
      absolute = function(x, y) { x[y > 0] <- x[y > 0] + y[y > 0]; x })
    for (nm in names(viol)) {
      options(IOHanalyzer.Violation_Function = viol[[nm]])
      attempt(set, paste0("penalty_", nm, "_RT_summary"),
              sprintf("options(IOHanalyzer.Violation_Function=<%s>); get_RT_summary(sub, <5 log targets>)", nm),
              per_pair(function(f, d, sub) tag(get_RT_summary(sub, tgt(f, d, sub)), pair_f = f, pair_d = d)))
      attempt(set, paste0("penalty_", nm, "_FV_summary"),
              sprintf("options(IOHanalyzer.Violation_Function=<%s>); get_FV_summary(sub, <5 log budgets>)", nm),
              per_pair(function(f, d, sub) tag(get_FV_summary(sub, bud(f, d, sub)), pair_f = f, pair_d = d)))
      attempt(set, paste0("penalty_", nm, "_FV_matrix"),
              sprintf("options(IOHanalyzer.Violation_Function=<%s>); get_FV(ds) for each data set", nm),
              bind(lapply(dsl, function(ds) {
                m <- get_FV(ds)
                z <- long(unname(m), "row", "run", "fv")
                tag(z, algId = attr(ds, "algId"), funcId = attr(ds, "funcId"), DIM = attr(ds, "DIM"))
              })))
    }
    options(IOHanalyzer.Violation_Function = NULL)
  }
}

# ---- fixtures ----------------------------------------------------------------------------------
sets <- c("ioh040_min", "ioh040_max", "ioh040_cons", "ioh040_pos", "extdata", "coco")
for (set in sets) {
  dsl <- suppressMessages(DataSetList(file.path(fix_dir, set)))
  run_fixture(set, dsl)
}

# ---- versions.json -----------------------------------------------------------------------------
pk <- c("IOHanalyzer", "PlayerRatings", "eaf", "RVCompare", "arrow", "data.table", "dplyr", "jsonlite", "Rcpp")
versions <- list(
  R = R.version.string,
  packages = setNames(lapply(pk, function(p) as.character(packageVersion(p))), pk),
  IOHanalyzer_git = "caecada^ (81b8dba1c7749e55c28d66a2f951bc9c75f8d2a5)",
  cran_snapshot = getOption("repos")[["CRAN"]],
  seed = SEED,
  outputs = index[order(names(index))],
  skipped = skipped[order(names(skipped))]
)
write(toJSON(versions, auto_unbox = TRUE, pretty = TRUE, digits = NA), file.path(out_dir, "versions.json"))
cat(sprintf("wrote %d files, skipped %d\n", length(index), length(skipped)))
