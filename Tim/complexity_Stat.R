library(dplyr)

# --- Load CSVs ---
df_intra <- read.csv("D:/EEG_Data_stage/stat_plotting/values/sleep_metrics_longform_intra.csv",
                     stringsAsFactors = FALSE)
df_extra <- read.csv("D:/EEG_Data_stage/stat_plotting/values/sleep_metrics_longform_extra.csv",
                     stringsAsFactors = FALSE)

# --- Combine intra and extra by subject/night, metric, and sleep_state ---
df_pair <- df_intra %>%
  rename(intra_value = value) %>%
  inner_join(
    df_extra %>% rename(extra_value = value),
    by = c("subject_night", "metric", "sleep_state")
  ) %>%
  mutate(diff = intra_value - extra_value)

# --- Function to run Wilcoxon test with direction ---
run_wilcox <- function(vals_intra, vals_extra) {
  # two-sided p
  test_two <- wilcox.test(vals_intra, vals_extra, paired = TRUE, exact = FALSE)
  
  # determine direction
  median_diff <- median(vals_intra - vals_extra, na.rm = TRUE)
  direction <- ifelse(median_diff > 0, "intra > extra",
                      ifelse(median_diff < 0, "intra < extra", "no difference"))
  
  data.frame(
    p_value = test_two$p.value,
    median_diff = median_diff,
    direction = direction
  )
}

# --- Apply to each metric × sleep_state ---
results <- df_pair %>%
  group_by(metric, sleep_state) %>%
  summarise(
    res = list(run_wilcox(intra_value, extra_value)),
    .groups = "drop"
  ) %>%
  unnest(res)

# --- Adjust for multiple comparisons using BH FDR ---
results <- results %>%
  mutate(p_adj = p.adjust(p_value, method = "BH"))

# --- Optional: inspect results ---
significant_results <- results %>%
  filter(p_adj < 0.05)
results

significant_results

library(dplyr)
library(rstatix)
library(tidyr)
library(ggplot2)

# --- Load CSVs ---
df_intra <- read.csv("D:/EEG_Data_stage/stat_plotting/values/sleep_metrics_longform_intra.csv",
                     stringsAsFactors = FALSE)
df_extra <- read.csv("D:/EEG_Data_stage/stat_plotting/values/sleep_metrics_longform_extra.csv",
                     stringsAsFactors = FALSE)

# --- Add condition labels ---
df_intra <- df_intra %>% mutate(condition = "intra")
df_extra <- df_extra %>% mutate(condition = "extra")

# --- Combine ---
df_all <- bind_rows(df_intra, df_extra)

# --- Function to run Kruskal-Wallis and compute effect size ---
analyze_stage_discriminability <- function(df, metric_name, condition_name) {
  df_sub <- df %>% filter(metric == metric_name, condition == condition_name)
  
  # Kruskal-Wallis test across stages
  kw_res <- kruskal_test(value ~ sleep_state, data = df_sub)
  
  # Effect size epsilon-squared (non-parametric)
  H <- kw_res$statistic
  k <- n_distinct(df_sub$sleep_state)
  n <- nrow(df_sub)
  epsilon_sq <- (H - k + 1) / (n - k)
  
  # Post-hoc pairwise Wilcoxon tests
  posthoc <- df_sub %>%
    pairwise_wilcox_test(value ~ sleep_state, p.adjust.method = "BH") %>%
    select(group1, group2, p)
  
  list(
    metric = metric_name,
    condition = condition_name,
    H = H,
    epsilon_sq = epsilon_sq,
    kruskal_p = kw_res$p,
    posthoc = posthoc
  )
}

# --- Run for all metrics × conditions ---
metrics <- unique(df_all$metric)
conditions <- c("intra", "extra")

results <- list()

for (m in metrics) {
  for (c in conditions) {
    res <- analyze_stage_discriminability(df_all, m, c)
    results <- append(results, list(res))
  }
}

# --- Summarize effect sizes ---
effect_summary <- lapply(results, function(x) {
  data.frame(
    metric = x$metric,
    condition = x$condition,
    H_stat = x$H,
    epsilon_sq = x$epsilon_sq,
    kruskal_p = x$kruskal_p
  )
}) %>% bind_rows()

# --- Optional: Inspect post-hoc for a metric × condition ---
# Example: aperiodic intra
posthoc_example <- results[[1]]$posthoc

# --- Compare discriminability ---
# Larger epsilon_sq → more distinct stages → better for classification
effect_summary <- effect_summary %>%
  arrange(metric, desc(epsilon_sq))

print(effect_summary)

citation()
