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
