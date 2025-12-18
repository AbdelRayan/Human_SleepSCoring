library(dplyr)
library(tidyr)
library(rstatix)

# --- Combine all conditions ---
df_intra$condition <- "intra"
df_rodent$condition <- "rodent"
df_extra$condition <- "extra"
df_all <- bind_rows(df_intra, df_rodent, df_extra)

# --- Shapiro-Wilk test per metric × condition ---
normality_results <- df_all %>%
  group_by(metric, condition) %>%
  summarise(
    shapiro_p = if(n() >= 3 & n() <= 5000) shapiro.test(value)$p.value else NA_real_,
    n = n(),
    significant = shapiro_p < 0.05,
    .groups = "drop"
  )

# --- Function to run paired Wilcoxon for a given pair ---
run_wilcox_pair <- function(df, cond1, cond2) {
  df1 <- df %>% filter(condition == cond1)
  df2 <- df %>% filter(condition == cond2)
  
  df_pair <- df1 %>%
    inner_join(df2, by = c("subject_night", "metric", "sleep_state"), suffix = c("_1", "_2")) %>%
    mutate(diff = value_1 - value_2)
  
  df_pair %>%
    group_by(metric, sleep_state) %>%
    summarise(
      test = list(wilcox.test(value_1, value_2, paired = TRUE, exact = FALSE)),
      median_diff = median(diff, na.rm = TRUE),
      direction = ifelse(median_diff > 0, paste(cond1, ">", cond2),
                         ifelse(median_diff < 0, paste(cond1, "<", cond2), "no difference")),
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(p_value = test$p.value) %>%
    select(-test) %>%
    ungroup() %>%
    mutate(p_adj = p.adjust(p_value, method = "BH"))
}

# --- Apply for all pairs ---
results_intra_rodent <- run_wilcox_pair(df_all, "intra", "rodent")
results_intra_extra  <- run_wilcox_pair(df_all, "intra", "extra")
results_rodent_extra <- run_wilcox_pair(df_all, "rodent", "extra")

# --- Combine all pairwise results ---
all_results <- bind_rows(
  results_intra_rodent %>% mutate(pair = "intra vs rodent"),
  results_intra_extra  %>% mutate(pair = "intra vs extra"),
  results_rodent_extra %>% mutate(pair = "rodent vs extra")
)

# --- Optional: view normality and pairwise Wilcoxon results ---
# print(normality_results)
print(all_results)

# --- Filter only significant Wilcoxon comparisons ---
significant_pairs <- all_results %>%
  filter(p_adj < 0.05) %>%
  arrange(metric, sleep_state, pair) %>%
  select(metric, sleep_state, pair, median_diff, direction, p_value, p_adj)

# --- Optional: view significant results ---
print(significant_pairs)

# --- Average across sleep states per subject-night × metric × condition ---
df_avg <- df_all %>%
  group_by(subject_night, metric, condition) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  )

# --- Pivot to wide for Friedman test ---
df_wide <- df_avg %>%
  pivot_wider(
    names_from  = condition,
    values_from = value
  ) %>%
  drop_na(intra, rodent, extra)

# --- Friedman test per metric ---
friedman_results <- df_wide %>%
  group_by(metric) %>%
  summarise(
    p_value = friedman.test(
      as.matrix(cbind(intra, rodent, extra))
    )$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    p_adj = p.adjust(p_value, method = "BH"),
    significant = p_adj < 0.05
  )

# --- Results ---
print(friedman_results)