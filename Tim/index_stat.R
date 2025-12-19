# -------------------------------
# Paired comparisons with majority-based test + FDR correction
# -------------------------------

library(dplyr)
library(rstatix)
library(tidyr)

# Load CSV
df <- read.csv("G:/EEG_Data_stage/stat_plotting/values/index_vals_per_night.csv")

# Convert to factors
df$condition <- factor(df$condition, levels = c("IC", "EC", "Rodent"))
df$state <- factor(df$state, levels = c("Wake", "N1", "N2", "N3", "REM"))
df$index <- factor(df$index, levels = c("w", "n", "r"))

# -------------------------------
# 1. Test normality per group
# -------------------------------
normality <- df %>%
  group_by(state, index, condition) %>%
  summarise(shapiro_p = shapiro.test(value)$p.value, .groups = "drop")

# Determine majority-based test
n_normal <- sum(normality$shapiro_p > 0.05)
n_non_normal <- sum(normality$shapiro_p <= 0.05)
test_to_use <- ifelse(n_normal >= n_non_normal, "t", "wilcox")
cat("Majority-based test chosen:", ifelse(test_to_use == "t", "Paired t-test", "Wilcoxon signed-rank test"), "\n")

# -------------------------------
# 2. Compute mean ± SEM per state/index/condition
# -------------------------------
summary_stats <- df %>%
  group_by(state, index, condition) %>%
  summarise(mean = mean(value),
            sem = sd(value)/sqrt(n()),
            .groups = "drop")

# -------------------------------
# 3. Paired comparisons
# -------------------------------
pairwise_comparisons <- list(
  c("IC", "EC"),
  c("IC", "Rodent"),
  c("EC", "Rodent")
)

results <- data.frame()

for(s in levels(df$state)){
  for(idx in levels(df$index)){
    df_subset <- df %>% filter(state == s, index == idx)
    
    for(comp in pairwise_comparisons){
      cond1 <- comp[1]
      cond2 <- comp[2]
      
      df_pair <- df_subset %>% filter(condition %in% c(cond1, cond2)) %>%
        arrange(subject_night)  # ensure pairing
      
      x <- df_pair$value[df_pair$condition == cond1]
      y <- df_pair$value[df_pair$condition == cond2]
      
      if(test_to_use == "t"){
        test_res <- t.test(x, y, paired = TRUE)
        statistic <- test_res$statistic
        p_val <- test_res$p.value
        test_name <- "Paired t-test"
      } else {
        test_res <- wilcox.test(x, y, paired = TRUE, exact = FALSE)
        statistic <- test_res$statistic
        p_val <- test_res$p.value
        test_name <- "Wilcoxon signed-rank"
      }
      
      results <- rbind(results, data.frame(
        state = s,
        index = idx,
        comparison = paste(cond1, "vs", cond2),
        test = test_name,
        statistic = statistic,
        p_value = p_val
      ))
    }
  }
}

# -------------------------------
# 4. FDR correction
# -------------------------------
results$adj_p_value <- p.adjust(results$p_value, method = "fdr")

# Filter significant comparisons
results_significant <- results %>% filter(adj_p_value < 0.05)

# Merge summary stats for reporting (optional)
results_wide <- results_significant %>%
  separate(comparison, into = c("cond1", "cond2"), sep = " vs ") %>%
  left_join(summary_stats %>% rename(mean1 = mean, sem1 = sem), 
            by = c("state", "index", "cond1" = "condition")) %>%
  left_join(summary_stats %>% rename(mean2 = mean, sem2 = sem), 
            by = c("state", "index", "cond2" = "condition")) %>%
  mutate(
    direction = case_when(
      mean1 > mean2 & adj_p_value < 0.05 ~ "higher",
      mean1 < mean2 & adj_p_value < 0.05 ~ "lower",
      TRUE ~ "ns"
    ),
    significance = case_when(
      adj_p_value < 0.001 ~ "***",
      adj_p_value < 0.01  ~ "**",
      adj_p_value < 0.05  ~ "*",
      TRUE                ~ "ns"
    )
  )

results_wide




# Save results
#write.csv(results_wide, "G:/EEG_Data_stage/stat_plotting/values/pairwise_majority_test_results.csv", row.names = FALSE)
#cat("Pairwise comparison results saved!\n")
