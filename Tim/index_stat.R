# -------------------------------
# Libraries
# -------------------------------
library(dplyr)
library(tidyr)
library(rstatix)

# -------------------------------
# Load data
# -------------------------------
df <- read.csv(
  "C:/Users/timmi/Documents/schooldrive/OneDrive - HAN/stage/stat_plotting/values/index_vals_per_night.csv",
  stringsAsFactors = FALSE
)

# -------------------------------
# Factor definitions
# -------------------------------
df <- df %>%
  mutate(
    subject_night = factor(subject_night),
    condition = factor(condition, levels = c("IC", "EC", "Rodent")),
    state     = factor(state, levels = c("Wake", "N1", "N2", "N3", "REM")),
    index     = factor(index, levels = c("w", "n", "r"))
  )

# -------------------------------
# Sanity check: complete pairing
# -------------------------------
pairing_check <- df %>%
  count(state, index, subject_night) %>%
  filter(n != 3)

if (nrow(pairing_check) > 0) {
  stop("Incomplete pairing detected for some subject_night × state × index combinations")
}

# -------------------------------
# Paired Wilcoxon tests
# IC vs Rodent
# EC vs Rodent
# -------------------------------
stat_results <- df %>%
  group_by(state, index) %>%
  pairwise_wilcox_test(
    value ~ condition,
    paired = TRUE,
    id = "subject_night",
    comparisons = list(
      c("IC", "Rodent"),
      c("EC", "Rodent")
    )
  ) %>%
  ungroup()

# -------------------------------
# FDR correction across ALL tests
# -------------------------------
stat_results <- stat_results %>%
  mutate(p.adj = p.adjust(p, method = "fdr"))

# -------------------------------
# Effect sizes (rank-biserial)
# -------------------------------
eff_results <- df %>%
  group_by(state, index) %>%
  wilcox_effsize(
    value ~ condition,
    paired = TRUE,
    id = "subject_night",
    comparisons = list(
      c("IC", "Rodent"),
      c("EC", "Rodent")
    )
  ) %>%
  ungroup()

# -------------------------------
# Merge statistics and effects
# -------------------------------
final_results <- stat_results %>%
  left_join(
    eff_results,
    by = c("state", "index", "group1", "group2")
  ) %>%
  arrange(state, index, group1)

# -------------------------------
# Output
# -------------------------------
print(final_results)
final_results <- final_results %>%
  mutate(
    signif_fdr = case_when(
      p.adj < 0.001 ~ "***",
      p.adj < 0.01  ~ "**",
      p.adj < 0.05  ~ "*",
      TRUE          ~ ""
    )
  )
significant_results <- final_results %>%
  filter(p.adj < 0.05)


print(significant_results, n=23)
