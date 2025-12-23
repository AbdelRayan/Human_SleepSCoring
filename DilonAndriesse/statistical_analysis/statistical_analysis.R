# ==== Install packages ====

### run to install packages
# install.packages("BiocManager")
# BiocManager::install("rhdf5")
# install.packages(c("dplyr", "tidyr", "ggplot2", "lme4", "lmerTest", "emmeans", "rstatix"))
# install.packages("PMCMRplus")
# install.packages("knitr")
# install.packages("readr")
###


# ==== Load packages ====

library(rhdf5)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)
library(rstatix)
library(stringr)
library(PMCMRplus)
library(tibble)
library(knitr)
library(glue)
library(readr)


# ==== Load HDF5 and pre-process ====
# variables
# file <- "D:/dilon_data/hdf5_collection/static_sleep-edf_norm.h5"
file <- "D:/dilon_data/hdf5_collection/static_complete-dataset_norm.h5"

# get subject names
h5 <- h5ls(file)
subject_names <- h5$name[grep("^S", h5$name)]

all_data <- list()

# iterate subjects
for (subject in subject_names) {
  # get 2d feature array and transpose into (n_samples, n_features)
  features <- t(h5read(file, paste0("/", subject, "/Features")))
  
  # get sleep states array
  states <- h5read(file, paste0("/", subject, "/Mapped_scores"))
  
  # create data frame
  subject_df <- data.frame(
    subject_night = subject,
    epoch = 1:nrow(features),
    sleep_state = states
  )
  
  # Add features
  subject_df[paste0("F", 1:ncol(features))] <- features
  
  all_data[[subject]] <- subject_df
}

# Combine all nights
df <- do.call(rbind, all_data)
str(df)

avg_df <- df %>%
  filter(sleep_state %in% 0:4) %>%
  group_by(subject_night, sleep_state) %>%
  summarize(across(starts_with("F"), mean, na.rm = TRUE), .groups="drop")
str(avg_df)
# ==== Statistical analysis per feature w.r.t. sleep states (friedman) ====

# create long data frame for statistical analysis
complete_nights_df <- avg_df %>%
  group_by(subject_night) %>%
  filter(n_distinct(sleep_state) == 5) %>%
  ungroup()

long_df <- complete_nights_df %>%
  pivot_longer(
    cols = starts_with("F"),
    names_to = "feature",
    values_to = "value"
  )

# check normality of data on subject level averages
normality_results <- long_df %>%
  group_by(feature) %>%
  summarise(
    p_shapiro = shapiro.test(value)$p.value
  ) %>%
  ungroup()

normality_results

# plot histograms for each feature - manual verification
# generated with chatgpt
ggplot(long_df, aes(x = value)) +
  geom_histogram(bins = 20, color = "black", fill = "skyblue") +
  facet_wrap(~feature, scales = "free") +
  theme_minimal() +
  labs(
    title = "Histogram of Subject-Level Average Values per Feature",
    x = "Average value per subject/night",
    y = "Count"
  )

friedman_results <- long_df %>%
  group_by(feature) %>%
  friedman_test(value ~sleep_state | subject_night)

# sort results
friedman_results_sorted <- friedman_results %>%
  mutate(feature_num = as.numeric(str_extract(feature, "\\d+"))) %>%
  arrange(feature_num) %>%
  select(-feature_num)

friedman_results_sorted

# generated using chatgpt
# Total number of subject_night before filtering
total_subjects <- n_distinct(avg_df$subject_night)

# Total number of subject_night after filtering
kept_subjects <- complete_nights_df %>%
  summarise(n = n_distinct(subject_night)) %>%
  pull(n)

# Number removed
removed_subjects <- total_subjects - kept_subjects

# Print results
cat("Total subjects:", total_subjects, "\n")
cat("Subjects kept:", kept_subjects, "\n")
cat("Subjects removed:", removed_subjects, "\n")

# Eisinga c.s. exact test
feature_names <- long_df %>%
  distinct(feature) %>%
  pull(feature)

# Create a named vector for mapping
state_map <- c(
  "0" = "Wake",
  "1" = "N1",
  "2" = "N2",
  "3" = "N3",
  "4" = "REM"
)

state_order <- c("Wake", "N1", "N2", "N3", "REM")

for (x in feature_names) {
  feature_data <- long_df %>%
    filter(feature==x) %>%
    mutate(
      sleep_state = factor(sleep_state),
      subject_night = factor(subject_night)
    )
  
  eisinga_results <- frdAllPairsExactTest(
    y = feature_data$value,
    groups = feature_data$sleep_state,
    blocks = feature_data$subject_night,
    p.adjust.method = "holm"
  )
  
  p_matrix <- eisinga_results$p.value
  stats_matrix <- eisinga_results$statistic
  
  stats_df <- as.data.frame(stats_matrix) %>% 
    rownames_to_column("Comparison1")
  p_df <- as.data.frame(p_matrix) %>% 
    rownames_to_column("Comparison1")
  
  stats_df
  
  stats_long <- stats_df %>%
    pivot_longer(-Comparison1, names_to = "Comparison2", values_to = "Test_Statistic")
  
  p_long <- p_df %>%
    pivot_longer(-Comparison1, names_to = "Comparison2", values_to = "Adjusted_P")
  
  # Combine statistic and p-value
  results_clean <- left_join(stats_long, p_long, by = c("Comparison1", "Comparison2"))
  
  results_clean
  
  # Optional: add significance stars
  results_clean <- results_clean %>%
    mutate(
      Significance = case_when(
      Adjusted_P < 0.001 ~ "***",
      Adjusted_P < 0.01  ~ "**",
      Adjusted_P < 0.05  ~ "*",
      TRUE               ~ ""
      )
    )
  results_clean <- results_clean %>%
    filter(!is.na(Test_Statistic) & !is.na(Adjusted_P))
  
  results_clean <- results_clean %>%
    mutate(
      Comparison1 = state_map[as.character(Comparison1)],
      Comparison2 = state_map[as.character(Comparison2)],
    ) %>%
    mutate(
      Comparison1 = factor(Comparison1, levels = state_order),
      Comparison2 = factor(Comparison2, levels = state_order)
    ) %>%
    arrange(Comparison2, Comparison1) %>%
    mutate(Comparison = paste(Comparison2, "vs", Comparison1)) %>%
    select(Comparison, Test_Statistic, Adjusted_P, Significance)
  
  results_clean <- results_clean %>%
    mutate(
      Adjusted_P = ifelse(
        Adjusted_P < 2.2e-16, 
        "< 2.22e-16",
        format(Adjusted_P, scientific = TRUE, digits = 3)
      )
    )
    
  # print(kable(results_clean, format = "pandoc", digits = 3, caption= glue("Eisinga c.s. exact test of {x}")))
  write.table(results_clean, file = "", sep = "\t", row.names = FALSE, quote = FALSE)
}


# ==== Statistical analysis within sleep state ====
# ---- Data frame adjustments ----

# create long data frame for statistical analysis
complete_nights_df <- avg_df %>%
  group_by(subject_night) %>%
  filter(n_distinct(sleep_state) == 5) %>%
  ungroup()

long_df <- complete_nights_df %>%
  pivot_longer(
    cols = starts_with("F"),
    names_to = "feature",
    values_to = "value"
  )
long_df

results <- long_df %>%
  group_by(sleep_state) %>%
  pairwise_wilcox_test(
    value ~ feature,
    paired=TRUE,
    p.adjust.method = "holm",
    id = "subject_night"
  ) %>%
  add_significance("p.adj")

write_tsv(results, "wilcox_results.tsv")

print(results, n=Inf)


# ---- Wake state ----

# .... Main indices ....

# create data frame with only wake
w_state_df <- long_df %>%
  filter(sleep_state == 0, feature %in% c("F1", "F2", "F3"))

# Pairwise Wilcoxon signed-rank test (within-subject)
w_state_df %>%
  wilcox_test(value ~ feature, paired = FALSE, p.adjust.method = "holm") %>%
  filter(group1 == 0)  # keep only comparisons where Index W is tested

w_state_df


# .... Extra indices ....


# ------ N1 state ------
# ------ N2 state ------
# ------ N3 state ------
# ------ REM state ------
# ---- Statistical analysis distribution of features within sleep states ----
# install.packages("vegan")
library(vegan)

df <- df %>%
  filter(sleep_state %in% 0:4) %>%
  group_by(subject_night, sleep_state)

# create long data frame for statistical analysis
complete_nights_df <- df %>%
  group_by(subject_night) %>%
  filter(n_distinct(sleep_state) == 5) %>%
  ungroup()

set.seed(42)

df_subset <- complete_nights_df %>%
  group_by(subject_night, sleep_state) %>%
  slice_sample(n = 100) %>%
  ungroup()

# Example for one sleep state
state_df <- df_subset %>%
  filter(sleep_state == 0)

state_df

# Remove identifiers
feature_matrix <- state_df %>%
  select(-subject_night)

# Optional but recommended
feature_matrix <- scale(feature_matrix)

adonis2(feature_matrix ~ sleep_state, method = "bray", data = state_df, na.rm=TRUE)


