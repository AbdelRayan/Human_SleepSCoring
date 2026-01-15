#==== load libraries ====
library(R.matlab)
library(dplyr)
library(coin)

#==== Load data ====
# uncomment for base EMG
# emg1_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new/new_emg_test.mat")
# emg2_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new/old_emg_test.mat")
# uncomment for index W
# emg1_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new_emg/w_new_emg_test.mat")
# emg2_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new_emg/w_old_emg_test.mat")
# uncomment for index R
emg1_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new_emg/r_new_emg_test.mat")
emg2_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new_emg/r_old_emg_test.mat")

hyp_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/thesis_images/old_vs_new_emg/hpyno_test.mat")

old_index_n_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/data_visualization/index_n_smoothed_old.mat")
old_aperiodic_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/data_visualization/aperiodic_old.mat")
new_index_n_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/data_visualization/index_n_smoothed_new.mat")
new_aperiodic_mat <- readMat("C:/Users/andri/school/bio-informatics/internship/donders/vsc/Human_SleepSCoring/DilonAndriesse/data_visualization/aperiodic_new.mat")


#==== compare old vs new channels ====
old_cells <- old_index_n_mat$value
new_cells <- new_index_n_mat$value
# old_cells <- old_aperiodic_mat$value
# new_cells <- new_aperiodic_mat$value

# old
old_df <- data.frame(
  Subject = sapply(old_cells[, 1], function(x) x[[1]]),
  State = sapply(old_cells[, 2], function(x) x[[1]]),
  FeatureIndex = as.integer(sapply(old_cells[, 3], function(x) x[[1]])),
  Feature = sapply(old_cells[, 4], function(x) x[[1]]),
  Value = as.numeric(sapply(old_cells[, 5], function(x) x[[1]])),
  stringsAsFactors = FALSE
)

# new
new_df <- data.frame(
  Subject = sapply(new_cells[, 1], function(x) x[[1]]),
  State = sapply(new_cells[, 2], function(x) x[[1]]),
  FeatureIndex = as.integer(sapply(new_cells[, 3], function(x) x[[1]])),
  Feature = sapply(new_cells[, 4], function(x) x[[1]]),
  Value = as.numeric(sapply(new_cells[, 5], function(x) x[[1]])),
  stringsAsFactors = FALSE
)

# merge into single df
merged <- merge(
  old_df,
  new_df,
  by = c("Subject", "State", "FeatureIndex", "Feature"),
  suffixes = c("_old", "_new")
)

# apply statistics
results <- merged %>%
  group_by(State) %>%
  summarise(
    p_value = wilcox.test(Value_new, Value_old,
                          paired = TRUE,
                          exact = FALSE)$p.value,
    median_diff = median(Value_new - Value_old),
    n = n()
  )

# adjusted p-value
results$p_fdr <- p.adjust(results$p_value, method = "fdr")

results


#==== compare old vs new EMG ====
# extract vectors
emg1  <- as.numeric(emg1_mat$EMG)
emg2  <- as.numeric(emg2_mat$EMG)
state <- as.vector(hyp_mat$states)

# ensure vector are the same length
n <- min(length(emg1), length(emg2), length(state))
emg1 <- emg1[1:n]
emg2 <- emg2[1:n]
hypno <- state[1:n]

# create index
epoch <- seq_len(n)

# combine data into one df
df_epoch <- data.frame(
  epoch = epoch,
  state = factor(hypno),
  emg1 = emg1,
  emg2 = emg2
)

# run paired permutation wilcox test for each sleep state
results <- df_epoch %>%
  group_by(state) %>%
  group_modify(~ {
    test <- wilcoxsign_test(
      emg1 ~ emg2,
      data = .x,
      distribution = approximate(nresample = 5000)
    )
    data.frame(
      mean_diff = mean(.x$emg1 - .x$emg2),
      p_value   = pvalue(test)
    )
  })

results