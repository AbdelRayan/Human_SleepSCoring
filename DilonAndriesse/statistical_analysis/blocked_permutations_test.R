# ==== Load packages ====
library(rhdf5)
library(dplyr)
library(tidyr)
library(readr)
library(glue)

# ==== Load HDF5 file into data frame ====
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


# ==== Blocked permutations test ====

# ---- Setup ----

df <- df[!df$sleep_state %in% c(5, 6), ]

set.seed(42)

features <- paste0("F", 1:3)
pairs <- combn(features, 2, simplify = TRUE)

n_perm <- 5000
sleep_states <- unique(df$sleep_state)

results <- data.frame(
  sleep_state = integer(),
  feature1 = character(),
  feature2 = character(),
  obs_stat = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)


# ---- Pairwise comparisons ----

for (ss in sleep_states) {
  
  # extract sleep state data frame
  df_ss <- df[df$sleep_state == ss, ]
  
  # extract all subjects from given state
  subjects <- unique(df_ss$subject_night)
  n_subj <- length(subjects)
  
  # ±1 matrix: rows = permutations, cols = subjects
  signs <- matrix(sample(c(-1, 1), n_perm * n_subj, replace = TRUE), 
                  nrow = n_perm, ncol = n_subj)
  
  # Map each row in df to a column in 'signs'
  subj_idx_ss <- match(df_ss$subject_night, subjects)
  
  for (i in 1:ncol(pairs)) {
    # extract features
    f1 <- pairs[1, i]
    f2 <- pairs[2, i]
    
    # compute differences per epoch
    diff_vec <- df_ss[[f1]] - df_ss[[f2]]
    obs_stat <- mean(diff_vec)
    
    # permutation
    perm_stats <- numeric(n_perm)
    for (p in 1:n_perm) {
      perm_stats[p] <- mean(diff_vec * signs[p, subj_idx_ss])
    }
    # perm_matrix <- diff_vec * signs[, subj_idx_ss]
    # perm_stats <- rowMeans(perm_matrix)
    
    # compute p-values
    p_val <- (sum(abs(perm_stats) >= abs(obs_stat)) + 1) / (n_perm + 1)
    
    # store results
    results <- rbind(results, data.frame(
      sleep_state = ss,
      feature1 = f1,
      feature2 = f2,
      obs_stat = obs_stat,
      p_value = p_val,
      stringsAsFactors = FALSE
    ))
    
    xlims <- range(c(perm_stats, obs_stat))
    
    hist(perm_stats, breaks = 50,
         xlim = xlims,
         main = glue("Permutation Distribution in sleep state {ss} ({f1} − {f2})"),
         xlab = glue("Mean({f1} − {f2})"))
    
    abline(v = obs_stat, col = "red", lwd = 2)
    
  }
}

results <- results %>%
  group_by(sleep_state) %>%
  mutate(p_adj = p.adjust(p_value, method = "holm"))

write_tsv(results, "main_indices.tsv")

results
