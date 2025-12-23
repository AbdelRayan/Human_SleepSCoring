# ==== Load packages ====
library(rhdf5)
library(dplyr)
library(tidyr)
library(readr)
library(lme4)

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


# ==== Linear mixed model ====

df$diff_F1_F2 <- df$F1 - df$F2

# Fit model
model <- lmer(diff_F1_F2 ~ sleep_state + (1 | subject_night), data = df)

# Extract residuals
res <- resid(model)

# Histogram / QQ plot
hist(res)
qqnorm(res); qqline(res)

# Plot residuals vs fitted
plot(fitted(model), res)
abline(h = 0, col = "red")