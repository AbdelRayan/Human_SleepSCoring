library(dplyr)
library(lme4)
library(emmeans)

# Collapse NREM substages into single group
df2 <- df %>%
  mutate(state_grouped = case_when(
    state %in% c("N1", "N2", "N3") ~ "NREM",
    TRUE ~ state
  ))

results_global <- data.frame()
results_posthoc <- data.frame()

for(cond in unique(df2$condition)){
  for(idx in unique(df2$index)){
    
    sub <- df2 %>% filter(condition == cond, index == idx)
    
    if(length(unique(sub$state_grouped)) > 1){
      model <- lmer(value ~ state_grouped + (1|subject_night), data = sub)
      
      # Global test for any state difference
      anova_res <- anova(model)
      global_p <- anova_res["state_grouped","Pr(>F)"]
      
      results_global <- rbind(results_global,
                              data.frame(condition = cond,
                                         index = idx,
                                         p_global = global_p))
      
      if(global_p < 0.05){
        em <- emmeans(model, ~ state_grouped)
        max_state <- as.data.frame(em)$state_grouped[which.max(as.data.frame(em)$emmean)]
        results_posthoc <- rbind(results_posthoc,
                                 data.frame(condition = cond,
                                            index = idx,
                                            max_state = max_state,
                                            emmean = as.data.frame(em)$emmean[which.max(as.data.frame(em)$emmean)]))
      }
    }
  }
}

# Adjust p-values for multiple tests
results_global$p_adj <- p.adjust(results_global$p_global, method = "fdr")

# Merge global test and max_state info
summary_table <- results_posthoc %>%
  left_join(results_global, by = c("condition","index"))
summary_table <- summary_table %>%
  mutate(significance = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01  ~ "**",
    p_adj < 0.05  ~ "*",
    TRUE          ~ "ns"
  ))

# Print the table
print(summary_table)

df_nrem <- df %>%
  filter(state %in% c("N1","N2","N3"))

results_global_nrem <- data.frame()
results_posthoc_nrem <- data.frame()

for(cond in unique(df_nrem$condition)){
  for(idx in unique(df_nrem$index)){
    
    sub <- df_nrem %>% filter(condition == cond, index == idx)
    
    if(length(unique(sub$state)) > 1){
      
      model <- lmer(value ~ state + (1|subject_night), data = sub)
      anova_res <- anova(model)
      global_p <- anova_res["state","Pr(>F)"]
      
      results_global_nrem <- rbind(
        results_global_nrem,
        data.frame(condition = cond,
                   index = idx,
                   p_global = global_p)
      )
      
      if(!is.na(global_p) && global_p < 0.05){
        em <- emmeans(model, ~ state)
        em_df <- as.data.frame(em)
        
        max_state <- em_df$state[which.max(em_df$emmean)]
        max_emmean <- max(em_df$emmean)
        
        results_posthoc_nrem <- rbind(
          results_posthoc_nrem,
          data.frame(condition = cond,
                     index = idx,
                     max_nrem_stage = max_state,
                     emmean = max_emmean)
        )
      }
    }
  }
}

# FDR correction
results_global_nrem$p_adj <- p.adjust(results_global_nrem$p_global, method = "fdr")

summary_nrem <- results_posthoc_nrem %>%
  left_join(results_global_nrem, by = c("condition","index")) %>%
  mutate(significance = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01  ~ "**",
    p_adj < 0.05  ~ "*",
    TRUE          ~ "ns"
  ))

print(summary_nrem)
