library(tidyverse)


df <- read.csv(".../HE_samples_IPH_sum_sigmoid_075.csv")
rnaseq <- read.csv(".../AE_rnaseq_vst.csv")
#rnaseq[, 2:ncol(rnaseq)] <- log2(rnaseq[, 2:ncol(rnaseq)] + 1e-6)
metadata <- read.csv(".../20230810.CONVOCALS.samplelist.withSMAslides.csv")
metadata$STUDY_NUMBER <- paste0("AE", metadata$STUDY_NUMBER)

exp_metadata <- read.csv(".../AE_exp_metadata.csv")
exp_metadata <- subset(exp_metadata, select = -X)
metadata <- merge(metadata, exp_metadata, by="STUDY_NUMBER")

colnames(metadata)[colnames(metadata) == "STUDY_NUMBER"] <- "case_id"
covariates <- metadata[,c("case_id", "Gender", "Age", "StudyName")]
covariates <- covariates[covariates$Gender %in% c('female'), ]

#covariates <- metadata[,c("case_id", "Gender", "Age", "StudyName", "Symptoms.Update3G")]
#covariates <- covariates[covariates$Symptoms.Update3G %in% c('Symptomatic', 'Asymptomatic'), ]
#covariates$symptoms <- as.integer(covariates$Symptoms.Update3G == "Symptomatic")

df$case_id <- sapply(strsplit(df$case_id, "\\."), `[`, 1)

areas <- df[, c("case_id", "area")]


out_df <- data.frame(
  gene = character(),
  coefficient = numeric(),
  p_val = numeric()
)

 
for (idx in 1:length(rnaseq[, 1])){
  gene_name <- rnaseq[idx, 1]
  tmp <- t(rnaseq[idx,2:dim(rnaseq)[2]])
  tmp <- (data.frame(tmp, case_id = rownames(tmp)))
  colnames(tmp) = c("exp", "case_id")
  
  #merged <- merge(cp, tmp, by="case_id")
  #merged <- merge(tmp, covariates, by="case_id")
  merged <- merge(areas, tmp, by="case_id")
  merged <- merge(merged, covariates, by="case_id")
  merged <- na.omit(merged)
  
  formula_string <- paste("exp ~", "area", "+ Age + StudyName")
  # Convert the string into a formula
  model_formula <- as.formula(formula_string)
  
  # Fit the model
  model <- lm(model_formula, data = merged)
  
  #model <- lm (exp ~ gt + Age + Gender + StudyName, data=merged )
  summary(model)
  
  new_row <- list(gene = gene_name, coefficient = summary(model)$coefficients["area", "Estimate"], p_val = summary(model)$coefficients["area", "Pr(>|t|)"])
  print(gene_name)
  
  if (idx == 1){
    print(paste(dim(merged))) 
  }
  
  out_df<- rbind(out_df, new_row)
}

# Basic boxplot
#boxplot(area ~ Hypertension.composite, data = combined_df, main = "Boxplot with Mean Points", xlab = "Group", ylab = "Values")


print(sprintf("Dim: %d", dim(merged)[1]))

out_df$significance <- NA_character_
out_df$p_val_adj <- p.adjust(out_df$p_val, method = "fdr")

out_df %>% distinct(gene, .keep_all = TRUE)
out_df$log10_pval <- -log10(out_df$p_val_adj)
names(out_df)[names(out_df) == "coefficient"] <- "FoldChange"

if (nrow (out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > 0.5),]) > 0){
  out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > 0.5),]$significance <- "UP"
}
if (nrow (out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange < -0.5),]) > 0){
  out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange < -0.5),]$significance <- "DOWN"
}

out_df$symbol = sub("_.*", "", out_df$gene)

out_df$to_plot <- ifelse(out_df$gene %in% (out_df %>% drop_na(significance) %>%  group_by(significance) %>% slice_max(abs(FoldChange), n=10) %>% pull(gene)), out_df$symbol, NA_character_ )


#out_df <- read.csv( "/home/f.cisternino/IBD/AE_data/results-plaque-composition/IPH_GLYCC_vst.csv")


if (nrow (out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > -0.5 & out_df$FoldChange < 0.5),]) > 0){
  out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > -0.5 & out_df$FoldChange < 0.5),]$significance <- "US"
}

write.csv(out_df, file = paste0(".../results-plaque-composition/DGE-IPH-Female.csv"), row.names = FALSE)


if (nrow (out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > -0.5 & out_df$FoldChange < 0.5),]) > 0){
  out_df[abs(out_df$log10_pval) > -log10(0.05) & (out_df$FoldChange > -0.5 & out_df$FoldChange < 0.5),]$significance <- "US"
}


plot = ggplot(out_df, aes(x = FoldChange, y = log10_pval)) +
  geom_point(aes(color = significance), size = 1.5) +
  scale_color_manual(values = c("UP" = "#B02528", "DOWN"= "#6E94CD", "US" = "darkgreen"), na.value = "grey70") +
 
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "black") +
  ggrepel::geom_label_repel(aes(label=to_plot), max.overlaps = 30, force=6.0, max.iter = 50000, nudge_x = 0.3, nudge_y = 0.3, alpha=0.9) +
  labs(x = "", y = "", fill=NULL) +
  xlim(-round(max(abs(out_df$FoldChange)) + 0.5) , round(max(abs(out_df$FoldChange)) + 0.5) ) + 
  ylim(0, 8)+
  theme_bw() +
  theme(legend.position = "none", text = element_text(size=15),panel.grid = element_blank())
  
#ggsave("/home/f.cisternino/IBD/AE_data/Fig1.pdf", plot=plot, width = 6, height = 7, dpi=300)
ggsave(paste0(".../results-plaque-composition/", "IPH-Female", ".png"), plot=plot, width = 7, height = 9, dpi=300)
#ggsave(paste0("/home/f.cisternino/IBD/AE_data/results-plaque-composition/", "GLYCC-F", ".png"), plot=plot, width = 6, height = 6, dpi=300)
#ggsave(paste0("/home/f.cisternino/IBD/AE_data/results-plaque-composition/", "IPH-GT", ".png"), plot=plot, width = 7, height = 9, dpi=300)


  #cellprofiler_df <- read.csv("/home/f.cisternino/IBD/AE_data/stains_cellprofiler_output.csv")
  
  #sampled <- rnaseq[sample(nrow(rnaseq), 1000, replace=FALSE), 
  #sampled <- sampled[, 2:dim(sampled)[2] ]
  # A <-  colMeans(sampled, na.rm=TRUE)
  
  #hmox1_numeric <- subset(rnaseq, symbol=='HMOX1')[2:dim(rnaseq)[2]]
  #B <- unlist(unname(as.vector(hmox1_numeric["2211",])))
  
  #hmox_sampled <- data.frame(
  #Value = c(A, B),
  #Group = rep(c("Sampled", "HMOX1"), each = 1093)
  #)
  
