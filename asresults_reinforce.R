library(tidyr)
library(magrittr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(grid)
library(scales)
library(aslib)
library(llama)
library(randomForest)
library(parallelMap)
source('reinforce.R')

dataset_hard <- ExtractHardInstancesGRAPHS2015()

# Model Training
if (!file.exists("model_regr_hard.rds")) {
  parallelStartSocket(4)
  parallelLibrary("llama", "mlr")
  start_time <- Sys.time()
  system.time(model_regr_hard <- regressionPairs(makeLearner("regr.randomForest"), dataset_hard))
  end_time <- Sys.time()
  saveRDS(model_regr_hard, "model_regr_hard.rds")
  cat("Training started at", format(start_time, "%X"), "and ended at", format(end_time, "%X"), "\n")
  end_time - start_time
} else {
  model_regr_hard <- readRDS("model_regr_hard.rds")
  cat("Loaded model_regr_hard from disk.\n")
}

if (!file.exists("model_reinforce.rds")) {
  start_time <- Sys.time()
  system.time(model_reinforce <- REINFORCE_AS(dataset_hard, EPOCHS = 30, NUM_BATCHES = 64, DROPOUT_PROB = 0.5, TB_ROOTFOLDER = paste(getwd(), "test_asresults"), OUTFILE = "test_reinforce.txt"))
  end_time <- Sys.time()
  saveRDS(model_reinforce, "model_reinforce.rds")
  cat("Training started at", format(start_time, "%X"), "and ended at", format(end_time, "%X"), "\n")
  end_time - start_time
} else {
  model_reinforce <- readRDS("model_reinforce.rds")
  cat("Loaded model_reinforce from disk.\n")
}


# Algorithm selection results 
resvbs <- data.frame(model = "Virtual best solver",
                    mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, vbs)),
                    solved = sum(successes(dataset_hard, vbs)),
                    mean.performance = mean(parscores(dataset_hard, vbs, factor = 1)),
                    median.performance = median(parscores(dataset_hard, vbs, factor = 1)))
ressb <- data.frame(model = "Single best solver",
                   mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, singleBest)),
                   solved = sum(successes(dataset_hard, singleBest)),
                   mean.performance = mean(parscores(dataset_hard, singleBest, factor = 1)),
                   median.performance = median(parscores(dataset_hard, singleBest, factor = 1)))
resrp <- data.frame(model = "Pairwise random forest regression",
                   mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, model_regr_hard)),
                   solved = sum(successes(dataset_hard, model_regr_hard)),
                   mean.performance = mean(parscores(dataset_hard, model_regr_hard, factor = 1)),
                   median.performance = median(parscores(dataset_hard, model_regr_hard, factor = 1)))
resrl <- data.frame(model = "REINFORCE",
                   mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, model_reinforce)),
                   solved = sum(successes(dataset_hard, model_reinforce)),
                   mean.performance = mean(parscores(dataset_hard, model_reinforce, factor = 1)),
                   median.performance = median(parscores(dataset_hard, model_reinforce, factor = 1)))
results <- rbind(resvbs, resrp, ressb, resrl)


# CDF plot, using original runtime values
runtimes <-  data.frame(PRFR = parscores(dataset_hard, model_regr_hard, factor = 1), 
                        REINFORCE = parscores(dataset_hard, model_reinforce, factor = 1), 
                        VBS = parscores(dataset_hard, vbs, factor = 1), 
                        SBS = parscores(dataset_hard, singleBest, factor = 1))
runtimes_long <- gather(runtimes, model, time, PRFR:SBS)

cdfplot = ggplot(runtimes_long, aes(x = time, col = model)) +
  stat_ecdf() +
  scale_linetype_manual(values=c(3,1), guide = FALSE) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x, n = 10),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, (1e8)-1)) +
  coord_cartesian(xlim = c(1, (1e8)-1),
                  ylim = c(0,1)) +
  ylab("fraction of instances solved") + xlab("runtime [ms]") +
  annotation_logticks(sides = "b") +
  theme_tufte(base_family='Times', base_size = 14) +
  guides(col = guide_legend(ncol = 2, keyheight = .8)) +
  theme(legend.justification=c(1,0), legend.position=c(1,0.7), aspect.ratio = 0.6, axis.line = element_line(colour="black"), panel.grid = element_line(), panel.grid.major = element_line(colour="lightgray"))

cdfplot_zoom = ggplot(runtimes_long, aes(x = time, col = model)) +
  stat_ecdf() +
  scale_linetype_manual(values=c(3,1), guide = FALSE) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x, n = 3),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, (1e8)-1)) +
  coord_cartesian(xlim = c(1e7, (1e8)-1),
                  ylim = c(.963,1)) +
  annotation_logticks(sides = "b") +
  theme_tufte(base_family='Times', base_size = 14) +
  theme(legend.position="none",
        axis.title.x=element_blank(), axis.title.y=element_blank(),
        panel.background = element_rect(fill='white', colour = "white"),
        axis.line = element_line(colour="black"),
        panel.grid = element_line(),
        panel.grid.major = element_line(colour="lightgray"))

vp = viewport(width = 0.57, height = 0.41, x = 0.71, y = 0.41)
print(cdfplot)
print(cdfplot_zoom, vp = vp)

summary(runtimes)


# CDF plot, using log-scaled runtime values
runtimes_logscaled <- runtimes_long %>% 
  mutate(time = replace(time, time == 0, 1)) %>% 
  mutate(time = log10(time))

cdfplot_log <- ggplot(runtimes_logscaled, aes(x = time, col = model)) +
  stat_ecdf() +
  ylab("fraction of instances solved") + xlab("log(runtime)") +
  theme_tufte(base_family='Times', base_size = 14) +
  guides(col = guide_legend(ncol = 2, keyheight = .8)) +
  theme(legend.justification=c(1,0), legend.position=c(1,0.6), aspect.ratio = 0.6, axis.line = element_line(colour="black"), panel.grid = element_line(), panel.grid.major = element_line(colour="lightgray"))

cdfplot_log_zoom <- ggplot(runtimes_logscaled, aes(x = time, col = model)) +
  stat_ecdf() +
  coord_cartesian(xlim = c(4, 8), ylim = c(.75,1)) +
  theme_tufte(base_family='Times', base_size = 14) +
  theme(legend.position="none",
        axis.title.x=element_blank(), axis.title.y=element_blank(),
        panel.background = element_rect(fill='white', colour = "white"),
        axis.line = element_line(colour="black"),
        panel.grid = element_line(),
        panel.grid.major = element_line(colour="lightgray"))

vp = viewport(width = 0.55, height = 0.41, x = 0.72, y = 0.39)
print(cdfplot_log)
print(cdfplot_log_zoom, vp = vp)

runtimes_logscaled_wide <- runtimes_logscaled %>% 
                           mutate(ID = rep(c(1:2336),4)) %>% 
                           spread(model, time)
summary(runtimes_logscaled_wide[,2:5])