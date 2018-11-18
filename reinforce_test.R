library(aslib)
library (llama)

source('reinforce.R')

# Load dataset -----------------------------------------------------------
dataset_hard <- ExtractHardInstancesGRAPHS2015()

# Tensorboard config -----------------------------------------------------------
LAUNCH_TENSORBOARD <- TRUE
TENSORBOARD_LOGPATH <- paste(getwd(), "test_singlerun")
if (LAUNCH_TENSORBOARD) {
  tensorboard(TENSORBOARD_LOGPATH, "stop")
  tensorboard(TENSORBOARD_LOGPATH, "start") 
}

# Single run -----------------------------------------------------------
model_reinforce <- REINFORCE_AS(dataset_hard, EPOCHS = 25, NUM_BATCHES = 64, DROPOUT_PROB = 0.5, TB_ROOTFOLDER = TENSORBOARD_LOGPATH)


# Tune hyperparameters -----------------------------------------------------------
res_all <- data.frame()
runData <- list()
epochs <- 25
TENSORBOARD_LOGPATH <- paste(getwd(), "test_singlerun")
for (dropout in c(0.3, 0.5)) { #0.7
  for(batches in c(16,32)) #64
  {
    modelName <- paste("BATCH", batches, "DP", dropout, sep = "_")
    start_time <- Sys.time()
    model_reinforce <- REINFORCE_AS(dataset_hard, EPOCHS = epochs, NUM_BATCHES = batches, DROPOUT_PROB = dropout, TB_ROOTFOLDER = paste("test_multiruns", modelName, sep = "/"), OUTFILE = "test_multiruns.txt")
    end_time <- Sys.time()
    resrl = data.frame(model = modelName,
                        mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, model_reinforce)),
                        solved = sum(successes(dataset_hard, model_reinforce)),
                        mean.performance = mean(parscores(dataset_hard, model_reinforce, factor = 1)),
                        median.performance = median(parscores(dataset_hard, model_reinforce, factor = 1)))
    cat(modelName, ": Mean MCP =", prettyNum(resrl$mean.misclassification.penalty, big.mark = ","), ", Train duration =", end_time - start_time, "minutes\n")
    runData[[modelName]] <- model_reinforce
    res_all <- rbind(res_all, resrl)
  }
}
resvbs = data.frame(model = "virtual best",
                    mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, vbs)),
                    solved = sum(successes(dataset_hard, vbs)),
                    mean.performance = mean(parscores(dataset_hard, vbs, factor = 1)),
                    median.performance = median(parscores(dataset_hard, vbs, factor = 1)))
ressb = data.frame(model = "single best",
                   mean.misclassification.penalty = mean(misclassificationPenalties(dataset_hard, singleBest)),
                   solved = sum(successes(dataset_hard, singleBest)),
                   mean.performance = mean(parscores(dataset_hard, singleBest, factor = 1)),
                   median.performance = median(parscores(dataset_hard, singleBest, factor = 1)))
res_all <- rbind(resvbs, ressb, res_all)
View(res_all)
