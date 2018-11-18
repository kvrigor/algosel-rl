library(aslib)
library(llama)
library(tensorflow)

REINFORCE_AS <- function(dataset_hard, EPOCHS = 25, NUM_BATCHES = 64, DROPOUT_PROB = 0.5, TB_ROOTFOLDER = "testresults", OUTFILE = "") {
  NUM_FEATURES <- length(dataset_hard$features)
  NUM_ACTIONS <- length(dataset_hard$performance)
  BATCH_SIZE <- NULL
  
  # Feature normalization  -----------------------------------------------------------
  Zscore <- function(x, center, width) { (x-center)/width }
  features <- dataset_hard$data[dataset_hard$features]
  summary <- as.data.frame(rbind(sapply(features, median), sapply(features, sd)), row.names = c("mean", "sd"))
  for (i in 1:35) { features[i] <- Zscore(features[i], summary["mean", i], summary["sd", i]) }
  
  # Reward scaling  -----------------------------------------------------------
  RewardNorm <- function(x, base = 4, mid = 3) {
    if (max(x) == min(x)) {
      return(rep(0,length(x)))
    } else {
      power <- mid - rank(x, ties.method = "max")
      return(sapply(power, function(p) ifelse(p<0, (-base**(-p)), base**p)))
    }
  }
  perf <- dataset_hard$data[dataset_hard$performance]
  perf2 <- dataset_hard$data[dataset_hard$performance]
  for (i in 1:nrow(perf)) { perf[i,] <- RewardNorm(perf[i,]) } 
  
  # 10-fold cross validation  -----------------------------------------------------------
  normedData <- cbind(data.frame(iid = dataset_hard$data$instance_id, stringsAsFactors = FALSE), perf, features)
  full_stats <- list()
  test_MCPs <- matrix(rep(0,10), nrow = 10, ncol = 2)
  predictions <- data.frame(instance_id = factor(NULL, levels = levels(dataset_hard$data$instance_id)), 
                            algorithm = factor(NULL, levels = dataset_hard$performance), 
                            score = double(), 
                            iteration = integer())
  
  for (k in 1:length(dataset_hard$train)) {
    # Set TF variables -----------------------------------------------------------
    H1_UNITS <- 50
    H2_UNITS <- 50
    H3_UNITS <- 25
    LEARNING_RATE <- 0.001 
    tf$reset_default_graph()
    generate_layer_summaries <- function(tensors) {
      for (var in tensors) {
        name <- attr(var, "names")
        with(tf$name_scope(paste0('stats_', name)), {
          mean <- tf$reduce_mean(var)
          tf$summary$scalar("mean", mean)
          stddev <- tf$sqrt(tf$reduce_mean(tf$square(var - mean)))
          tf$summary$scalar("sd", stddev)
          tf$summary$scalar("max", tf$reduce_max(var))
          tf$summary$scalar("min", tf$reduce_min(var))
          tf$summary$histogram(name, var)
        })
      }
    }
    
    # State Approximator NN -----------------------------------------------------------
    with(tf$name_scope('StateApproximator'), {
      ph_State <- tf$placeholder(tf$float32, shape(BATCH_SIZE, NUM_FEATURES), name = "features")
      ph_KeepProb <- tf$placeholder(tf$float32, name = "dropoutProb")
      with(tf$name_scope('H1'), {
        w <- tf$Variable(tf$truncated_normal(shape(NUM_FEATURES, H1_UNITS), stddev = 1.0 / sqrt(NUM_FEATURES)), name = 'weights')
        b <- tf$Variable(tf$zeros(shape(H1_UNITS), name = 'biases'))
        layer_H1_pre <- tf$nn$leaky_relu(tf$matmul(ph_State, w) + b, name = "LReLU1")
        layer_H1 <- tf$nn$dropout(layer_H1_pre, ph_KeepProb)
        generate_layer_summaries(c(weights = w, biases = b, activations = layer_H1_pre))
      })
      with(tf$name_scope('H2'), {
        w <- tf$Variable(tf$truncated_normal(shape(H1_UNITS, H2_UNITS), stddev = 1.0 / sqrt(H1_UNITS)), name = 'weights')
        b <- tf$Variable(tf$zeros(shape(H2_UNITS)), name = 'biases')
        layer_H2_pre <- tf$nn$leaky_relu(tf$matmul(layer_H1, w) + b, name = "LReLU2")
        layer_H2 <- tf$nn$dropout(layer_H2_pre, ph_KeepProb)
        generate_layer_summaries(c(weights = w, biases = b, activations = layer_H2_pre))
      })
      with(tf$name_scope('H3'), {
        w <- tf$Variable(tf$truncated_normal(shape(H2_UNITS, H3_UNITS), stddev = 1.0 / sqrt(H2_UNITS)), name = 'weights')
        b <- tf$Variable(tf$zeros(shape(H3_UNITS)), name = 'biases')
        layer_H3_pre <- tf$nn$tanh(tf$matmul(layer_H2, w) + b, name = "Tanh3")
        layer_H3 <- tf$nn$dropout(layer_H3_pre, ph_KeepProb)
        generate_layer_summaries(c(weights = w, biases = b, activations = layer_H3_pre))
      })
      with(tf$name_scope('Output'), {
        w <- tf$Variable(tf$truncated_normal(shape(H3_UNITS, NUM_ACTIONS), stddev = 1.0 / sqrt(H3_UNITS)), name = 'weights')
        b <- tf$Variable(tf$zeros(shape(NUM_ACTIONS)), name = 'biases')
        layer_Out <- tf$nn$relu(tf$matmul(layer_H3, w) + b, name = "ReLU4")
        generate_layer_summaries(c(weights = w, biases = b, activations = layer_Out))
      })
    })
    
    # Agent Policy ------------------------------------------------------------
    with(tf$name_scope('AgentPolicy'), {
      ph_Action <- tf$placeholder(tf$int32, shape(BATCH_SIZE), name = "actions")
      with(tf$name_scope('SoftMax'), {
        op_action_probs <- tf$nn$softmax(layer_Out)
        summ_aprobs <- tf$summary$histogram("A_Probs", op_action_probs)
      })
      with(tf$name_scope('ActionLogProbs'), {
        op_sampled_action_prob <- tf$reduce_max(tf$multiply(tf$one_hot(ph_Action, NUM_ACTIONS), op_action_probs), axis = 1L)
        op_log_probs <- tf$log(op_sampled_action_prob, name = "sampledLogActionProbs")
      })
    })
    
    # Gradient Estimation & Optimizer-----------------------------------------------------------
    with(tf$name_scope('Objective'), {
      ph_Reward <- tf$placeholder(tf$float32, shape(BATCH_SIZE), name = "rewards")
      with(tf$name_scope('REINFORCE'), {
        # grad(obj) = -log(picked_action_probs) * rewards
        op_reinforce <- tf$negative(tf$multiply(ph_Reward, op_log_probs))
        tf$summary$scalar("Reward", tf$reduce_mean(ph_Reward))
        tf$summary$histogram("Reward", tf$reduce_mean(ph_Reward))
        tf$summary$scalar("REINFORCE", tf$reduce_mean(op_reinforce))
      })
    })
    with(tf$name_scope('PerformanceGradient'), {
      op_perfGrad <- tf$reduce_mean(op_reinforce, axis = 0L)
      tf$summary$scalar("PerfGrad", op_perfGrad)
    })
    with(tf$name_scope('GradientOptimizer'), {
      op_optimizer <- tf$train$AdamOptimizer(LEARNING_RATE)
      var_global_step <- tf$Variable(0L, name = 'global_step', trainable = FALSE)
      op_train <- op_optimizer$minimize(op_perfGrad, global_step = var_global_step)
    })
    with(tf$name_scope('MCPTracker'), {
      ph_MCP <- tf$placeholder(tf$float32, name = "MeanLogMCP")
      summ_mcp <- tf$summary$scalar("MeanLogMCP", ph_MCP)
      summ_mcp2 <- tf$summary$histogram("MeanLogMCP", ph_MCP)
    })
    merged <- tf$summary$merge_all()
    
    # Initialize TF session-----------------------------------------------------------
    
    if (exists("sess")) {sess$close()}
    sess <- tf$Session()
    sess$run(tf$global_variables_initializer())
    train_writer <- tf$summary$FileWriter(paste(TB_ROOTFOLDER, paste0("train", k), sep = '/'), sess$graph)
    test_writer <- tf$summary$FileWriter(paste(TB_ROOTFOLDER, paste0("test", k), sep = '/'), sess$graph)
    
    # Training ----------------------------------------------------------------------------------------------------------------------
    trainSet <- dataset_hard$train[[k]]
    testSet <- dataset_hard$test[[k]]
    
    cat("\n\n ******* Train/test set", k,"*******\n\n", file = OUTFILE)
    logMCP <- c()
    training_stats <- list()
    jj <- 0
    for (epoch in 1:EPOCHS) {
      trainBatch <- split(sample(trainSet), 1:NUM_BATCHES)
      sz <- length(trainBatch)
      batch_runs <- data.frame(PerfGrad = double(sz), MCP = double(sz))
      j <- 0
      for (batch in trainBatch) {
        context <- as.matrix(normedData[batch, dataset_hard$features])
        action_probs <- sess$run(op_action_probs, dict(ph_KeepProb = 1, ph_State = context))
        sampled_action <- apply(action_probs, 1, function(ap) sample(1:NUM_ACTIONS, 1, prob = ap))
        reward <- sapply(1:length(batch), function(i) normedData[batch[i], dataset_hard$performance[sampled_action[i]]])
        minibatch_MCP <- sum(sapply(1:length(batch), function(i) perf2[i, sampled_action[i]] - min(perf2[i,])))
        feed_dict <- dict(ph_KeepProb = DROPOUT_PROB, ph_State = context, ph_Action = sampled_action - 1, ph_Reward = reward, ph_MCP = log10(1+(minibatch_MCP / length(batch))))
        values <- sess$run(list(op_perfGrad, op_train, merged), feed_dict)
        j <- j + 1
        batch_runs[j,] <- c(values[[1]], minibatch_MCP) 
      }
      train_writer$add_summary(values[[3]], epoch)
      train_writer$flush()
      
      batch_MCP <- sum(batch_runs$MCP) / length(trainSet)
      cat("   Epoch", epoch, ": Mean MCP =", prettyNum(batch_MCP, big.mark = ","), "\n", file = OUTFILE)
      training_stats[[paste0("Epoch", epoch)]] <- list(PerfGrad = median(batch_runs$PerfGrad), MCP = batch_MCP, BatchData = batch_runs)
      logMCP <- c(logMCP, log10(1 + batch_MCP))
    }
    #plot(logMCP)
    
    sz <- length(testSet)
    test_runs <- data.frame(MCP = double(sz))
    test_pred <- data.frame(instance_id = factor(character(sz), levels = levels(dataset_hard$data$instance_id)), algorithm = factor(character(sz), levels = dataset_hard$performance), score = double(sz), iteration = integer(sz))
    j <- 1
    for (iid in testSet) {
      context <- as.matrix(normedData[iid, dataset_hard$features])
      action_probs <- sess$run(list(op_action_probs, summ_aprobs), dict(ph_KeepProb = 1, ph_State = context))
      test_writer$add_summary(action_probs[[2]], j)
      sampled_action <- sample(1:NUM_ACTIONS, 1, prob = action_probs[[1]])
      reward <- normedData[iid, dataset_hard$performance[[sampled_action]]]
      testMCP <- as.double(perf2[iid, sampled_action] - min(perf2[iid,]))
      values <- sess$run(list(summ_mcp, summ_mcp2), dict(ph_MCP = log10(1+testMCP)))
      test_writer$add_summary(values[[1]], j)
      test_writer$add_summary(values[[2]], j)
      test_runs[j,] <- c(testMCP) 
      
      test_pred$instance_id[j] <- as.character(normedData[iid,1])
      test_pred$algorithm[j] <- dataset_hard$performance[sampled_action]
      test_pred$score[j] <- reward
      test_pred$iteration[j] <- k
      j <- j + 1
    }
    test_writer$flush()
    cat("   Test Mean MCP =", prettyNum(sum(test_runs$MCP) / sz, big.mark = ","), "\n", file = OUTFILE)
    full_stats[[paste0("Set", k)]] <- list(Train = training_stats, TrainLogMCP = logMCP, Test = test_runs)
    test_MCPs[k,] <- c(sum(test_runs$MCP), sz)
    predictions <- rbind(predictions, test_pred)
  }
  meanMCP <- colSums(test_MCPs)[1]/colSums(test_MCPs)[2]
  
  cat("\n *** Mean MCP =", prettyNum(meanMCP, big.mark = ","), file = OUTFILE)
  retval = list(predictions=predictions, meanMCP=meanMCP, testInfo=full_stats, testMCPs=test_MCPs, models=NULL, predictor=NULL)
  class(retval) = "llama.model"
  attr(retval, "type") = "bandit"
  attr(retval, "hasPredictions") = TRUE
  attr(retval, "addCosts") = TRUE
  return(retval)
}
class(REINFORCE_AS) = "llama.modelFunction"

ExtractHardInstancesGRAPHS2015 <- function() {
  # Load dataset from file and convert to LLAMA object
  dataset_raw <- parseASScenario("GRAPHS-2015")
  dataset_structured <- convertToLlamaCVFolds(dataset_raw)
  
  #Presolver 1: Filter instances solved by IncompleteLAD
  dataset_notPresolved <- dataset_structured
  presolved_ids <- dataset_raw$feature.runstatus$instance_id[dataset_raw$feature.runstatus$lad_features == "presolved"]
  dataset_notPresolved$data <- subset(dataset_structured$data, !(dataset_structured$data$instance_id %in% presolved_ids))
  dataset_notPresolved$best <- subset(dataset_structured$best, !(dataset_structured$data$instance_id %in% presolved_ids))
  
  # Presolver 2: Filter instances solved by VF2 within 50 ms
  dataset_hard <- dataset_notPresolved
  dataset_hard$data <- subset(dataset_notPresolved$data, dataset_notPresolved$data$vf2 > 50)
  dataset_hard$best <- subset(dataset_notPresolved$best, dataset_notPresolved$data$vf2 > 50)
  dataset_hard <- cvFolds(dataset_hard, stratify = TRUE)
  
  return(dataset_hard)
}


