library(aslib)

dataset = parseASScenario("GRAPHS-2015")
summary(dataset)

# Summary of features 
getFeatureNames(dataset)
summarizeFeatureValues(dataset)

# Summary of algorithm performance
getAlgorithmNames(dataset)
summarizeAlgoPerf(dataset) 
summarizeAlgoRunstatus(dataset)

# Algorithm performance plots
plotAlgoPerfBoxplots(dataset, impute.zero.vals = TRUE, log = TRUE)
plotAlgoPerfDensities(dataset, impute.zero.vals = TRUE, log = TRUE)
plotAlgoPerfCDFs(dataset, impute.zero.vals = TRUE, log = TRUE)
plotAlgoPerfScatterMatrix(dataset, impute.zero.vals = TRUE, log = TRUE)

# Correlation matrix
plotAlgoCorMatrix(dataset)

