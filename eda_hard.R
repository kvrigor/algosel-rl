library(ggplot2)
library(tidyr)
library(dplyr)
library(purrr)
library(magrittr)
library(aslib)
library (llama)

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
dataset_hard <- cvFolds(dataset_hard)
cat("Count of all instances  =", nrow(dataset_structured$data), ", Count of hard instances =", nrow(dataset_hard$data)) # should be 5725 and 2336

# Peruse dataset
head(dataset_hard$data[,c("instance_id",dataset_hard$features)],10)
head(dataset_hard$data[,c("instance_id",dataset_hard$performance)],10)

# Feature histograms
dataset_hard$data %>%
                  subset(select = c(2:37)) %>% 
                  keep(is.numeric) %>%
                  gather() %>%
                  group_by(key) %>%
                  mutate(med = median(value)) %>%
                  ggplot(aes(value)) +
                  geom_histogram() +
                  facet_wrap(~ key, scales = "free", ncol = 4) +
                  geom_vline(aes(xintercept= med, group = key),
                             color = "red", linetype = "dashed",
                             size = 1)

# Algorithm runtimes histograms
dataset_hard$data %>%
                  subset(select = dataset_hard$performance) %>% 
                  keep(is.numeric) %>%
                  gather() %>%
                  group_by(key) %>%
                  mutate(med = median(value)) %>%
                  ggplot(aes(value)) +
                  geom_histogram() +
                  facet_wrap(~ key, scales = "free", ncol = 2) +
                  geom_vline(aes(xintercept= med, group = key),
                             color = "red", linetype = "dashed",
                             size = 1)

# Algorithm runtimes histograms, log-scaled
dataset_hard$data %>%
                  subset(select = dataset_hard$performance) %>% 
                  add(1) %>% log10() %>% 
                  keep(is.numeric) %>%
                  gather() %>%
                  group_by(key) %>%
                  mutate(med = median(value)) %>%
                  ggplot(aes(value)) +
                  geom_histogram() +
                  facet_wrap(~ key, scales = "free", ncol = 2) +
                  geom_vline(aes(xintercept= med, group = key),
                             color = "red", linetype = "dashed",
                             size = 1)