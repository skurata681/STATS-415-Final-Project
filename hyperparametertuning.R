##########LIBRARY#######################
set.seed(245)

library(haven)
library(readr)
library(dplyr)
library(tidyverse)

#libraries for subset methods
library(SignifReg)
library(leaps)

#shrinkage methods
library(glmnet)
library(pls)
library(lares)

#LDA and QDA model
library(MASS)

#Tree library
library(tree)
#RandomForest
library(randomForest)
#gradient boosting
library(gbm)

#support vector classifier
library(e1071)

#neural network
library(keras)
library(neuralnet)
library(superml)

#KNN library
library(FNN)
###################DATA#########################

#2009
bmx09 <- read_xpt("Data/2009/BMX_F.XPT")
bpx09 <- read_xpt("Data/2009/BPX_F.XPT")
demo09 <- read_xpt("Data/2009/DEMO_F.XPT")
dr109 <- read_xpt("Data/2009/DR1TOT_F.XPT")
smq09 <- read_xpt("Data/2009/SMQ_F.XPT")
tchol09 <- read_xpt("Data/2009/TCHOL_F.XPT")

data09 = merge(bmx09, bpx09, by = "SEQN", all.x = TRUE)
listy = list(demo09, dr109, smq09, tchol09)
for(i in 1:4){
  data09 = merge(data09, listy[[i]], by = "SEQN", all.x = TRUE)
}
write_csv(data09, "Data/2009/data2009.csv")

#2011
bmx11 <- read_xpt("Data/2011/BMX_G.XPT")
bpx11 <- read_xpt("Data/2011/BPX_G.XPT")
demo11 <- read_xpt("Data/2011/DEMO_G.XPT")
dr111 <- read_xpt("Data/2011/DR1TOT_G.XPT")
smq11 <- read_xpt("Data/2011/SMQ_G.XPT")
tchol11 <- read_xpt("Data/2011/TCHOL_G.XPT")

data11 = merge(bmx11, bpx11, by = "SEQN", all.x = TRUE)
listy = list(demo11, dr111, smq11, tchol11)
for(i in 1:4){
  data11 = merge(data11, listy[[i]], by = "SEQN", all.x = TRUE)
}
write_csv(data11, "Data/2011/data2011.csv")

#2013
bmx13 <- read_xpt("Data/2013/BMX_H.XPT")
bpx13 <- read_xpt("Data/2013/BPX_H.XPT")
demo13 <- read_xpt("Data/2013/DEMO_H.XPT")
dr113 <- read_xpt("Data/2013/DR1TOT_H.XPT")
smq13 <- read_xpt("Data/2013/SMQ_H.XPT")
tchol13 <- read_xpt("Data/2013/TCHOL_H.XPT")

data13 = merge(bmx13, bpx13, by = "SEQN", all.x = TRUE)
listy = list(demo13, dr113, smq13, tchol13)
for(i in 1:4){
  data13 = merge(data13, listy[[i]], by = "SEQN", all.x = TRUE)
}
write_csv(data13, "Data/2013/data2013.csv")

#2015
bmx15 <- read_xpt("Data/2015/BMX_I.XPT")
bpx15 <- read_xpt("Data/2015/BPX_I.XPT")
demo15 <- read_xpt("Data/2015/DEMO_I.XPT")
dr115 <- read_xpt("Data/2015/DR1TOT_I.XPT")
smq15 <- read_xpt("Data/2015/SMQ_I.XPT")
tchol15 <- read_xpt("Data/2015/TCHOL_I.XPT")

data15 = merge(bmx15, bpx15, by = "SEQN", all.x = TRUE)
listy = list(demo15, dr115, smq15, tchol15)
for(i in 1:4){
  data15 = merge(data15, listy[[i]], by = "SEQN", all.x = TRUE)
}
write_csv(data15, "Data/2015/data2015.csv")

#2017
bmx17 <- read_xpt("Data/2017/BMX_J.XPT")
bpx17 <- read_xpt("Data/2017/BPX_J.XPT")
demo17 <- read_xpt("Data/2017/DEMO_J.XPT")
dr117 <- read_xpt("Data/2017/DR1TOT_J.XPT")
smq17 <- read_xpt("Data/2017/SMQ_J.XPT")
tchol17 <- read_xpt("Data/2017/TCHOL_J.XPT")

data17 = merge(bmx17, bpx17, by = "SEQN", all.x = TRUE)
listy = list(demo17, dr117, smq17, tchol17)
for(i in 1:4){
  data17 = merge(data17, listy[[i]], by = "SEQN", all.x = TRUE)
}
write_csv(data17, "Data/2017/data2017.csv")

#Total
data = bind_rows(data09,data11)
listy = list(data13,data15,data17)
for(i in 1:3){
  data = bind_rows(data, listy[[i]])
}
write_csv(data,"Data/full_data.csv")

train = read_csv("Data/Kaggle/train.csv")
train = merge(train, data, by = "SEQN", all.x = TRUE)
train=train[, colSums(is.na(train))==0]
train = train[, -which(names(train)%in%c("SMDUPCA", "SMD100BR", "DR1DRSTZ", "DRABF", "RIDSTATR"))]
write_csv(train, "Data/Total/train.csv")
test = read_csv("Data/Kaggle/test.csv")
test = merge(test, data, by = "SEQN", all.x = TRUE)
test=test[, colSums(is.na(test))==0]
test = test[, -which(names(test)%in%c("SMDUPCA", "SMD100BR", "DR1DRSTZ", "DRABF", "RIDSTATR"))]
drops = c(setdiff(names(test), names(train)))
test = test[,!(names(test) %in% drops)]
write_csv(test, "Data/Total/test.csv")

##################Data Preparation#####################

scale_train<-scale(train)
scale_train<-as.data.frame(scale_train)
test$y <- seq(1:3823)
scale_test<-scale(test)
scale_test<-as.data.frame(scale_test)
# scale_train <- scale_train[sample(1:nrow(scale_train)), ]
#training and testing data for neural networks
test_idx <- sample(seq(nrow(scale_train)), size= round(nrow(scale_train)*.3)) 
train_idx <- seq(nrow(scale_train))[-test_idx]
train_neural_networks <- scale_train[train_idx,]
test_neural_networks <- scale_train[test_idx,]

#train and test models for neural networks
xTr = model.matrix(lm(y ~ . - 1, data=train_neural_networks))
yTr = train_neural_networks$y
xTe = model.matrix(lm(y ~ . - 1, data=test_neural_networks))
yTe = test_neural_networks$y

####################Define Model#########################
FLAGS <- flags(
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.4),
  flag_numeric("dropout4", 0.3),
  flag_numeric("units1", 20),
  flag_numeric("units2", 20),
  flag_numeric("units3", 20),
  flag_numeric("units4", 20),
  flag_numeric("regulizar1", 0.1),
  flag_numeric("regulizar2", 0.1),
  flag_numeric("regulizar3", 0.1),
  flag_numeric("regulizar4", 0.1),
  flag_integer("epochs1", 32),
  flag_numeric("batchsize1", 6000),
  flag_numeric("learning1", 0.01)
)

model_exhaust <- keras_model_sequential () %>%
layer_dense(units = FLAGS$units1, activation = "relu", kernel_regularizer = regularizer_l1(FLAGS$regulizar1)) %>%
layer_dropout(rate=FLAGS$dropout1) %>%
layer_dense(units = FLAGS$units2, activation = "relu", kernel_regularizer = regularizer_l1(FLAGS$regulizar2)) %>%
layer_dropout(rate=FLAGS$dropout2) %>%
layer_dense(units = FLAGS$units3, activation = "relu", kernel_regularizer = regularizer_l1(FLAGS$regulizar3)) %>%
layer_dropout(rate=FLAGS$dropout3) %>%
layer_dense(units = FLAGS$units4, activation = "relu", kernel_regularizer = regularizer_l1(FLAGS$regulizar4)) %>%
layer_dropout(rate=FLAGS$dropout4) %>%
  layer_dense(units = 1)

model_exhaust %>% compile(loss = "mse",
optimizer = optimizer_adam(learning_rate=FLAGS$learning1))

################Fit Model to Data#########################################

history <- model_exhaust %>% fit(
xTr, yTr,
epochs = FLAGS$epochs1,
batch_size = FLAGS$batchsize1,
verbose = 1,
validation_split = 0.2
)

score <- model_exhaust %>% evaluate(
  xTe, yTe,
  verbose = 0
)


npred <- c(predict(model_exhaust, xTe))
(plain_net <- cor(npred, yTe)^2) # test r^2

##################Output Metric ############################
cat('Test loss:', score[[1]], plain_net,'\n')