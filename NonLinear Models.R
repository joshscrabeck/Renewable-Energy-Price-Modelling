
library(forecast)
library(KFAS)
library(tidyverse)
library(ggplot2)
library(xts)
library(urca)
library(plotly)
library(naniar)
library(prophet)
library(tseries)
library(fastDummies)
library(bizdays)
library(RQuantLib)
library(lubridate)
library(reticulate)
library(tsfknn)
library(keras)
library(skmeans)
library(tsfknn)
library(recipes)
library(tibbletime)
library(MLmetrics)

wd <- setwd("C:/Users/joshl/OneDrive/Desktop/Porfolio/Energy Price Forecasting")

###Read the Dataset
data <- read.csv2("time_series_dataset.csv", dec = ".")
data$Data <- as.Date(data$Data, format = '%Y-%m-%d')
print(head(data))


###I divide the dataset into train sets and test sets. The latter will include data from the last two years.
ts <- xts(data$value, start=c(2010,1), order.by=data$Data)  #trasformo i dati in x time series

# data as xts
train <- ts["2010-01-01/2016-12-31"]
test2 <- ts["2017-01-01/2018-12-31"] 
shift <- ts["2016-01-01/2016-12-31"]

#data as ts to produce graphs later
train_ts <- ts(train, start = 1, end = length(train),frequency = frequency(train)) 
test_ts <- ts(test2, start = length(train)+1, end = length(test2)+length(train),frequency = frequency(test2))
shift_ts <- ts(shift, start = length(ts["2010-01-01/2016-01-01"])+1, end = length(shift)+length(ts["2010-01-01/2016-01-01"])+1,frequency = frequency(shift))


#data as df
traindf <- data %>%  dplyr::filter(Data <= ymd("2016-12-31"))
testdf <- data %>%  dplyr::filter(Data > ymd("2016-12-31"))


###As regards non-linear models, the first model we decided to test is a K-Nearest Neighbor algorithm.
###Let's try two different attempts of the algorithm by changing the Multiple-step-ahead strategy: MIMO or recursive. The value of K was chosen based on the root of the training dataset.knn <- knn_forecasting(train_ts, h = 730, lags = 1:730,  k = 50, msas = "MIMO") 
knn <- knn_forecasting(train_ts, h = 730, lags = 1:730,  k = 50, msas = "MIMO") 

ggplot() +
  autolayer(test_ts, series="Data",size=1) +
  autolayer(knn$prediction,series="Fit",size=1.5) +
  xlab("Time") +
  ylab("Value")+
  ggtitle('KNN with MIMO VS real data')

print(MAPE(knn$prediction, test_ts))


###Recursive
knn2 <- knn_forecasting(train_ts, h = 730, lags = 1:365,  k = 50, msas = "recursive") 

ggplot() +
  autolayer(test_ts, series="Data",size=1) +
  autolayer(knn2$prediction,series="Fit",size=1) +
  xlab("Time") +
  ylab("Value")+
  ggtitle('KNN with recursive VS real data')

print(MAPE(knn2$prediction, test_ts))

###The second instance is the one that performs best in terms of MAPE.

###We continue by considering two different types of Recurrent Neural Networks: Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU). The first one we will try is the LSTM.
###First of all, a preprocessing phase on the data is necessary.

data <- read.csv2("time_series_dataset.csv", dec = ".")


data <- data %>%
  mutate(Data = as_date(Data))

data <- data[-c(790, 2251),] #viene eliminato il 29 Feb

train <- data[1:2555,] 
test <- data[2556:3285,] 

df <- bind_rows(
  train %>% add_column(key = "train"),
  test %>% add_column(key = "test"))

#Rescale and center the data

df2 <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

rescaled <- bake(df2, df)


center <- df2$steps[[2]]$means["value"] #Centered Value
scaling  <- df2$steps[[3]]$sds["value"] #Scaled Value


###Network parameters
lag_setting <- 730 
batch_size <- 365          
time_steps <- 1
epochs <- 50


###For RNN networks to exploit data, they must be three-dimensional vectors including the value of the observation, the number of lags and the number of predictors.
train_lag <- rescaled %>%
    mutate(value_lag = lag(value, 365)) %>%
    filter(!is.na(value_lag)) %>%
    filter(key == "train")

x_train <- array(data = train_lag$value_lag, dim = c(length(train_lag$value_lag), time_steps, 1))
y_train <- array(data = train_lag$value, dim = c(length(train_lag$value), time_steps))

test_lag <- rescaled %>%
    mutate(value_lag = lag(value, 365)) %>%
    filter(!is.na(value_lag)) %>%
    filter(key == "test")

x_test <- array(data = test_lag$value_lag, dim = c(length(test_lag$value_lag), time_steps, 1))
y_test <- array(data = test_lag$value, dim = c(length(test_lag$value), time_steps))


###LSTM
mod_lstm <- keras_model_sequential()

mod_lstm %>%
    layer_lstm( units            = 100, 
               input_shape      = c(time_steps, 1), 
               batch_size       = batch_size,
               stateful         = T,
               return_sequences = T) %>%
    layer_lstm(units            = 40,
               stateful         = T,
               return_sequences = T) %>%
    layer_lstm(units            = 40,
               stateful         = T,
               return_sequences = F) %>%
    layer_dense(units = 1) 

mod_lstm %>% 
    compile(loss = 'mae', optimizer = 'adam')

mod_lstm

for (i in 1:epochs) {
    mod_lstm %>% fit(x       = x_train, 
                  y          = y_train, 
                  batch_size = batch_size,
                  epochs     = 1, 
                  verbose    = 1, 
                  shuffle    = FALSE)

    cat("Epoch: ", i)
}


###Predictions on the test set:
pred_scaled_lstm <- mod_lstm %>% 
    predict(x_test, batch_size = batch_size) %>%
    .[,1] 

pred_lstm <- tibble(
    Data   = test_lag$Data,
    value   = (pred_scaled_lstm * scaling + center)^2
) 

ggplot() +
autolayer(ts(test$value), series="Data",size=1) +
autolayer(ts(pred_lstm$value),series="Fit",size=1) +
xlab("Time") +
ylab("Value")+
ggtitle('LSTM VS real data')

print(MAPE(pred_lstm$value, test_ts))


###GRU
mod_gru <- keras_model_sequential()

mod_gru %>%
    layer_gru( units            = 100, 
               input_shape      = c(time_steps, 1), 
               batch_size       = batch_size,
               stateful         = T,
               return_sequences = T) %>%
    layer_gru( units            = 40,
               stateful         = T,
               return_sequences = T) %>%
    layer_gru( units            = 40,
               stateful         = T,
               return_sequences = F) %>%
    layer_dense(units = 1)

mod_gru %>% 
    compile(loss = 'mae', optimizer = 'adam') 

mod_gru

for (i in 1:epochs) {
    mod_gru %>% fit(x        = x_train, 
                  y          = y_train, 
                  batch_size = batch_size,
                  epochs     = 1, 
                  verbose    = 1, 
                  shuffle    = FALSE)

    cat("Epoch: ", i)
    
}




###Predictions on the test set:
pred_scaled_gru <- mod_gru %>% 
    predict(x_test, batch_size = batch_size) %>%
    .[,1] 

pred_gru <- tibble(
    Data   = test_lag$Data,
    value   = (pred_scaled_gru * scaling + center)^2
) 

ggplot() +
autolayer(ts(test$value), series="Data",size=1) +
autolayer(ts(pred_gru$value),series="Fit",size=1) +
xlab("Time") +
ylab("Value")+
ggtitle('GRU VS real data')

print(MAPE(pred_gru$value, test_ts))

##Between the two networks the best one seems to be the LSTM. Overall, the best model among the non-linear ones appears to be the k-nearest neighbors algorithm, with recursive instantiation, which obtains an optimal MAPE value.
##Let's therefore predict the values ​​from 1-Jan-2019 to 30-Nov-2019 with this model:

pred_ML <- knn_forecasting(ts(data$value, frequency = 7), h = 334, lags = 1:365, k = 50, msas = "recursive")

autoplot(ts(test2, start = length(train)+1+430, end = length(test2)+length(train),frequency = frequency(test2)),size=0.7) +
autolayer(ts(pred_ML$prediction, start = (length(train)+length(test2)+1)), series="Forecast",size=1)+
xlab("Time") +
ylab("Value")+
ggtitle('Forecast with KNN')



###We write the expected data into the csv file.
prev <- read.csv("SDMTSA_790544_1.csv", sep = ",", dec = ".")
prev <- as.data.frame(prev)
prev$ML <- pred_ML$prediction

write.csv(prev, file="SDMTSA_790544_1.csv", row.names = FALSE , dec = ".")



###Finally, we show the graphs of the predictions that were made with the different methods:
d <- read.csv("SDMTSA_790544_1.csv", sep = ",", dec = ".")

autoplot(ts(d$ARIMA, start = 1),size=1,color ="steelblue2") +
xlab("Time") +
ylab("Value")+
ggtitle('Forecast with ARIMA(6,0,7)(1,1,1)[7]')
autoplot(ts(d$UCM, start = 1),size=1,color ="orangered") +
xlab("Time") +
ylab("Value")+
ggtitle('Forecast with UCM')
autoplot(ts(d$ML, start = 1),size=1,color = "forestgreen") +
xlab("Time") +
ylab("Value")+
ggtitle('Forecast with KNN')


