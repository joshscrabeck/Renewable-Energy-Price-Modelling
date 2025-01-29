
knitr::opts_chunk$set(echo = TRUE)

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
library(timeSeries)

wd <- setwd("C:/Users/joshl/OneDrive/Desktop/Porfolio/Energy Price Forecasting")

data <- read.csv2("time_series_dataset.csv", dec = ".")
data$Data <- as.Date(data$Data, format = '%Y-%m-%d')
print(head(data))


BoxCox.lambda(data$value)

ggplot(data,aes(x=Data, y=value)) +
  geom_line() +
  stat_smooth(method = "lm", se = T, col="red", level=0.99)+
  xlab("Time") +
  ylab("Value")





########################## ARIMA ########################## 

ts <- xts(data$value, start=c(2010,1), order.by=data$Data)

par(mfrow=c(1,2))
Acf(ts,lag.max = 80) 
Pacf(ts,lag.max = 80)

ggtsdisplay(diff(ts, 7),points = FALSE,lag.max = 80)


train <- ts["2010-01-01/2016-12-31"]
test <- ts["2017-01-01/2018-12-31"] 
shift <- ts["2016-01-01/2016-12-31"]


train_ts <- ts(train, start = 1, end = length(train),frequency = frequency(train)) 
test_ts <- ts(test, start = length(train)+1, end = length(test)+length(train),frequency = frequency(test))
shift_ts <- ts(shift, start = length(ts["2010-01-01/2016-01-01"])+1, end = length(shift)+length(ts["2010-01-01/2016-01-01"])+1,frequency = frequency(shift))


#data as df
traindf <- data %>%  dplyr::filter(Data <= ymd("2016-12-31"))
testdf <- data %>%  dplyr::filter(Data > ymd("2016-12-31"))

rbind(data.frame(Date=index(train), value=coredata(train), set="train"),
      data.frame(Date=index(test), value=coredata(test), set="test")) %>%
  ggplot(aes(x=Date, y=value)) +
  geom_line(aes(colour = set)) +
  stat_smooth(method = "lm", se = T, col="blue", level=0.99)

cat("percentage of observations in the training set: ", length(train)/length(ts))
cat("\npercentage of observations in the training set: ", length(test)/length(ts))


###Let's try considering a model that takes into account the seasonality of period 7 of the series
mod1 <- Arima(train, c(0,0,0), list(order=c(1,1,1), period=7), lambda = "auto")
ggtsdisplay(mod1$residuals, lag.max = 80)
summary(mod1)


###The model coefficients appear to be significant; furthermore, from the correlations of the model residuals, there appear to be various components to take into consideration. In particular, the ACF would suggest the presence of an MA(6) component given the first 6 lags. There are also other components that we will look for later.
###I try to proceed "vertically": models with different parameters in terms of log-likelihood AIC and BIC are compared. The one that turns out to be better will be used as a basis on which to continue the work
###ARIMA models with a large number of coefficients in double loops give errors 

for (j in 1:7){
  mod<-Arima(train, c(j,0,5),list(order = c(1,1,1), period = 7),lambda = "auto")
  cat("\nModel AR(",j,") MA( 5 )  --- log-lik:",mod$loglik,"--- aic:",mod$aic)
}

for (j in 1:7){
  mod<-Arima(train, c(j,0,6),list(order = c(1,1,1), period = 7),lambda = "auto")
  cat("\nModel AR(",j,") MA( 6 )  --- log-lik:",mod$loglik,"--- aic:",mod$aic)
}

for (j in 1:7){
  mod<-Arima(train, c(j,0,7),list(order = c(1,1,1), period = 7),lambda = "auto")
  cat("\nModel AR(",j,") MA( 7 )  --- log-lik:",mod$loglik,"--- aic:",mod$aic)
}

###The ARMA(6,7) model is the optimal one both in terms of log-likelihood and in terms of AIC.


###So let's continue with ARIMA(6,0,7)(1,1,1)[7]. 

mod2 <- Arima(train, c(6,0,7), list(order=c(1,1,1), period=7), lambda = "auto")
ggtsdisplay(mod2$residuals, lag.max = 80)
summary(mod2)

##Now, from the analysis of the correlograms, it is above all lag 20 that is significantly different from 0.
#Let's try to analyze the roots of the characteristic equation of the model:
autoplot(mod2)
Mod(1/polyroot(c(1,-mod2$coef[1:6])))

##From the analysis, no integration of the non-seasonal component appears to be necessary.

##:For completeness, let's try to see what happens by proceeding with an integration
mod3 <- Arima(train, c(6,1,7), list(order=c(1,1,1), period=7), lambda = "auto")
par(mfrow=c(1,2))
Acf(mod3$residuals,lag.max = 80) 
Pacf(mod3$residuals,lag.max = 80)
summary(mod3)

##As we had already surmised from the root analysis, it also appears from the correlograms and goodness measures that the addition of integration worsens the model.
##Let's try to make a first plot of how the simple ARIMA(6,0,7)(1,1,1)[7] model predicts the test set data.

pred <- forecast(mod2, h=730)

autoplot(shift_ts) +
  autolayer(pred,series="Fit") +
  autolayer(test_ts, series="Data") +
  xlab("Time") +
  ylab("Value")+
  ggtitle('ARIMA(6,0,7)(1,1,1)[7] VS real data')

k <- 24
freq <- outer(1:nrow(data), 1:k)*2*pi/365.25    

cos <- cos(freq)                   
colnames(cos) <- paste("cos", 1:k)
sin <- sin(freq)                   
colnames(sin) <- paste("sin", 1:k)
reg <- as.matrix(cbind(cos,sin))


mod_reg <- Arima(train, c(6,0,7), list(order=c(1,1,1), period=7), 
              xreg=reg[1:(length(train)),], include.constant = T, lambda = "auto", method = "CSS")

ggtsdisplay(mod_reg$residuals, lag.max = 80)
summary(mod_reg)

###A further step consists in considering Italian holidays. This is done thanks to the RQuantLib library

load_quantlib_calendars("Italy", from = "2010-01-01", to = "2018-12-31") 

k <- 24

freq <- outer(1:nrow(data), 1:k)*2*pi/365.25    

cos <- cos(freq)                   
colnames(cos) <- paste("cos", 1:k)
sin <- sin(freq)                   
colnames(sin) <- paste("sin", 1:k)
reg_hol <- as.matrix(cbind(cos,sin))


data.frame(Data=data$Data) %>%
  mutate(holiday = as.numeric(!is.bizday(Data, "QuantLib/Italy")))%>%
  select(-starts_with("Data")) %>% 
  cbind(reg_hol) %>% 
  as.matrix() -> reg_hol


mod_reg_holidays <- Arima(train, c(6,0,7), list(order=c(1,1,1), period=7), 
                          xreg=reg_hol[1:(length(train)),], include.constant = T, lambda = "auto", method = "CSS")

ggtsdisplay(mod_reg_holidays$residuals, lag.max = 80)
summary(mod_reg_holidays)

###This second model, which also considers holidays, is better than the first both in terms of log-likelihood and AIC and MAPE on the training set.

###Let's now compare the two models on the test set graphically and in terms of MAPE:

pred <- forecast(mod_reg, h=730,
                 xreg=reg[(length(train)+1):(length(train)+730),])


ggplot() +
  autolayer(pred,series="Fit",size=1) +
  autolayer(test_ts, series="Data",size=1) +
  xlab("Time") +
  ylab("Value")+
  ggtitle('ARIMA(6,0,7)(1,1,1)[7] with harmonics VS real data')

print(MAPE(pred$mean, test_ts))

pred <- forecast(mod_reg_holidays, h=730,
                 xreg=reg_hol[(length(train)+1):(length(train)+730),])


ggplot() +
  autolayer(pred,series="Fit",size=1) +
  autolayer(test_ts, series="Data",size=1) +
  xlab("Time") +
  ylab("Value")+
  ggtitle('ARIMA(6,0,7)(1,1,1)[7] with harmonics and holiday dummies VS real data')

print(MAPE(pred$mean, test_ts))

###From the comparison between the two models it appears that the one with only harmonics is able to fit the test set data better. The one who also considers holidays probably suffers from overfitting on the training set.


###We therefore consider the first as the best and use it to make forecasts from 1-Jan-2019 to 30-Nov-2019:

k <- 24
freq <- outer(1:(nrow(data)+365), 1:k)*2*pi/365.25

cos <- cos(freq)                   
colnames(cos) <- paste("cos", 1:k)
sin <- sin(freq)                   
colnames(sin) <- paste("sin", 1:k)
reg_fin <- as.matrix(cbind(cos,sin))


mod_reg_fin <- Arima(ts, c(6,0,7), list(order=c(1,1,1), period=7), 
                     xreg=reg_fin[1:(length(ts)),], include.constant = T, lambda = "auto", method = "CSS")

pred <- forecast(mod_reg_fin, h=334, xreg=reg_fin[(length(ts)+1):(length(ts)+334),])
pred_Arima<- pred$mean

autoplot(pred,include=300)+
  xlab("Time") +
  ylab("Value")+
  ggtitle('Forecast with ARIMA(6,0,7)(1,1,1)[7]')





########################## UCM ########################## 


###We will test different UCM models: a first model with LLT plus seasonal regressors, an ILLT and an LLT estimated as RW.
###The first UCM model is therefore a Local Linear Trend plus dummies that consider the weekly seasonality and harmonics for the annual one.

ytrain <- as.numeric(train)
mod1 <- SSModel(ytrain ~ SSMtrend(2, list(NA,NA)) +
                  SSMseasonal(7, NA, "dummy") +
                  SSMseasonal(365, NA, "trig",
                              harmonics = 1:24),
                H = NA)

#Initial Conditions
vary <- var(ytrain, na.rm = TRUE)
mod1$P1inf <- mod1$P1inf * 0
mod1$a1[1] <- mean(ytrain, na.rm = TRUE)
diag(mod1$P1) <- vary



# Initial values for the variances we have to estimate
init <- numeric(5)
init[1] <- log(vary/10) 
init[2] <- log(vary/10) 
init[3] <- log(vary/100)
init[4] <- log(vary/100)
init[5] <- log(vary/10) 

#Updating function
update_fun <- function(pars, model){
  model$Q[1, 1, 1] <- exp(pars[1])
  model$Q[2, 2, 1] <- exp(pars[2])
  model$Q[3, 3, 1] <- exp(pars[3])
  diag(model$Q[4:51,4:51, 1]) <- exp(pars[4])
  model$H[1, 1, 1] <- exp(pars[5])
  model
}

fit1 <- fitSSM(mod1, init, update_fun, control = list(maxit = 1000))
cat("Convergence Code = ",fit1$optim.out$convergence)
cat("\nMAPE on train: ",MAPE(fitted(fit1$model),ytrain))

###0 indicates convergence of the algorithm.

smo1 <- KFS(fit1$model, smoothing = "state")
plot(timeSeries(ytrain, as.Date("2010-01-01") + 0:(length(ytrain)-1)))
lines(timeSeries(smo1$alphahat[, "level"], as.Date("2010-01-01") + 0:(length(ytrain)-1)),col = "red")

###Let's make one-step-ahead predictions on the test set:

y <- c(ytrain, rep(NA,length(test)))

mod1_test <- SSModel(y ~  SSMtrend(2, list(fit1$model$Q[1,1,1],fit1$model$Q[2,2,1])) +
                      SSMseasonal(7, fit1$model$Q[3,3,1], "dummy") +
                      SSMseasonal(365, fit1$model$Q[4, 4, 1], "trig",
                      harmonics = 1:24),
                      H = fit1$model$H)

mod1_test$a1 <- fit1$model$a1
mod1_test$P1 <- fit1$model$P1
mod1_test$P1inf <- fit1$model$P1inf

# Smoothing of state and signal variables
smo1_ <- KFS(mod1_test, smoothing = c("state", "signal"))


pred <- smo1_$muhat[(length(train)+1):(length(ts)), 1]

ggplot() +
autolayer(test_ts,series="Data",size=1) + 
autolayer(ts(pred, start = length(ytrain)+1), series="Fit",size=1)+
xlab("Time") +
ylab("Value")+
ggtitle('Simple LLT VS real data')

print(MAPE(pred, test_ts))



###the second model UCM and Integrated Local Linear Trend:
ytrain <- as.numeric(train)
mod2 <- SSModel(ytrain ~ SSMtrend(2, list(0,NA)) +
                      SSMseasonal(7, NA, "dummy") +
                      SSMseasonal(365, NA, "trig",
                      harmonics = 1:24),
                      H = NA)

#initial conditions
vary <- var(ytrain, na.rm = TRUE)
mod2$P1inf <- mod2$P1inf * 0
mod2$a1[1] <- mean(ytrain, na.rm = TRUE)
diag(mod2$P1) <- vary



# Initial values for the variances we have to estimate
init <- numeric(5)
init[1] <- 0
init[2] <- log(vary/10) 
init[3] <- log(vary/100)
init[4] <- log(vary/100)
init[5] <- log(vary/10)

#updating function
update_fun <- function(pars, model){
    model$Q[1, 1, 1] <- exp(pars[1])
    model$Q[2, 2, 1] <- exp(pars[2])
    model$Q[3, 3, 1] <- exp(pars[3])
    diag(model$Q[4:51,4:51, 1]) <- exp(pars[4])
    model$H[1, 1, 1] <- exp(pars[5])
    model
}

fit2 <- fitSSM(mod2, init, update_fun, control = list(maxit = 1000))
cat("Convergence Code = ",fit2$optim.out$convergence)
cat("\nMAP on train: ",MAPE(fitted(fit2$model),ytrain))


smo2 <- KFS(fit2$model, smoothing = "state")
plot(timeSeries(ytrain, as.Date("2010-01-01") + 0:(length(ytrain)-1)))
lines(timeSeries(smo2$alphahat[, "level"], as.Date("2010-01-01") + 0:(length(ytrain)-1)),col = "red")



###One-step-ahead predictions on the test set:
y <- c(ytrain, rep(NA,length(test)))

mod2_test <- SSModel(y ~  SSMtrend(2, list(0,fit2$model$Q[2,2,1])) +
                      SSMseasonal(7, fit2$model$Q[3,3,1], "dummy") +
                      SSMseasonal(365, fit2$model$Q[4, 4, 1], "trig",
                      harmonics = 1:24),
                      H = fit2$model$H)

mod2_test$a1 <- fit2$model$a1
mod2_test$P1 <- fit2$model$P1
mod2_test$P1inf <- fit2$model$P1inf

# Smoothing of state and signal variables
smo1_ <- KFS(mod2_test, smoothing = c("state", "signal"))


pred <- smo1_$muhat[(length(train)+1):(length(ts)), 1]

ggplot() +
autolayer(test_ts,series="Data",size=1) + 
autolayer(ts(pred, start = length(ytrain)+1), series="Fit",size=1)+
xlab("Time") +
ylab("Value")+
ggtitle('ILLT VS real data')

print(MAPE(pred, test_ts))



##for the torch model consider a Random Walk:
ytrain <- as.numeric(train)
mod3 <- SSModel(ytrain ~ SSMtrend(1, NA) +
                      SSMseasonal(7, NA, "dummy") +
                      SSMseasonal(365, NA, "trig",
                      harmonics = 1:24),
                      H = NA)


vary <- var(ytrain, na.rm = TRUE)
mod3$P1inf <- mod3$P1inf * 0
mod3$a1[1] <- mean(ytrain, na.rm = TRUE)
diag(mod3$P1) <- vary




init <- numeric(5)
init[1] <- log(vary/10) 
init[2] <- log(vary/100)
init[3] <- log(vary/100)
init[4] <- log(vary/10) 


update_fun <- function(pars, model){
    model$Q[1, 1, 1] <- exp(pars[1])
    model$Q[2, 2, 1] <- exp(pars[2])
    diag(model$Q[3:50, 3:50, 1]) <- exp(pars[3])
    model$H[1, 1, 1] <- exp(pars[4])
    model
}


fit3 <- fitSSM(mod3, init, update_fun, control = list(maxit = 1000))
cat("Convergence Code = ",fit3$optim.out$convergence)
cat("\nMAP on train: ",MAPE(fitted(fit3$model),ytrain))



smo3 <- KFS(fit3$model, smoothing = "state")
plot(timeSeries(ytrain, as.Date("2010-01-01") + 0:(length(ytrain)-1)))
lines(timeSeries(smo3$alphahat[, "level"], as.Date("2010-01-01") + 0:(length(ytrain)-1)),col = "red")

y <- c(ytrain, rep(NA,length(test)))

mod3_test <- SSModel(y ~  SSMtrend(1, fit3$model$Q[1,1,1]) +
                      SSMseasonal(7, fit3$model$Q[2,2,1], "dummy") +
                      SSMseasonal(365, fit3$model$Q[3, 3, 1], "trig",
                      harmonics = 1:24),
                      H = fit3$model$H)

mod3_test$a1 <- fit3$model$a1
mod3_test$P1 <- fit3$model$P1
mod3_test$P1inf <- fit3$model$P1inf

smo1_ <- KFS(mod3_test, smoothing = c("state", "signal"))


pred <- smo1_$muhat[(length(train)+1):(length(ts)), 1]

ggplot() +
autolayer(test_ts,series="Data",size=1) + 
autolayer(ts(pred, start = length(ytrain)+1), series="Fit",size=1)+
xlab("Time") +
ylab("Value")+
ggtitle('RW VS real data')

print(MAPE(pred, test_ts))

y <- c(as.numeric(ts), rep(NA,334))

mod_ucm_fin <- SSModel(y ~  SSMtrend(1, fit3$model$Q[1,1,1]) +
                      SSMseasonal(7, fit3$model$Q[2,2,1], "dummy") +
                      SSMseasonal(365, fit3$model$Q[3, 3, 1], "trig",
                      harmonics = 1:24),
                      H = fit3$model$H)

mod_ucm_fin$a1 <- fit3$model$a1
mod_ucm_fin$P1 <- fit3$model$P1
mod_ucm_fin$P1inf <- fit3$model$P1inf

smo_fin <- KFS(mod_ucm_fin, smoothing = c("state", "signal"))

pred_ucm <- smo_fin$muhat[(length(train)+length(test)+1):(length(y)), 1]

autoplot(ts(test, start = length(train)+1+430, end = length(test)+length(train),frequency = frequency(test)),size=0.7) +
autolayer(ts(pred_ucm, start = (length(train)+length(test)+1)), series="Forecast",size=1)+
xlab("Time") +
ylab("Value")+
ggtitle('Forecast with RW')


dataframe_pred <- data.frame(matrix(ncol = 4, nrow = 334))
colnames(dataframe_pred) <- c("Data", "ARIMA", "UCM", "ML")
dataframe_pred$Data <- seq(as.Date("2019-01-01"),as.Date("2019-11-30"),1)
dataframe_pred$ARIMA <- pred_Arima
dataframe_pred$UCM <- pred_ucm

write.csv(dataframe_pred, file="Final_Output.csv", sep = ";", dec = ".", row.names=FALSE)




