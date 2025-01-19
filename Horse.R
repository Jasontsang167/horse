install.packages(c(
  "data.table", "dplyr", "zoo", "ggplot2", 
  "corrplot", "xgboost", "ranger", "e1071", "survival", "readxl"
))



# 加載必要套件
library(readxl)
library(data.table)
library(dplyr)
library(zoo)
library(corrplot)
library(ggplot2)
library(survival)
library(xgboost)
library(ranger)
library(e1071)
install.packages("data.table", dependencies = TRUE, type = "source")

# 設定工作路徑
setwd("/Users/Jason/Documents")

# 自定義函數
lg <- function(x, y) {
  if (length(x) == 1) {
    return(NA)  # 如果分組只有一行，返回 NA
  }
  c(y, x[1:(length(x) - 1)])
}
get.mav <- function(bp, n) {
  require(zoo)
  rollapply(bp, width = n, mean, align = "right", partial = TRUE, na.rm = TRUE)
}

# 加載數據
data <- data.table(read_excel("D:/R_data/Database 2008-2009.xlsx"))
df <- data[order(Run)]

# 特徵工程
df$Class.Cal <- ifelse(df$Class == 7, 1, ifelse(df$Class == 1, 2, ifelse(df$Class == 2, 3, 
                                                                         ifelse(df$Class == 3, 4, ifelse(df$Class == 4, 5, ifelse(df$Class == 5, 6, 7))))))
df <- df[, LagClass := lg(Class.Cal, NA), by = "Name"]
df$L1Class <- ifelse(is.na(df$LagClass), 0, df$Class.Cal - df$LagClass)
df <- df[, L1FinPos := lg(FinPos, NA), by = "Name"]
Avg.FP <- mean(df$L1FinPos, na.rm = TRUE)
df <- df[, AVG4FinPos := as.numeric(na.fill(get.mav(L1FinPos, 4), Avg.FP)), by = "Name"]
df <- df[, L1Rating := ifelse(lg(Rating, 0) == 0 | Rating == 0, 0, Rating - lg(Rating, 0)), by = "Name"]
df <- df[, L1HrWt := ifelse(is.na(HrWt - lg(HrWt, NA)), 0, HrWt - lg(HrWt, NA)), by = "Name"]
df$date1 <- as.Date(df$Date, origin = "1899-12-30")
df <- df[, date2 := lg(date1, NA), by = "Name"]
df$date3 <- na.fill(as.numeric(df$date1 - df$date2), 365)
df$LastRun <- ifelse(df$date3 > 365, 365, df$date3)
df$FO <- ifelse(df$FinPos == 1, 1, 0)

# 選擇特徵
var <- c("Run", "Date", "HrNO", "Age", "HrWt", "WtCr", "L1Class", "AVG4FinPos", "L1Rating", "L1HrWt", "LastRun", "FO", "FinPos", "FinOdd")
data.pick <- df[, ..var]
data.pick$HrNO <- as.factor(data.pick$HrNO)

# 資料分割
train <- data.pick[data.pick$Date <= "2009-12-31"]
test <- data.pick[data.pick$Date > "2009-12-31"]

# 條件邏輯回歸模型
fit <- clogit(FO ~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun + strata(Run), data = train)
test_matrix <- model.matrix(~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun - 1, data = test)
pred_prob <- exp(test_matrix %*% coef(fit))
pred_prob <- pred_prob / ave(pred_prob, test$Run, FUN = sum)

# XGBoost 模型
train_matrix <- data.matrix(train[, -c(1:3, 12, 13, 14)])
test_matrix <- data.matrix(test[, -c(1:3, 12, 13, 14)])
train_label <- train$FO

xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 50, max.depth = 5, eta = 0.1, objective = "binary:logistic")
xgb_pred <- predict(xgb_model, test_matrix)
xgb_importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_importance)

# 隨機森林模型
rf_model <- ranger(FO ~ ., data = train, num.trees = 100, mtry = 5, probability = TRUE)
rf_pred <- predict(rf_model, data = test)$predictions[, 2]

# 支持向量機模型
svm_model <- svm(FinPos ~ ., data = train[, -c(1:3, 12, 13, 14)], type = "eps-regression", probability = TRUE)
svm_pred <- predict(svm_model, test[, -c(1:3, 12, 13, 14)])

# 集成學習
stack_features <- data.frame(Logistic = pred_prob, XGBoost = xgb_pred, RandomForest = rf_pred, SVM = svm_pred)
stack_model <- xgboost(data = as.matrix(stack_features), label = test$FO, nrounds = 50, objective = "binary:logistic")
stack_pred <- predict(stack_model, as.matrix(stack_features))

# 評估準確率和回報率
accuracy <- mean(round(stack_pred) == test$FO)
return_rate <- sum((round(stack_pred) == 1 & test$FO == 1) * test$FinOdd) / nrow(test)
cat("Accuracy:", accuracy, "\nReturn Rate:", return_rate, "\n")

# 繪製特徵重要性圖
plot(rf_model$variable.importance, main = "Random Forest Feature Importance", col = "blue")

