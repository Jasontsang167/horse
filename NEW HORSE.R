# 設置工作目錄
setwd("C:/Users/Jason/Desktop/HORSE")

# 載入所需的 R 套件
library(readxl)
library(data.table)
library(dplyr)
library(zoo)
library(corrplot)
library(survival)
library(xgboost)

# 定義輔助函數
lg <- function(x, y) c(y, x[1:(length(x) - 1)])  # 前一場數據
get.mav <- function(bp, n) {  # 計算移動平均值
  require(zoo)
  rollapply(bp, width = n, mean, align = "right", partial = TRUE, na.rm = TRUE)
}

# 加載數據並整理
data <- data.table(read_excel("D:/R_data/Database 2008-2009.xlsx"))
df <- data[order(Run)]

###################### 數據預處理####################
# 定義輔助函數 lg：用於生成滯後值（lag）
lg <- function(x, y) {
  if (length(x) == 1) {
    return(y)  # 如果分組中只有一行，返回 y（通常是 NA）
  }
  c(y, x[1:(length(x) - 1)])
}

# 計算班次校準值（Class.Cal）
df$Class.Cal <- ifelse(df$Class == 7, 1,
                       ifelse(df$Class == 1, 2,
                              ifelse(df$Class == 2, 3,
                                     ifelse(df$Class == 3, 4,
                                            ifelse(df$Class == 4, 5,
                                                   ifelse(df$Class == 5, 6,
                                                          ifelse(df$Class == 6, 7, NA)))))))

# 按照馬匹名稱（Name）分組計算滯後班次（LagClass）
df <- df[, LagClass := lg(Class.Cal, NA), by = Name]

# 計算班次變化值（L1Class）
df$L1Class <- ifelse(is.na(df$LagClass), 0, df$Class.Cal - df$LagClass)

# 檢查輸出
table(is.na(df$LagClass))  # 檢查 LagClass 中是否存在 NA 值
head(df[, .(Name, Class.Cal, LagClass, L1Class)])  # 查看部分輸出結果
# 計算過去比賽數據
df <- df[, L1FinPos := lg(FinPos, NA), by = Name]
Avg.FP <- mean(df$L1FinPos, na.rm = TRUE)
df <- df[, AVG4FinPos := as.numeric(na.fill(get.mav(L1FinPos, 4), Avg.FP)), by = Name]
df <- df[, L1Rating := ifelse(lg(Rating, 0) == 0 | Rating == 0, 0, Rating - lg(Rating, 0)), by = Name]
df <- df[, L1HrWt := ifelse(is.na(HrWt - lg(HrWt, NA)), 0, HrWt - lg(HrWt, NA)), by = Name]

# 計算天數差異
df$date1 <- as.Date(df$Date, origin = "1899-12-30")
df <- df[, date2 := lg(date1, NA), by = Name]
df$date3 <- na.fill(as.numeric(df$date1 - df$date2), 365)
df$LastRun <- ifelse(df$date3 > 365, 365, df$date3)

# 二元分類結果
df <- df[, FO := ifelse(FinPos == 1, 1, 0)]

# 選擇所需特徵
var <- c("Run", "Date", "HrNO", "Age", "HrWt", "WtCr", "L1Class", "AVG4FinPos", "L1Rating", "L1HrWt", "LastRun", "FO", "FinPos", "FinOdd")
data.pick <- select(df, one_of(var))
data.pick$HrNO <- as.factor(data.pick$HrNO)

# 訓練與測試數據集劃分
train <- data.pick[data.pick$Date <= "2009-12-31", ]
test <- data.pick[data.pick$Date > "2009-12-31", ]
train.cl <- train[complete.cases(train), ]
test.cl <- test[complete.cases(test), ]

# 條件邏輯迴歸模型
m <- model.matrix(~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun - 1, data = test.cl)
fit <- clogit(FO ~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun + strata(Run), train.cl)
pp <- exp(m %*% coef(fit))  # 預測分數
pps <- ave(pp, test.cl$Run, FUN = sum)
pred.cl <- pp / pps  # 勝出概率

# XGBoost 模型（多分類名次預測）


# 準備訓練和測試數據
mtrain_xgb <- data.matrix(train.cl[,-c(1:3, 12, 13, 14)])  # 提取訓練特徵
mtest_xgb <- data.matrix(test.cl[,-c(1:3, 12, 13, 14)])   # 提取測試特徵
output_vector <- as.numeric(train.cl$FinPos) - 1          # 名次轉為數字類別（0 開始）

# 訓練 XGBoost 模型
bst <- xgboost(
  data = mtrain_xgb, 
  label = output_vector, 
  nround = 50, 
  objective = "multi:softprob", 
  num_class = max(output_vector) + 1
)

# 預測每匹馬的名次概率
pred_probs <- predict(bst, mtest_xgb)
pred_matrix <- matrix(pred_probs, ncol = max(output_vector) + 1, byrow = TRUE)
pred_rank <- max.col(pred_matrix)  # 每匹馬最高概率對應的名次

# 整合結果
# 確保 pred.cl 已經計算完成
if (!exists("pred.cl")) {
  stop("pred.cl 尚未計算，請先執行條件邏輯回歸模型生成 pred.cl")
}
test.cl$Predicted_Probability <- pred.cl  # 添加勝出概率
test.cl$Predicted_Rank <- pred_rank       # 添加預測名次

# 評估模型表現
# Top-1 準確率：預測名次與實際名次完全匹配的比例
accuracy_top1 <- mean(test.cl$FinPos == test.cl$Predicted_Rank)
cat("Top-1 名次準確率: ", accuracy_top1, "\n")

# Top-4 準確率：實際名次在預測的前 4 名中的比例
accuracy_top4 <- mean(
  sapply(seq_len(nrow(pred_matrix)), function(i) test.cl$FinPos[i] %in% order(pred_matrix[i, ], decreasing = TRUE)[1:4])
)
cat("Top-4 名次準確率: ", accuracy_top4, "\n")

# 查看結果
result_table <- test.cl[, c("Run", "HrNO", "FinPos", "Predicted_Rank", "Predicted_Probability")]
head(result_table)





####這段代碼是用來進行 Extreme Gradient Boosting (XGBoost) 的二元分類模型訓練和預測
# 1. 數據轉換
mtrain.xgb <- data.matrix(train.xgb[, -c(1:3, 12, 13, 14)])  # 移除不必要列
mtest.xgb <- data.matrix(test.xgb[, -c(1:3, 12, 13, 14)])    # 測試特徵矩陣
output_vector1 <- train.xgb[, "FO"] == 1  # 訓練集目標標籤
output_vector2 <- test.xgb[, "FO"] == 1   # 測試集目標標籤

# 2. 模型訓練
bst <- xgboost(
  data = mtrain.xgb,
  label = output_vector1,
  nround = 50,               # 訓練迭代次數
  objective = "binary:logistic",  # 二元分類
  eta = 0.1,                 # 學習率
  max_depth = 6,             # 最大樹深
  subsample = 0.8,           # 子樣本比例
  colsample_bytree = 0.8     # 每棵樹的特徵比例
)

# 3. 預測測試集勝出概率
pred.xgb <- predict(bst, mtest.xgb)  # 返回每匹馬勝出的概率
test.xgb$Predicted_Probability <- pred.xgb

# 4. 特徵重要性分析
name.xgb <- colnames(mtrain.xgb)
importance.xgb <- xgb.importance(feature_names = name.xgb, model = bst)

# 5. 輸出重要特徵和圖形
print(head(importance.xgb, 10))  # 顯示最重要的 10 個特徵
xgb.plot.importance(importance_matrix = importance.xgb)

# 6. 樹模型可視化
xgb.plot.tree(model = bst, trees = 0:1, render = TRUE)
