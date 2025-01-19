setwd("C:/Users/Jason/Desktop/HORSE")
library(readxl)
library(data.table)
library(plyr)
library(dplyr)
library(zoo)
library(ggplot2)
library(corrplot)
library(survival)
require(xgboost)
library(ranger)
library(e1071)
library(DiagrammeR)

#Lag Function for previous race定義輔助函數
lg <- function(x,y)c(y, x[1:(length(x)-1)])
#Calculating Moving Average計算移動平均值，用於生成過去數次比賽的平均表現
get.mav <- function(bp,n){require(zoo)
  rollapply(bp, width=n,mean,align="right",partial=TRUE,na.rm=TRUE)}

##########載入和整理數據
data<-data.table(read_excel("D:/R_data/Database 2008-2009.xlsx"))
df<-data
df<-df[with(df, order(Run)), ]
##########Class calibration計算每匹馬的班次升降變化
#Original: Elite / International - 7, New Horse - 6, Class 1 - 5
#Calibrated: Elite / Inter => 1, Class 1-5 => 2 - 6, New Horse => 7
df$Class.Cal<-ifelse(df$Class==7,1,ifelse(df$Class==1,2,ifelse(df$Class==2,3,ifelse(df$Class==3,4,ifelse(df$Class==4,5,ifelse(df$Class==5,6,ifelse(df$Class==6,7,NA)))))))
##########Upgrade or Downgrade of Class計算每匹馬的班次升降變化。
lg <- function(x, y) {
  if (length(x) == 1) {
    return(NA)  # 如果分組中只有一行，返回 NA
  }
  c(y, x[1:(length(x) - 1)])
}

df <- df[, LagClass := lg(Class.Cal, NA), by = c("Name")]
df$L1Class <- ifelse(is.na(df$LagClass), 0, df$Class.Cal - df$LagClass)
##########Finishing Position from last race計算過去比賽表現計算每匹馬過去四場比賽的平均完成名次。（過去四場數據）不足，將用 Avg.FP 填充缺失值
df<-df[,L1FinPos := lg(FinPos,NA), by = c("Name")]
Avg.FP<-mean(df$L1FinPos,na.rm=T)
df<-df[,AVG4FinPos:=as.numeric(na.fill(get.mav(L1FinPos,4),Avg.FP)),by=c("Name")]
##########Difference from previous rating
df<-df[,L1Rating := ifelse(lg(Rating,0)==0|Rating==0,0,Rating-lg(Rating,0)), by = c("Name")]
##########Difference from previous horse weight計算上一場比賽的評分和馬匹體重變化
df<-df[,L1HrWt := ifelse(is.na(HrWt-lg(HrWt,NA)),0,HrWt-lg(HrWt,NA)), by = c("Name")]
##########Day difference from previous race計算距離上一次比賽的天數
df$date1<-as.Date(df$Date,origin = "1899-12-30")
df<-df[,date2:=lg(date1,NA),by=c("Name")]
df$date3<-na.fill(as.numeric(df$date1-df$date2),365)
df$LastRun<-ifelse(df$date3>365,365,df$date3)
##########Transform Finishing Position to Binary result將比賽結果轉換為二元分類：1（勝出）或 0（未勝出）
df<-df[,FO:=ifelse(FinPos==1,1,0)]


##########Variable to pick and subset data
var<-c("Run","Date","HrNO","Age","HrWt","WtCr","L1Class","AVG4FinPos","L1Rating",
       "L1HrWt","LastRun","FO","FinPos","FinOdd")
data.pick<-select(df,one_of(var))
##########Convert feature to factor
data.pick$HrNO<-as.factor(data.pick$HrNO)
##########Summary of data數據可視化與檢查
summary(data.pick[,4:7])
##########Checking NA value in data set檢查數據摘要和缺失值情況
na.num<-sum(is.na(data.pick))
print(paste0("Number of Not Available items in data set: ",na.num))
##########Data Visualization with Boxplot for variables對特徵進行分布可視化，檢查異常值
par(mfrow=c(1,4))#set margin
boxplot(data.pick$Age,col="red")
boxplot(data.pick$HrWt,col="blue")
hist(data.pick$LastRun,col="yellow")
hist(data.pick$L1Class,col="green")

##########Correlation Plot訓練與測試數據劃分
par(mfrow=c(1,1))
correlations <- cor(data.pick[,4:14])# calculate correlations
corrplot(correlations, method="circle")# create correlation plot

train<-data.pick[data.pick$Date<="2009-12-31", ]
test<-data.pick[data.pick$Date>"2009-12-31", ]

train.cl<-train
test.cl<-test
train.xgb<-train
test.xgb<-test
train.rf<-train
test.rf<-test
train.svm<-train
test.svm<-test
############################自己加的#########################################
# 清理數據，確保完整性
train.cl <- train.cl[complete.cases(train.cl), ]
test.cl <- test.cl[complete.cases(test.cl), ]
# 檢查特徵對應
colnames(train.cl)
colnames(test.cl)
# Step 1: 現有代碼（計算勝出概率）
m <- model.matrix(~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun - 1, data=test.cl)  #模型矩陣每行對應一匹馬在該場比賽中的特徵值
fit <- clogit(FO ~ Age + HrWt + WtCr + L1Class + AVG4FinPos + L1Rating + L1HrWt + LastRun + strata(Run), train.cl)  # 條件邏輯迴歸模型
pp <- exp(m %*% coef(fit))  #該值代表每匹馬匹相對的“勝出可能性分數”每個數字越大，表示該匹馬基於模型參數的相對得分越高 # 每匹馬的預測得分
pps <- ave(pp, test.cl$Run, FUN = sum)   #pps 是每場比賽中所有馬匹的得分總和 # 每場比賽得分總和
pred.cl <- pp / pps  #是每匹馬匹的勝出概率PP/PPS 越接近 1 表示該匹馬勝出的可能性越高 # 每匹馬的勝出概率

# Step 2: 多分類名次預測（使用 XGBoost）
library(xgboost)

# 準備數據集
mtrain_xgb <- data.matrix(train.cl[,-c(1:3,12,13,14)])  # 提取訓練特徵
mtest_xgb <- data.matrix(test.cl[,-c(1:3,12,13,14)])   # 提取測試特徵
output_vector <- as.numeric(train.cl$FinPos) - 1       # 名次轉為數字類別（0 開始）

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

# 將概率轉為名次預測
pred_matrix <- matrix(pred_probs, ncol = max(output_vector) + 1, byrow = TRUE)
pred_rank <- max.col(pred_matrix)  # 每匹馬最高概率對應的名次

# Step 3: 整合結果
test.cl$Predicted_Probability <- pred.cl       # 加入勝出概率
test.cl$Predicted_Rank <- pred_rank            # 加入預測名次

# Step 4: 評估模型表現
# 比較實際名次與預測名次
accuracy_top1 <- mean(test.cl$FinPos == test.cl$Predicted_Rank)
cat("Top-1 名次準確率: ", accuracy_top1, "\n")

# 如果需要分析 Top-N 準確率（例如，是否預測名次在前 4 名中）
accuracy_top4 <- mean(test.cl$FinPos %in% apply(pred_matrix, 1, order, decreasing = TRUE)[1:4, ])
cat("Top-4 名次準確率: ", accuracy_top4, "\n")

# 打印測試集的結果
head(test.cl[, c("Run", "HrNO", "FinPos", "Predicted_Rank", "Predicted_Probability")])



# 檢查結果
length(pp) == nrow(test.cl)  # 是否一致
summary(pred.cl)  # 檢查概率分布
#如果有缺失值請清理數據
test.cl <- test.cl[complete.cases(test.cl), ]






##########Using Conditional Logistic Regression##########
##########Modelling Conditional Logistic Regression模型訓練與預測 條件邏輯迴歸
fit<-clogit(FO~Age+HrWt+WtCr+L1Class+AVG4FinPos+L1Rating+L1HrWt+LastRun
            +strata(Run),train.cl)
##########Form a data matrix for prediction使用條件邏輯迴歸模型預測每匹馬的勝出概率
m<-model.matrix(~Age+HrWt+WtCr+L1Class+AVG4FinPos+L1Rating+L1HrWt+LastRun
                -1, data=test.cl)
##########Coeficient times data matrix and predicted probability
pp<-exp(m %*% coef(fit))
pps<-ave(pp, test.cl$Run, FUN=sum)
pred.cl<-pp/pps


##########Using Extreme Gradient Bossting##########
##########Transform data frame to matrix
mtrain.xgb<-data.matrix(train.xgb[,-c(1:3,12,13,14)])
mtest.xgb<-data.matrix(test.xgb[,-c(1:3,12,13,14)])
##########Take out final odds
trainodd.xgb<-data.matrix(train.xgb[,14])
testodd.xgb<-data.matrix(test.xgb[,14])
##########Vectorized output
output_vector1 = train.xgb[,"FO"]==1
output_vector2 = test.xgb[,"FO"]==1
##########Extreme gradient boosting training
bst<-xgboost(
  data = mtrain.xgb, label = output_vector1,nround = 1,objective="binary:logistic")
##########Predict probability
pred.xgb<-predict(bst,mtest.xgb)
##########Variable Importance
name.xgb<-colnames(mtrain.xgb)
importance.xgb<- xgb.importance(feature_names = name.xgb, model = bst)

head(importance.xgb,10)
xgb.plot.importance(importance_matrix = importance.xgb)

xgb.plot.tree(model=bst, trees=0:1, render=T)



##########Using Random Forest##########
##########Transform data frame to matrix
dtrain.rf<-train.rf[,-c(1:3,13,14)]
dtest.rf<-test.rf[,-c(1:3,13,14)]
dtrain.rf$FO<-as.factor(dtrain.rf$FO)
dtest.rf$FO<-as.factor(dtest.rf$FO)
trainodd.rf<-train.rf[,14]
testodd.rf<-test.rf[,14]
##########Building Random Forrest with Ranger library
pbrf.model<-ranger(FO~.,dtrain.rf,probability = TRUE,num.trees = 100,mtry = 5,write.forest = TRUE,min.node.size = 3,importance="impurity")
##########Prediction
pred.rf<-predict(pbrf.model, dat=dtest.rf)


##########Variable Importance
plot(pbrf.model$variable.importance)

##########Using Support Vector Machine##########
#library("Rgtsvm", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.3") #Use this instead of e1071 if you are using linux with GPU for 10 times faster training
##########Transform data frame to matrix
dtrain.svm<-train.svm[,-c(1:3,12,14)]
dtest.svm<-test.svm[,-c(1:3,12,14)]
dtrainx.svm<-train.svm[,-c(1:3,12,13,14)]
dtrainy.svm<-train.svm[,c(13)]
dtestx.svm<-test.svm[,-c(1:3,12,13,14)]
dtesty.svm<-test.svm[,c(13)]
##########Run Support Vector Machine
svm.model<-svm(FinPos~.,data=dtrain.svm,type = "eps-regression",probaility=T)
pred.svm<-predict(svm.model,dtestx.svm)


result.overall<-as.data.frame(cbind(test,pred.cl,pred.xgb,pred.rf$predictions,pred.svm))
colnames(result.overall)[colnames(result.overall)=="V1"] <- "pred.cl"
colnames(result.overall)[colnames(result.overall)=="1"] <- "pred.rf"
result.overall$`0`<-NULL
result.overall<-ddply(result.overall,.(Run),transform,rankO=rank(FinOdd,ties.method="min"))
result.overall<-ddply(result.overall,.(Run),transform,rank.cl=rank(-pred.cl,ties.method="min"))
result.overall<-ddply(result.overall,.(Run),transform,rank.xgb=rank(-pred.xgb,ties.method="min"))
result.overall<-ddply(result.overall,.(Run),transform,rank.rf=rank(-pred.rf,ties.method="min"))
result.overall<-ddply(result.overall,.(Run),transform,rank.svm=rank(pred.svm,ties.method="min"))

print(paste0("Favor Odds: Accuracy ", sum(ifelse(result.overall$FO==1&result.overall$rankO==1,1,0))/sum(result.overall$FO==1)
             ,"; Return ",
             sum(ifelse(result.overall$FO==1&result.overall$rankO==1,1,0)*result.overall$FinOdd)/sum(result.overall$FO==1)-1
))

print(paste0("CLogistic Regression: Accuracy ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.cl==1,1,0))/sum(result.overall$FO==1)
             ,"; Return ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.cl==1,1,0)*result.overall$FinOdd)/sum(result.overall$FO==1)-1
))

print(paste0("Xgboost: Accuracy ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.xgb==1,1,0))/sum(result.overall$FO==1)
             ,"; Return ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.xgb==1,1,0)*result.overall$FinOdd)/sum(result.overall$FO==1)-1
))

print(paste0("Random Forest: Accuracy ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.rf==1,1,0))/sum(result.overall$FO==1)
             ,"; Return ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.rf==1,1,0)*result.overall$FinOdd)/sum(result.overall$FO==1)-1
))

print(paste0("Support Vector Machine: Accuracy ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.svm==1,1,0))/sum(result.overall$FO==1)
             ,"; Return ",
             sum(ifelse(result.overall$FO==1&result.overall$rank.svm==1,1,0)*result.overall$FinOdd)/sum(result.overall$FO==1)-1
))

print(paste0("Total number of run: ",sum(result.overall$FO==1)))

##########Stacking Result from 4 models
test.stack<-cbind(test.svm,pred.cl,pred.svm,pred.xgb,pred.rf$predictions[,2])
##########Data conversion
dstackx<-data.matrix(test.stack[,c(15:18)])
dstacky<-test.stack[,c(12)]==1
##########Stacking model using XGBoost
bst.stack<-xgboost(data = dstackx, label = dstacky,nround = 5,objective ="binary:logistic",max.depth = 5,eta = 0.8, nthread = 8, gamma=1)
pred.bst.stack<-predict(bst.stack,dstackx)