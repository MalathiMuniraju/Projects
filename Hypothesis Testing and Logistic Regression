#To import the dataset
df<-read.csv("D:/HomeLoan.csv",header=T)

#To explore the dataset
summary(df)
str(df)
head(df)
tail(df)

#To replace blank values with NA
df<-read.csv("D:/HomeLoan.csv",header=T,na.strings=c(""))

#To check the missing values in each variable
sapply(df,function(x) sum(is.na(x)))

#To check the unique values in each variable
sapply(df,function(x)length(unique(x)))

##REPLACING MISSING VALUES
#To fill Gender by mode
addmargins(table(df$Gender))
df$Gender<-as.character(df$Gender)
df$Gender=ifelse(is.na(df$Gender),"Male",df$Gender)
df$Gender<-as.factor(df$Gender)
addmargins(table(df$Gender))

#Replacing missing value of Married with mode
addmargins(table(df$Married))
df$Married<-as.character(df$Married)
df$Married=ifelse(is.na(df$Married),"Yes",df$Married)
df$Married<-as.factor(df$Married)


#Replacing missing value of Dependents with mode
addmargins(table(df$Dependents))
df$Dependents<-as.character(df$Dependents)
df$Dependents=ifelse(is.na(df$Dependents),"0",df$Dependents)
df$Dependents<-as.factor(df$Dependents)

#Replacement of self employed based on Education
addmargins(table(df$Education,df$Self_Employed))
class(df$Self_Employed)
df$Self_Employed=as.character(df$Self_Employed)
df$Self_Employed<-ifelse(df$Education=="Graduate" & is.na(df$Self_Employed),"No",
                         ifelse(df$Education=="Not Graduate" & is.na(df$Self_Employed),"Yes",df$Self_Employed))
df$Self_Employed=as.factor(df$Self_Employed)

#To replace missing value of loan amount by median
df$LoanAmount[is.na(df$LoanAmount)]=median(df$LoanAmount,na.rm=TRUE)

#To replace missing value of Loan amount term by median
df$Loan_Amount_Term[is.na(df$Loan_Amount_Term)]=median(df$Loan_Amount_Term,na.rm=TRUE)

#To replace missing credit history based on Loanstatus
df$Credit_History<-ifelse(df$Loan_Status=="Y" & is.na(df$Credit_History),1,
                          ifelse(df$Loan_Status=="N" & is.na(df$Credit_History),0,df$Credit_History))

#To check the missing values in each variable
sapply(df,function(x) sum(is.na(x)))

#To check the unique values in each variable
sapply(df,function(x)length(unique(x)))

#Lof Transformation of loan amount
xt<-log(df$LoanAmount)
x2t<-log(df$ApplicantIncome)
x3t<-log(df$CoapplicantIncome)

#To check whether the avearge loan cross the gender is same or not
#Create the dataframe for transformed loanamount and gender
attach(df)
Avg_gen_Amount<-data.frame(xt,Gender)
summary(Avg_gen_Amount)

#To create the subset for Female and Male 
attach(Avg_gen_Amount)
Avg_female_Amount<-subset(Avg_gen_Amount,Gender=="Female")
Avg_female_Amount$Gender<-factor(Avg_female_Amount$Gender)
levels(Avg_female_Amount$Gender)
dim(Avg_female_Amount)

Avg_Male_Amount<-subset(Avg_gen_Amount,Gender=="Male")
Avg_Male_Amount$Gender<-factor(Avg_Male_Amount$Gender)
levels(Avg_Male_Amount$Gender)
dim(Avg_Male_Amount)

#To check the normality assumption
qqnorm(Avg_female_Amount$xt);qqline(Avg_female_Amount$xt)
qqnorm(Avg_Male_Amount$xt);qqline(Avg_Male_Amount$xt)

#To check the variance of the the population
library(car)
leveneTest(xt~Gender,data=Avg_gen_Amount)
fligner.test(xt~Gender,data=Avg_gen_Amount) # As this is non parametric test 
boxplot(Avg_female_Amount$xt,Avg_Male_Amount$xt,main="Variance of loan Amount across gender",xlab="Gender",ylab="Transformed Loan Amount")

#Performing 'Two Sample Student's T-Test' as the homogeneity assumption has been proved
t.test(Avg_female_Amount$xt,Avg_Male_Amount$xt,alt="two.sided",var.equal = TRUE)

summary(Avg_female_Amount$xt)
summary(Avg_Male_Amount$xt)

#Inverse Tansformation of the loan amount
inve_trans_female<-2.71828^(Avg_female_Amount$xt)
summary(inve_trans_female)
inve_trans_male<-2.71828^(Avg_Male_Amount$xt)
summary(inve_trans_male)



##TO check the average loan amount across the married and unmarried are same or not
#Create the dataframe for transformed loanamount and gender
attach(df)
Avg_marr_Amount<-data.frame(xt,Married)
summary(Avg_gen_Amount)

#To create the subset for Female and Male 
attach(Avg_gen_Amount)
Avg_married_Amount<-subset(Avg_marr_Amount,Married=="Yes")
Avg_married_Amount$Married<-factor(Avg_married_Amount$Married)
levels(Avg_married_Amount$Married)
dim(Avg_married_Amount)

Avg_unmarried_Amount<-subset(Avg_marr_Amount,Married=="No")
Avg_unmarried_Amount$Married<-factor(Avg_unmarried_Amount$Married)
levels(Avg_unmarried_Amount$Married)
dim(Avg_unmarried_Amount)

#To check the normality assumption
qqnorm(Avg_married_Amount$xt);qqline(Avg_married_Amount$xt)
qqnorm(Avg_unmarried_Amount$xt);qqline(Avg_unmarried_Amount$xt)

#To check the variance of the the population
library(car)
leveneTest(xt~Married,data=Avg_marr_Amount)
fligner.test(xt~Married,data=Avg_marr_Amount) # As this is non parametric test 
boxplot(Avg_married_Amount$xt,Avg_unmarried_Amount$xt,main="Variance of loan Amount across married & unmarried",xlab="Married",ylab="Transformed Loan Amount")

#Performing 'Two Sample Student's T-Test' as the homogeneity assumption has been proved
t.test(Avg_married_Amount$xt,Avg_unmarried_Amount$xt,alt="two.sided",var.equal = TRUE)

summary(Avg_married_Amount$xt)
summary(Avg_unmarried_Amount$xt)

#Inverse Tansformation of the loan amount
inve_trans_married<-2.71828^(Avg_married_Amount$xt)
summary(inve_trans_married)
inve_trans_unmarried<-2.71828^(Avg_unmarried_Amount$xt)
summary(inve_trans_unmarried)


#Logistic Regression to find the propabilities of customer getting loan amount
#Create new dataframe with transformed variables
df<-df[,-c(1,7,9)]
df<-data.frame(df,xt,x2t)

#To split the dataset into test and train
set.seed(500)
test_trainIndices<-sample.int(n=nrow(df), 0.70*nrow(df),replace=F)
trainingdata<-df[test_trainIndices, ]
testdata<-df[-test_trainIndices, ]
dim(trainingdata)
dim(testdata)

#Fittig Model on training data with all variables
model1<-glm(Loan_Status~.,data=trainingdata,family=binomial(link="logit"))
summary(model1)
library(pscl)
pR2(model1)
library(car)
vif(model1)
anova(model1,test="Chisq")

#Predicting on the test data
predicted1<-predict(model1,testdata,type="response")
head(predicted1)
summary(predicted1)
length(predicted1)
predicted2<-plogis(predicted1)
head(predicted2)
summary(predicted2)

#To know the optimal cutoff value
library(InformationValue)
optcutoff<-optimalCutoff(testdata$Loan_Status,predicted2)
optcutoff

#To convert the predictions into class
model_pred_Direction<-rep("N",185)
model_pred_Direction[predicted1>0.5]="Y"
head(model_pred_Direction)
class(model_pred_Direction)
model_pred_Direction<-as.factor(model_pred_Direction)
levels(model_pred_Direction)

misclassificationerror<-mean(model_pred_Direction!=testdata$Loan_Status)
misclassificationerror
print(paste("Accuracy",1-misclassificationerror))

library(caret)
confusionMatrix(testdata$Loan_Status,model_pred_Direction, positive = levels(testdata$Loan_Status)[2])

#To draw the ROC
library(ROCR)
pred<-prediction(predicted1,testdata$Loan_Status)
perf<-performance(pred,measure="tpr",x.measure = "fpr")
plot(perf,main="ROC Curev",xlab="True Negative Rate(Specificity)",ylab="True Positive Rate(Sensitivity)")
abline(0,1)

auc<-performance(pred,measure="auc")
auc<-auc@y.values[[1]]
auc
print(paste("AUC",auc))

#Try fitting the model with significant variable
model<-glm(Loan_Status~Education+Credit_History+Property_Area,data=trainingdata,family=binomial(link="logit"))
summary(model)
library(pscl)
pR2(model)
library(car)
vif(model)
anova(model,test="Chisq")

#Predicting on the testdata
testdata<-subset(testdata,select=c(4,8,9,10))
pred_sign<-predict(model,newdata=testdata,type="response")
head(pred_sign)
summary(pred_sign)
length(pred_sign)
pred_sign_prob<-plogis(pred_sign)
head(pred_sign_prob)
summary(pred_sign_prob)

#To know the optimal cutoff value
library(InformationValue)
optcutoff<-optimalCutoff(testdata$Loan_Status,pred_sign_prob)
optcutoff

#To convert the predictions into class
model_predsign_Direction<-rep("N",185)
model_predsign_Direction[pred_sign>0.52]="Y"
head(model_predsign_Direction)
class(model_predsign_Direction)
model_predsign_Direction<-as.factor(model_predsign_Direction)
levels(model_predsign_Direction)

misclassificationerror<-mean(model_predsign_Direction!=testdata$Loan_Status)
misclassificationerror
print(paste("Accuracy",1-misclassificationerror))

library(caret)
confusionMatrix(testdata$Loan_Status,model_predsign_Direction, positive = levels(testdata$Loan_Status)[2])


#To draw the AUC
library(ROCR)
pred<-prediction(pred_sign,testdata$Loan_Status)
perf<-performance(pred,measure="tpr",x.measure = "fpr")
plot(perf,main="ROC Curve",xlab="True Negative Rate(Specificity)",ylab="True Positive Rate(Sensitivity)")
abline(0,1)

auc<-performance(pred,measure="auc")
auc<-auc@y.values[[1]]
auc
print(paste("AUC",auc))





