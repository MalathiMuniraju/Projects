mydata<-read.csv("F:\\Machine Learning Assignment\\Bank Details.csv",header=T)
str(mydata)
dim(mydata)
library(Rcmdr)

#Data Exploration
##Variable Age
boxplot(mydata$age,main="Dispersion of variable age")
summary(mydata$age)
library(ggplot2)
qplot(x=mydata$age,data=mydata,main="Distribution of age",xlab="Age",ylab="Frequency")
#Binning the age variable
mydata$age<-cut(mydata$age,c(15,20,30,40,50,60,70,80,90,100),labels=c("1","2","3","4","5","6","7","8","9"))
qplot(x=mydata$age,data=mydata,main="Distribution of age",xlab="Age",ylab="Frequency")
table(mydata$age)

##Variable Job
length(mydata$job)
table(mydata$job)
barplot(table(mydata$job),main="Count of each job category",xlab="Job Category",ylab="Count")
table(mydata$job,mydata$y)
with(mydata, Barplot(job, by=y, style="divided", legend.pos="above", 
                     xlab="job", ylab="Frequency"))
#Dropping unknown data from job category
mydata<-subset(mydata,subset=(job!="unknown"))
mydata$job<-factor(mydata$job)
table(mydata$job)
length(mydata$job)

##Variable Marital
table(mydata$marital)
barplot(table(mydata$marital))
table(mydata$marital,mydata$y)
with(mydata, Barplot(marital, by=y, style="divided", legend.pos="above", 
                     xlab="marital", ylab="Frequency"))
length(mydata$marital)

##Variable Education
table(mydata$education)
barplot(table(mydata$education))
addmargins(table(mydata$education,mydata$y))
with(mydata, Barplot(education, by=y, style="divided", legend.pos="above", 
                     xlab="education", ylab="Frequency"))
##Dropping unknown value from the variable education
mydata<-subset(mydata,subset=(education!="unknown"))
mydata$education<-factor(mydata$education)
table(mydata$education)
length(mydata$education)

##Variable Default
table(mydata$default)
addmargins(table(mydata$default,mydata$y))
#Dropping the default column as the values are almost same
mydata<-subset(mydata,select=-c(default))


##Variable balance
boxplot(mydata$balance,main="Distribution of Balance",xlab="Balance",ylab="Frequency")
summary(mydata$balance)
qplot(x=mydata$balance,data=mydata,main="Distribution of Balance",xlab="Balance",ylab="Frequency")
#Binning the age variable
mydata$balance<-as.numeric(mydata$balance)
mydata$balance<-cut(mydata$balance,c(-10000,0,10000,50000,130000),labels=c("Negative","Low","Medium","High"))
qplot(x=mydata$balance,data=mydata,main="Distribution of Balance",xlab="Balance",ylab="Frequency")
table(mydata$balance)
length(mydata$balance)

##Variable Housing
table(mydata$housing)
barplot(table(mydata$housing),main="Distribution of Housing Loan",xlab="Housing Loan")
addmargins(table(mydata$housing,mydata$y))
with(mydata, Barplot(housing, by=y, style="divided", legend.pos="above", 
                     xlab="housing", ylab="Frequency"))


##Variable Loan
table(mydata$loan)
barplot(table(mydata$loan),main="Distribution of Personal Loan",xlab="Personal Loan")
addmargins(table(mydata$loan,mydata$y))
with(mydata, Barplot(loan, by=y, style="divided", legend.pos="above", 
                     xlab="loan", ylab="Frequency"))

##Variable Contact
table(mydata$contact)
barplot(table(mydata$contact),main="Distribution of Contact",xlab="Contact")
addmargins(table(mydata$contact,mydata$y))
with(mydata, Barplot(contact, by=y, style="divided", legend.pos="above", 
                     xlab="contact", ylab="Frequency"))
#Dropping unknown values from the contact
mydata<-subset(mydata,subset=(contact!="unknown"))
mydata$contact<-factor(mydata$contact)
table(mydata$contact)
length(mydata$contact)

##Variable Day 
qplot(x=mydata$day,main="Last Contact Day of the Week",xlab="Days")

##Variable Month
qplot(x=mydata$month,main="Last Contact Month of the Year",xlab="Months")

#Dropping duration column as it greatly affects the outcome column and duration will be know after the call has been made
mydata<-subset(mydata,select=-c(duration))

##Variable Campaign
summary(mydata$campaign)
qplot(x=mydata$campaign,data=mydata,main="Campaigning",xlab="No.of contacts",ylab="Frequency")
#Binning the age variable
mydata$campaign<-as.numeric(mydata$campaign)
mydata$campaign<-cut(mydata$campaign,c(0,15,30,45,65),labels=c("Low","Medium","High","Extreme"))
qplot(x=mydata$campaign,data=mydata,main="Distribution of Calls",xlab="No.of Contacts",ylab="Frequency")
table(mydata$campaign)
addmargins(table(mydata$campaign,mydata$y))
length(mydata$campaign)

##Variable pdays
summary(mydata$pdays)
qplot(x=mydata$pdays,data=mydata,main="Days passed after previous Campigning",xlab="No.of Days Passed",ylab="Frequency")
#Binning the age variable
mydata$pdays<-as.numeric(mydata$pdays)
mydata$pdays<-cut(mydata$pdays,c(-5,200,600,800,900),labels=c("Low","Medium","High","Extreme"))
qplot(x=mydata$pdays,data=mydata,main="Days passed after previous Campigning",xlab="No.of Days Passed",ylab="Frequency")
table(mydata$pdays)
addmargins(table(mydata$pdays,mydata$y))
length(mydata$pdays)

##Variable Previous
summary(mydata$previous)
qplot(x=mydata$previous,data=mydata,main="No.of contacts performed before this campaign",xlab="No.of contacts",ylab="Frequency")
length(mydata$previous)
#Binning the age variable
mydata$previous<-as.numeric(mydata$previous)
mydata$previous<-cut(mydata$previous,c(-10,0,30,60,300),labels=c("Low","Medium","High","Extreme"))
levels(mydata$previous)
qplot(x=mydata$previous,data=mydata,main="No.of contacts performed before this campaign",xlab="No.of contacts",ylab="Frequency")
table(mydata$previous)
addmargins(table(mydata$previous,mydata$y))
length(mydata$previous)

##Variable poutcome
table(mydata$poutcome)
addmargins(table(mydata$poutcome,mydata$y))
#Dropping the poutcome column as most of the values are unknow
mydata<-subset(mydata,select=-c(poutcome))

levels(mydata$y)
mydata$y<-relevel(mydata$y,"yes")


#sampling and modelling

library(ROCR)
library(caret)
library(nnet)
 set.seed(123)
#Defining traing control
train_control<-trainControl(method="cv",number=10)
#train the model
model<-train(y~.,data=mydata,trControl=train_control,method="nnet")
#make predictions
predicted<-predict(model,mydata)
head(predicted)
predicted_prob<-predict(model,mydata,type="prob")
head(predicted_prob)
confusionMatrix(predicted,mydata$y)
mult_measures<-prediction(predicted_prob[ ,1],mydata[ ,14])
ROC<-performance(mult_measures,measure="tpr",x.measure="fpr")
auc<-performance(mult_measures,measure="auc")
auc<-auc@y.values
auc_legend<-paste(c("AUC",auc),collapse = "")
plot(ROC,col="red")
abline(a=0,b=1)
legend(0.6,0.3,auc_legend,lty=1,lwd=1,col="red")
table(mydata$y)
print(model)
summary(model)
plot(model)

#Repeated Nueral Network

#sampling and modelling

library(ROCR)
library(caret)
library(nnet)
set.seed(456)
#Defining traing control
train_control<-trainControl(method="repeatedcv",number=10,repeats=3)
#train the model
model<-train(y~.,data=mydata,trControl=train_control,method="nnet")
#make predictions
predicted<-predict(model,mydata)
head(predicted)
predicted_prob<-predict(model,mydata,type="prob")
head(predicted_prob)
confusionMatrix(predicted,mydata$y)
mult_measures<-prediction(predicted_prob[ ,1],mydata[ ,14])
ROC<-performance(mult_measures,measure="tpr",x.measure="fpr")
auc<-performance(mult_measures,measure="auc")
auc<-auc@y.values
auc_legend<-paste(c("AUC",auc),collapse = "")
plot(ROC,col="red")
abline(a=0,b=1)
legend(0.6,0.3,auc_legend,lty=1,lwd=1,col="red")

print(model)
summary(model)
plot(model)

#C5.0 model
#sampling and modelling

library(ROCR)
library(caret)
library(C50)
set.seed(101)
#Defining traing control
train_control<-trainControl(method="cv",number=10)
#train the model
model<-train(y~.,data=mydata,trControl=train_control,method="C5.0")
#make predictions
predicted<-predict(model,mydata)
head(predicted)
predicted_prob<-predict(model,mydata,type="prob")
head(predicted_prob)
confusionMatrix(predicted,mydata$y)
mult_measures<-prediction(predicted_prob[ ,1],mydata[ ,14])
ROC<-performance(mult_measures,measure="tpr",x.measure="fpr")
auc<-performance(mult_measures,measure="auc")
auc<-auc@y.values
auc_legend<-paste(c("AUC",auc),collapse = "")
plot(ROC,col="red")
abline(a=0,b=1)
legend(0.6,0.3,auc_legend,lty=1,lwd=1,col="red")

print(model)
summary(model)
plot(model)

#Repeated C5.0 model
#sampling and modelling

library(ROCR)
library(caret)
library(C50)
set.seed(404)
#Defining traing control
train_control<-trainControl(method="repeatedcv",number=10,repeats =3)
#train the model
model<-train(y~.,data=mydata,trControl=train_control,method="C5.0")
#make predictions
predicted<-predict(model,mydata)
head(predicted)
predicted_prob<-predict(model,mydata,type="prob")
head(predicted_prob)
confusionMatrix(predicted,mydata$y)
mult_measures<-prediction(predicted_prob[ ,1],mydata[ ,14])
ROC<-performance(mult_measures,measure="tpr",x.measure="fpr")
auc<-performance(mult_measures,measure="auc")
auc<-auc@y.values
auc_legend<-paste(c("AUC",auc),collapse = "")
plot(ROC,col="red")
abline(a=0,b=1)
legend(0.6,0.3,auc_legend,lty=1,lwd=1,col="red")

print(model)
summary(model)
plot(model)

#rpart Model
#sampling and modelling

library(ROCR)
library(caret)
library(rpart)
set.seed(404)
#Defining traing control
train_control<-trainControl(method="cv",number=10)
#train the model
model<-train(y~.,data=mydata,trControl=train_control,method="rpart")
#make predictions
predicted<-predict(model,mydata)
head(predicted)
predicted_prob<-predict(model,mydata,type="prob")
head(predicted_prob)
confusionMatrix(predicted,mydata$y)
mult_measures<-prediction(predicted_prob[ ,1],mydata[ ,14])
ROC<-performance(mult_measures,measure="tpr",x.measure="fpr")
auc<-performance(mult_measures,measure="auc")
auc<-auc@y.values
auc_legend<-paste(c("AUC",auc),collapse = "")
plot(ROC,col="red")
abline(a=0,b=1)
legend(0.6,0.3,auc_legend,lty=1,lwd=1,col="red")

print(model)
summary(model)
plot(model)





