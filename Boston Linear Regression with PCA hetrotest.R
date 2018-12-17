#Import dataset
boston_df<-read.csv("D:/boston.csv",header=T)
attach(boston_df)

#Exploration of data
summary(boston_df)
str(boston_df)
head(boston_df)
tail(boston_df)

#Importing libraries
library(lmtest)
library(car)
library(corrplot)

#Fitting and evaluating the model on the original dataset
lmoriginal<-lm(MV~.,data=boston_df)
lmoriginal
vif(lmoriginal)
summary(lmoriginal)

#from the vif multicolinearity exist in the data
#Correlation matrix and plotting the correlatio matrix
corr_boston<-cor(boston_df)
corr_boston
cor_plot<-corrplot(corr_boston,order="AOE",method="color",addCoef.col = "gray")

#Implementing PCA
##Removing dependent variabel
my_data<-subset(boston_df,select=-c(MV))
colnames(my_data)

##Principal Component Analysis
prin_comp<-prcomp(my_data,scale.=T)
names(prin_comp)

#Outputs the mean of variables
prin_comp$center

#Outputs the standard deviation of variables
prin_comp$scale

#Let's look at first 5 principal components and first 5 rows.
prin_comp$rotation[1:5,1:5]

dim(prin_comp$x)

#Resultant principal component plot
biplot(prin_comp,scale=0)

#Compute standard deviation of each principal component
std_dev<-prin_comp$sdev
std_dev

#Compute Variance
pr_var<-std_dev^2
pr_var

#Proportion of the variance explained
prop_varex<-pr_var/sum(pr_var)
prop_varex

#Scree Plot
plot(prop_varex,xlab="Principal Component",ylab="Proportion of Variance Explained",type="b")

#Cumulative scree plot
plot(cumsum(prop_varex),xlab="Principal Component",ylab="Cumulative Proportion of Variance Explained",type="b")


#Adding principal components with the dependent variable
boston1<-data.frame(MV,prin_comp$x)
boston1

#Considering first five components
boston1<-boston1[,1:6]


#Model with pricipal components
model=lm(MV~.,data=boston1)
vif(model)
summary(model)

#Model with original dataset
model1=lm(MV~.,data=boston_df)
summary(model1)

#Ckeck for heteroscedasticity and generate accurate estimate of std.error
library("sandwich")
bptest(model)
vcovHC(model,omega=NULL,type="HC4")
coeftest(model,df=Inf,vcov=vcovHC(model,type="HC4"))














