# Titanic_passenger_survival

This is Titanic Machine Learning Project with Support Vector Machine Classifier and Random Forests using scikit-learn.

In this project, I have used Python and scikit-learn to build SVC and random forest, and applied them to predict the survival rate of Titanic passengers.

Data preprocessing is one of the most prominent steps to make an effective prediction model in Machine Learning, and it is often a good practice to use data preprocessing 
pipelines. 

I have also built custom data transformers and chain where all these data pre-processing steps use scikit-learn pipelines.

The dataset is available from the below link:
https://www.kaggle.com/c/titanic/data

The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.

Here is a quick explanation of some of the features:

Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
Pclass: passenger class.
Name, Sex, Age: self-explanatory
SibSp: how many siblings & spouses of the passenger aboard the Titanic.
Parch: how many children & parents of the passenger aboard the Titanic.
Ticket: ticket id Fare: price paid (in pounds)
Cabin: passenger's cabin number
Embarked: where the passenger embarked the Titanic

The dataset is split into 2 parts, train.csv and test.csv for training and testing Machine Learning models respectively.
