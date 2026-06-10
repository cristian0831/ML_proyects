# kaggle project

# Marketing campaign

The main objective is to train a predictive model which allows the company to maximize the profit of the next marketing campaign. 

# Data set description

Historical accepted campaign: assign 0 (negative) or 1 (positive).
Reponse (target): response of the last campaign.
Customer personal info: Income, Teenhome, Marital status,...
Purchase behavior: prefered channel.
Customer spends: on the last two years.

# Pipeline description

Get data: from .csv file (complete structured data set).
Data understanding: knowledge about numerical and categorical features.
Feature engine: build new features increasing the correlation with the response variable.
Data cleaning (preparing for ML): handle missing data and encoding categorical attributes.
Train Model: using RandomForestClassifier train the model on training set to predict how will response the customer. 
Evaluate performance: 
1) use cross validation to calculate scores (on training data) on the first class (income_feature, prob_answer_1).
2) calculate metrics on training and scores data; precision, recall, finding the most appropiate threshold.
3) print confusion matrix using test data (unseen customers) and understand what says the metrics.
4) estimate the ROI.
5) make a list of "new" customers ordering by the higher probability to response 1.

# Improvements
Train with other models and evaluate their performance. 