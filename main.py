import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

#-------------------DATA GATHERING----------------------------

# Load the data
balance_by_age = pd.read_csv('datasets/finace_student_loan_debt/balance_by_age.csv')
debt_distribution = pd.read_csv('datasets/finace_student_loan_debt/debt_amt_distribution2014.csv')
home_secured_debt = pd.read_csv('datasets/finace_student_loan_debt/home_secured_debt_age30.csv')
non_mort_balance = pd.read_csv('datasets/finace_student_loan_debt/non_mort_balance.csv')

#--------------------------------------------------------------


#-------------------PREPROCESSING------------------------------

# Remove / average missing values based on DS size 

# Remove outliers

def remove_special_characters(df, column):
    # Use regular expression to replace special characters with ''
    df[column] = df[column].apply(lambda row: re.sub(r'[^A-Za-z0-9 ]+', '', row))
    return df

def remove_null_values(df):
    df = df.dropna()
    return df

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define criteria for an outlier
    outlier_condition = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    
    # Remove outliers
    df_out = df.loc[outlier_condition]
    
    return df_out

# use functions below on selected columns 

#---------------------------------------------------------------


#------------------DISTRIBUTION ANALYSIS-----------------------------------

debt_distribution['NumberOfBorrowers'] = debt_distribution['NumberOfBorrowers'].str.replace(',', '').astype(int)
debt_distribution.plot(x='Balance2014', y='NumberOfBorrowers', kind='bar', legend=False)
plt.title('Distribution of Loan Balances in 2014')
plt.xlabel('Loan Balance')
plt.ylabel('Number of Borrowers')
plt.show()

#---------------------------------------------------------------


#------------------CORRELATION ANALYSIS-----------------------------------

correlations = home_secured_debt.corr()
print(correlations)

#---------------------------------------------------------------

#------------------TIME SERIES ANALYSIS-----------------------------------

# Plotting time series data
home_secured_debt.set_index('Year').plot()
plt.ylabel('Percentage')
plt.show()

#---------------------------------------------------------------

#------------------CLASSIFICATION MODEL FOR RECESSION-----------------------------------

# Prepare the features (X) and the target (y)
X_rec = home_secured_debt[['HaveLoan27_30', 'NoLoan27_30']]
y_rec = home_secured_debt['Recession']

# Split the data into training set and test set
X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(X_rec, y_rec, test_size=0.2, random_state=0)

# Train the model
clf_rec = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_rec.fit(X_train_rec, y_train_rec)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_rec, y_train_rec)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train_rec, y_train_rec)

# Predict on the test data
y_pred_rec = clf_rec.predict(X_test_rec)
y_pred_logistic = logistic_model.predict(X_test_rec)
y_pred_decisionTree = decision_tree_model.predict(X_test_rec)

# Print the accuracy score
print("Accuracy:", accuracy_score(y_test_rec, y_pred_rec))
# Print the precision score
print("Precision:", precision_score(y_test_rec, y_pred_rec, average = 'weighted', zero_division=1))
# Print the recall score
print("Recall:", recall_score(y_test_rec, y_pred_rec, average = 'weighted'))
# Print the F1 score
print("F1 Score:", f1_score(y_test_rec, y_pred_rec, average = 'weighted'))
# Print the Logistic Regression Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test_rec, y_pred_logistic))
# Print the Decision Tree Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test_rec, y_pred_decisionTree))
# Print the Logistic Regression Report
print("Logistic Regression Report:", classification_report(y_test_rec, y_pred_logistic))

#---------------------------------------------------------------


#--------------------TIME SERIES FOR NON MORT---------------------------------

# Set 'Time' as index
non_mort_balance['Time'] = pd.to_datetime(non_mort_balance['Time'])
non_mort_balance.set_index('Time', inplace=True)

# Plotting the time series data
non_mort_balance.plot()
plt.ylabel("Billions of Dollars")
plt.show()

#---------------------------------------------------------------

#---------------------MACHINE LEARNING PREDICTION---------------


# Prepare the features (X) and the target (y)
X_mlp = non_mort_balance.index.values.reshape(-1,1)
y_mlp = non_mort_balance['Auto Loan']
X_mlp_timestamp = X_mlp.astype(int) // 10**9

# Split the data into training set and test set
X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = train_test_split(X_mlp_timestamp, y_mlp, test_size=0.2, random_state=0)

# Train the model
model_mlp = LinearRegression()
model_mlp.fit(X_train_mlp, y_train_mlp)

# Predict on the test data
y_pred_mlp = model_mlp.predict(X_test_mlp)

# Print the RMSE
print("RMSE:", np.sqrt(mean_squared_error(y_test_mlp, y_pred_mlp)))
# Print the MAE
print("MAE:", np.sqrt(mean_absolute_error(y_test_mlp, y_pred_mlp)))

#---------------------------------------------------------------

#-------------------CLUSTER ANALYSIS-----------------------------

# Cluster periods based on loan values
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(non_mort_balance[['HELOC', 'Auto Loan', 'Credit Card', 'Student Loan', 'Other']])
non_mort_balance['Cluster'] = clusters

# Check the result
print(non_mort_balance.head())


#---------------------------------------------------------------

#---------------TIME SERIES BALANCE BY AGE------------------------------

# Set 'Year' as index
balance_by_age['Year'] = pd.to_datetime(balance_by_age['Year'], format='%Y')
balance_by_age.set_index('Year', inplace=True)

# Plotting the time series data
balance_by_age.plot()
plt.ylabel("Billions of Dollars")
plt.show()

#---------------------------------------------------------------


#---------------FORECASTING BY AGE------------------------------

# Prepare the features (X) and the target (y)
X_age = balance_by_age.index.year.values.reshape(-1, 1)
y_age = balance_by_age['under30']

# Split the data into training set and test set
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y_age, test_size=0.2, random_state=0)

# Train the model
model_age = LinearRegression()
model_age.fit(X_train_age, y_train_age)

# Predict on the test data
y_pred_age = model_age.predict(X_test_age)

# Print the RMSE
print("RMSE:", np.sqrt(mean_squared_error(y_test_age, y_pred_age)))
# Print the MAE
print("MAE:", np.sqrt(mean_absolute_error(y_test_age, y_pred_age)))

#---------------------------------------------------------------