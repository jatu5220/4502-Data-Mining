import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#-------------------DATA GATHERING----------------------------

# open first csv file
with open('datasets/finace_student_loan_debt/balance_by_age.csv', 'r') as file1:
    balance_by_age = csv.reader(file1)


# open second csv file
with open('datasets/finace_student_loan_debt/debt_amt_distribution2014.csv', 'r') as file2:
    debt_distribution = csv.reader(file2)


# open third csv
with open('datasets/finace_student_loan_debt/home_secured_debt_age30.csv', 'r') as file3:
    home_secured_debt = csv.reader(file3)


# open fourth csv
with open('datasets/finace_student_loan_debt/non_mort_balance.csv', 'r') as file4:
    non_mort_balance = csv.reader(file4)
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
plt.show()

#---------------------------------------------------------------

#------------------CLASSIFICATION MODEL FOR RECESSION-----------------------------------

# Prepare the features (X) and the target (y)
X = home_secured_debt[['HaveLoan27_30', 'NoLoan27_30']]
y = home_secured_debt['Recession']

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Print the accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))


#---------------------------------------------------------------


#--------------------REGRESSION---------------------------------







#---------------------------------------------------------------

#---------------------MACHINE LEARNING PREDICTION---------------









#---------------------------------------------------------------