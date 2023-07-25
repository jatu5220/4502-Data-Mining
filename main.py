import csv
import pandas as pd
import re

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


#------------------CLUSTERING-----------------------------------








#---------------------------------------------------------------


#--------------------REGRESSION---------------------------------







#---------------------------------------------------------------

#---------------------MACHINE LEARNING PREDICTION---------------









#---------------------------------------------------------------