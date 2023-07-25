import csv
import pandas as pd
import re

#-------------------DATA GATHERING----------------------------

# open first csv file
with open('cc_defaulter/credit_card_defaulter.csv', 'r') as file1:
    cc_csv = csv.reader(file1)


# open second csv file
with open('cc_defaulter/bank-full.csv', 'r') as file2:
    bank_csv = csv.reader(file2)


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


#---------------------------------------------------------------


#------------------CLUSTERING-----------------------------------








#---------------------------------------------------------------


#--------------------REGRESSION---------------------------------







#---------------------------------------------------------------

#---------------------MACHINE LEARNING PREDICTION---------------









#---------------------------------------------------------------