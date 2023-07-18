import csv


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






#---------------------------------------------------------------


#------------------CLUSTERING-----------------------------------








#---------------------------------------------------------------


#--------------------REGRESSION---------------------------------







#---------------------------------------------------------------