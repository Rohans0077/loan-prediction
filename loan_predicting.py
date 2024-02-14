#Loan prediction project using machine learning in python

# Importing Library
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# Reading the training dataset in a dataframe using Pandas
df = pd.read_csv(r"C:\Users\AsusEJ555T\OneDrive\Desktop\internship2023\Loan_prediction_project_using_machine_learning_in_python\train.csv.xls")
# Reading the test dataset in a dataframe using Pandas
test = pd.read_csv(r"C:\Users\AsusEJ555T\OneDrive\Desktop\internship2023\Loan_prediction_project_using_machine_learning_in_python\test.csv.xls")
# Store total number of observation in training dataset
df_length =len(df)
# Store total number of columns in testing data set
test_col = len(test.columns)
# Summary of numerical variables for training data set
print(df.describe())
# Get the unique values and their frequency of variable Property_Area
print(df['Property_Area'].value_counts())
# Box Plot for understanding the distributions and to observe the outliers.
import matplotlib
# Histogram of variable ApplicantIncome
df['ApplicantIncome'].hist()
# Box Plot for variable ApplicantIncome of training data set
df.boxplot(column='ApplicantIncome')
df.show()
# Histogram of variable LoanAmount
df['LoanAmount'].hist(bins=50)
# Box Plot for variable LoanAmount by variable Gender of training data set
df.boxplot(column='LoanAmount', by = 'Gender')
# Loan approval rates in absolute numbers
loan_approval = df['Loan_Status'].value_counts()['Y']
print(loan_approval)
# Credit History and Loan Status
pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True)
#Function to output percentage row wise in a cross table
def percentageConvert(ser):
    return ser/float(ser[-1])
# Loan approval rate for customers having Credit_History (1)
df=pd.crosstab(df ["Credit_History"], df ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)
loan_approval_with_Credit_1 = df['Y'][1]
print("% of people whose loan is approved is ")
print(loan_approval_with_Credit_1*100)
