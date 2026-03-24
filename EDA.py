import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\WEWL\OneDrive - Capco\Desktop\ADS Assessment\Data\Classification_train.csv")

print(df.head())

print(df.shape)
x, y = df.shape
print(f"Number of data points: {x*y}")
print(df.describe(include=object))
print(df.describe().T.round(2))

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.rstrip("_")

for col in ["Annual_Income", "Monthly_Inhand_Salary", "Age","Num_of_Delayed_Payment", "Changed_Credit_Limit", "Outstanding_Debt"]:
    df[col] = df[col].astype(str).str.rstrip("_")
    df[col] = df[col].str.replace(",", "", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

def plot_bar(column, title):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column,palette="flare")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

plot_bar("Occupation", "Occupation Distribution")
plot_bar("Credit_Mix", "Credit_Mix Distribution")
plot_bar("Payment_of_Min_Amount", "Payment_of_Min_Amount Dist")



# After improvement and create a function for plot distribution 
def plot_distribution(column, title):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(title)
    plt.xlim(0, df[column].dropna().max())
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

print(df["Annual_Income"].dropna().max())


plot_distribution("Annual_Income","Annual_Income")
plot_distribution("Monthly_Inhand_Salary","Monthly_Inhand_Salary")
plot_distribution("Outstanding_Debt","Outstanding_Debt")
plot_distribution("Credit_Utilization_Ratio","Credit_Utilization_Ratio")
plot_distribution("Num_Bank_Accounts","Num_Bank_Accounts")
plot_distribution("Credit_Utilization_Ratio","Credit_Utilization_Ratio")
plot_distribution("Total_EMI_per_month","Total_EMI_per_month")
plot_distribution("Amount_invested_monthly","Amount_invested_monthly")
plot_distribution("Monthly_Balance","Monthly_Balance")
plot_distribution("Changed_Credit_Limit","Changed_Credit_Limit")
plot_distribution("Num_of_Delayed_Payment","Num_of_Delayed_Payment")