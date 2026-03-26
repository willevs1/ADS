import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# -------------------------------------------------------------------
# 1. Load the classification dataset
# -------------------------------------------------------------------
df = pd.read_csv("Data/Classification_train.csv", low_memory=False)


# -------------------------------------------------------------------
# 2. Initial inspection
# -------------------------------------------------------------------
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())


# -------------------------------------------------------------------
# 3. Drop columns that are not useful for modelling
# -------------------------------------------------------------------
df = df.drop(["ID", "Month", "Name", "SSN"], axis=1)


# -------------------------------------------------------------------
# 4. Clean text formatting in mixed columns
# -------------------------------------------------------------------
for col in ["Annual_Income", "Monthly_Inhand_Salary", "Age",
            "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"]:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.rstrip("_")
    df[col] = df[col].str.replace(",", "", regex=False)


# -------------------------------------------------------------------
# 5. Convert credit history age into numeric months
# -------------------------------------------------------------------
def convert_credit_history(text):
    if pd.isna(text):
        return np.nan

    text = str(text)
    parts = text.replace("Years", "Year").replace("Months", "Month")
    parts = parts.split(" and ")
    if len(parts) != 2:
        return np.nan

    try:
        years = int(parts[0].split()[0])
        months = int(parts[1].split()[0])
        return years * 12 + months
    except (ValueError, IndexError):
        return np.nan


df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_credit_history)


# -------------------------------------------------------------------
# 6. Convert numeric-looking fields into numbers
# -------------------------------------------------------------------
numeric_cols = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# -------------------------------------------------------------------
# 7. Engineer extra features used in the notebook
# -------------------------------------------------------------------
df["no_of_loan"] = df["Type_of_Loan"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(","))
)

payment_split = df["Payment_Behaviour"].str.split("_", expand=True)
if payment_split is not None and payment_split.shape[1] >= 4:
    df["Spent"] = payment_split[0]
    df["Value_Payments"] = payment_split[2]


# -------------------------------------------------------------------
# 8. Fill categorical missing values
# -------------------------------------------------------------------
df["Occupation"] = df["Occupation"].replace("_______", np.nan)
df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)
df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

categorical_cols = [
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Spent",
    "Value_Payments",
]

for col in categorical_cols:
    if col in df.columns:
        mode_value = df[col].mode(dropna=True)
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value.iloc[0])


# -------------------------------------------------------------------
# 9. Fill numeric missing values by customer, then fallback to column median
# -------------------------------------------------------------------
def clean_group_numeric(series):
    valid = series.dropna()
    if len(valid) == 0:
        return series

    median_value = valid.median()

    if len(valid) < 4:
        return series.fillna(median_value)

    q1 = valid.quantile(0.02)
    q3 = valid.quantile(0.98)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    series = series.apply(
        lambda x: median_value if pd.notna(x) and (x < lower or x > upper) else x
    )
    return series.fillna(median_value)


numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [col for col in numeric_cols if col != "Credit_Score"]

for col in numeric_cols:
    df[col] = df.groupby("Customer_ID")[col].transform(clean_group_numeric)
    df[col] = df[col].fillna(df[col].median())


# -------------------------------------------------------------------
# 10. Encode ordinal / binary categorical variables
# -------------------------------------------------------------------
if "Spent" in df.columns:
    df["Spent"] = df["Spent"].replace({"High": 1, "Low": 0})

if "Value_Payments" in df.columns:
    df["Value_Payments"] = df["Value_Payments"].replace(
        {"Small": 0, "Medium": 1, "Large": 2}
    )

df["Credit_Mix"] = df["Credit_Mix"].replace({"Bad": 0, "Standard": 1, "Good": 2})
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"Yes": 1, "No": 0})


# -------------------------------------------------------------------
# 11. One-hot encode remaining nominal categorical variables
# -------------------------------------------------------------------
df = pd.get_dummies(
    df,
    columns=["Occupation"],
    drop_first=True,
)


# -------------------------------------------------------------------
# 12. Final modelling dataset
# -------------------------------------------------------------------
df = df.drop(["Type_of_Loan", "Payment_Behaviour", "Customer_ID"], axis=1, errors="ignore")
df = df.dropna(axis=0)

print(df.isnull().sum())
print(df.shape)


# -------------------------------------------------------------------
# 13. Define features and target
# -------------------------------------------------------------------
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]


# -------------------------------------------------------------------
# 14. Train / test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# -------------------------------------------------------------------
# 15. Build model pipelines
# Logistic Regression and KNN are scaled.
# Tree models do not need scaling.
# SMOTE is applied inside each pipeline to avoid leakage.
# -------------------------------------------------------------------
models = {
    "Logistic Regression": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
    "KNN": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Decision Tree": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]),
    "Random Forest": ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(random_state=42)),
    ]),
}


# -------------------------------------------------------------------
# 16. Cross-validation for fair model comparison
# -------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_results.append({
        "Model": name,
        "CV Mean Accuracy": scores.mean(),
        "CV Std Accuracy": scores.std(),
    })

cv_results_df = pd.DataFrame(cv_results).sort_values(
    by="CV Mean Accuracy",
    ascending=False,
)
print(cv_results_df)


# -------------------------------------------------------------------
# 17. Fit each model on the training data and evaluate on the test set
# -------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------------------------
# 18. Plot confusion matrices for each model
# -------------------------------------------------------------------
labels = sorted(y_test.unique())
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[i],
    )
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------
# 19. Sensitivity and specificity by class
# -------------------------------------------------------------------
def class_sensitivity_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            "Class": label,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
        })

    return pd.DataFrame(results)


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_df = class_sensitivity_specificity(y_test, y_pred, labels)
    metrics_df["Model"] = name

    print(f"\n{name}")
    print(metrics_df)
