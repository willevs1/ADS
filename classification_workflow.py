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


TRAIN_PATH = "Data/Classification_train.csv"
TEST_PATH = "Data/Classification_test.csv"
TARGET_COLUMN = "Credit_Score"

NUMERIC_TEXT_COLS = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Age",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

NUMERIC_COLS = [
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

CATEGORICAL_FILL_COLS = [
    "Occupation",
    "Type_of_Loan",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Spent",
    "Value_Payments",
]

DROP_COLS = ["ID", "Month", "Name", "SSN"]


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


def basic_clean(frame):
    frame = frame.copy()
    frame = frame.drop(DROP_COLS, axis=1, errors="ignore")

    for col in NUMERIC_TEXT_COLS:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip()
            frame[col] = frame[col].str.rstrip("_")
            frame[col] = frame[col].str.replace(",", "", regex=False)

    if "Credit_History_Age" in frame.columns:
        frame["Credit_History_Age"] = frame["Credit_History_Age"].apply(convert_credit_history)

    for col in NUMERIC_COLS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "Type_of_Loan" in frame.columns:
        frame["no_of_loan"] = frame["Type_of_Loan"].fillna("").apply(
            lambda x: 0 if x == "" else len(str(x).split(","))
        )

    if "Payment_Behaviour" in frame.columns:
        payment_split = frame["Payment_Behaviour"].str.split("_", expand=True)
        if payment_split is not None and payment_split.shape[1] >= 4:
            frame["Spent"] = payment_split[0]
            frame["Value_Payments"] = payment_split[2]

    if "Occupation" in frame.columns:
        frame["Occupation"] = frame["Occupation"].replace("_______", np.nan)
    if "Credit_Mix" in frame.columns:
        frame["Credit_Mix"] = frame["Credit_Mix"].replace("_", np.nan)
    if "Payment_of_Min_Amount" in frame.columns:
        frame["Payment_of_Min_Amount"] = frame["Payment_of_Min_Amount"].replace("NM", np.nan)
    if "Payment_Behaviour" in frame.columns:
        frame["Payment_Behaviour"] = frame["Payment_Behaviour"].replace("!@9#%8", np.nan)

    return frame


def fit_preprocessing(train):
    train = basic_clean(train)

    category_fill_map = {}
    for col in CATEGORICAL_FILL_COLS:
        if col in train.columns:
            mode_value = train[col].mode(dropna=True)
            if not mode_value.empty:
                category_fill_map[col] = mode_value.iloc[0]
                train[col] = train[col].fillna(category_fill_map[col])

    numeric_fill_map = {}
    numeric_features = train.select_dtypes(include="number").columns.tolist()
    numeric_features = [col for col in numeric_features if col != TARGET_COLUMN]

    for col in numeric_features:
        if "Customer_ID" in train.columns:
            train[col] = train.groupby("Customer_ID")[col].transform(clean_group_numeric)
        numeric_fill_map[col] = train[col].median()
        train[col] = train[col].fillna(numeric_fill_map[col])

    if "Spent" in train.columns:
        train["Spent"] = train["Spent"].replace({"High": 1, "Low": 0})
    if "Value_Payments" in train.columns:
        train["Value_Payments"] = train["Value_Payments"].replace(
            {"Small": 0, "Medium": 1, "Large": 2}
        )
    if "Credit_Mix" in train.columns:
        train["Credit_Mix"] = train["Credit_Mix"].replace({"Bad": 0, "Standard": 1, "Good": 2})
    if "Payment_of_Min_Amount" in train.columns:
        train["Payment_of_Min_Amount"] = train["Payment_of_Min_Amount"].replace({"Yes": 1, "No": 0})

    train = pd.get_dummies(train, columns=["Occupation"], drop_first=True)
    train = train.drop(["Type_of_Loan", "Payment_Behaviour", "Customer_ID"], axis=1, errors="ignore")
    train = train.dropna(axis=0)

    feature_columns = [col for col in train.columns if col != TARGET_COLUMN]

    preprocessing_state = {
        "category_fill_map": category_fill_map,
        "numeric_fill_map": numeric_fill_map,
        "feature_columns": feature_columns,
    }
    return train, preprocessing_state


def transform_with_preprocessing(frame, preprocessing_state):
    frame = basic_clean(frame)

    for col, fill_value in preprocessing_state["category_fill_map"].items():
        if col in frame.columns:
            frame[col] = frame[col].fillna(fill_value)

    for col, fill_value in preprocessing_state["numeric_fill_map"].items():
        if col in frame.columns:
            if "Customer_ID" in frame.columns:
                frame[col] = frame.groupby("Customer_ID")[col].transform(clean_group_numeric)
            frame[col] = frame[col].fillna(fill_value)

    if "Spent" in frame.columns:
        frame["Spent"] = frame["Spent"].replace({"High": 1, "Low": 0})
    if "Value_Payments" in frame.columns:
        frame["Value_Payments"] = frame["Value_Payments"].replace(
            {"Small": 0, "Medium": 1, "Large": 2}
        )
    if "Credit_Mix" in frame.columns:
        frame["Credit_Mix"] = frame["Credit_Mix"].replace({"Bad": 0, "Standard": 1, "Good": 2})
    if "Payment_of_Min_Amount" in frame.columns:
        frame["Payment_of_Min_Amount"] = frame["Payment_of_Min_Amount"].replace({"Yes": 1, "No": 0})

    frame = pd.get_dummies(frame, columns=["Occupation"], drop_first=True)
    frame = frame.drop(["Type_of_Loan", "Payment_Behaviour", "Customer_ID"], axis=1, errors="ignore")

    for col in preprocessing_state["feature_columns"]:
        if col not in frame.columns:
            frame[col] = 0

    extra_cols = [
        col for col in frame.columns
        if col not in preprocessing_state["feature_columns"] + [TARGET_COLUMN]
    ]
    if extra_cols:
        frame = frame.drop(columns=extra_cols, errors="ignore")

    ordered_cols = preprocessing_state["feature_columns"][:]
    if TARGET_COLUMN in frame.columns:
        ordered_cols = ordered_cols + [TARGET_COLUMN]

    frame = frame.reindex(columns=ordered_cols, fill_value=0)
    return frame


def build_models():
    return {
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


# -------------------------------------------------------------------
# 1. Load raw train and test data
# -------------------------------------------------------------------
train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_raw = pd.read_csv(TEST_PATH, low_memory=False)

print("Raw train shape:", train_raw.shape)
print("Raw test shape:", test_raw.shape)


# -------------------------------------------------------------------
# 2. Fit preprocessing on train, then apply the same transformations to test
# -------------------------------------------------------------------
train, preprocessing_state = fit_preprocessing(train_raw)
test = transform_with_preprocessing(test_raw, preprocessing_state)

print("Processed train shape:", train.shape)
print("Processed test shape:", test.shape)
print(train.isnull().sum())
print(test.isnull().sum())


# -------------------------------------------------------------------
# 3. Define modelling data
# -------------------------------------------------------------------
X = train.drop(TARGET_COLUMN, axis=1)
y = train[TARGET_COLUMN]
X_submission = test.copy()


# -------------------------------------------------------------------
# 4. Train / validation split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# -------------------------------------------------------------------
# 5. Cross-validation for fair model comparison
# -------------------------------------------------------------------
models = build_models()
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
# 6. Fit and evaluate models on the validation split
# -------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# -------------------------------------------------------------------
# 7. Plot confusion matrices
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
# 8. Sensitivity and specificity
# -------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_df = class_sensitivity_specificity(y_test, y_pred, labels)
    metrics_df["Model"] = name

    print(f"\n{name}")
    print(metrics_df)


# -------------------------------------------------------------------
# 9. Fit a final model on all processed train rows and score test CSV
# -------------------------------------------------------------------
final_model = build_models()["Random Forest"]
final_model.fit(X, y)
test_predictions = final_model.predict(X_submission)

test_predictions_df = pd.DataFrame({
    "Predicted_Credit_Score": test_predictions,
})

print(test_predictions_df.head())
