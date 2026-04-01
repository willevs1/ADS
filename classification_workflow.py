import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


TRAIN_PATH = "Data/Classification_train.csv"
TEST_PATH = "Data/Classification_test.csv"
OUTPUT_PATH = "Data/rf_test_predictions.csv"
TARGET = "Credit_Score"


# ahahha these are the columns I cleaned as numbers in the notebook
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
    "Num_of_Loan",
]


def convert_credit_history(text):
    # ahahha turns "22 Years and 4 Months" into a single number of months
    if pd.isna(text):
        return np.nan

    match = re.search(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(text))
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return np.nan


def clean_group_numeric(series):
    # ahahha same idea as the notebook:
    # clean weird numeric values within each customer group
    valid = series.dropna()
    if len(valid) < 4:
        return series

    q1 = valid.quantile(0.02)
    q3 = valid.quantile(0.98)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    median_value = valid.median()

    series = series.apply(
        lambda x: median_value if pd.notna(x) and (x < lower or x > upper) else x
    )
    return series.fillna(median_value)


def load_data():
    print("1. Loading train and test data...")
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    return train, test


def basic_clean(df):
    # ahahha this is the very first cleanup
    df = df.copy()
    df = df.drop(["ID", "Month", "Name", "SSN"], axis=1, errors="ignore")

    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_credit_history)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.rstrip("_")
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def create_no_of_loan(df):
    # ahahha matching the notebook:
    # build no_of_loan from Type_of_Loan, then drop the original
    if "Type_of_Loan" in df.columns:
        df["no_of_loan"] = df["Type_of_Loan"].fillna("").apply(
            lambda x: 0 if str(x).strip() == "" else str(x).count(",") + 1
        )
        df.drop(["Type_of_Loan", "Num_of_Loan"], axis=1, inplace=True, errors="ignore")
    return df


def clean_numeric_by_customer(df):
    # ahahha same grouped clean you used in the notebook
    if "Customer_ID" not in df.columns:
        return df

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != TARGET]

    for col in numeric_cols:
        df[col] = df.groupby("Customer_ID")[col].transform(clean_group_numeric)
    return df


def fill_main_categoricals(df):
    # ahahha exactly the 3 main categorical fixes from the notebook
    if "Occupation" in df.columns:
        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
    if "Credit_Mix" in df.columns:
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)

    for col in ["Occupation", "Credit_Mix", "Payment_of_Min_Amount"]:
        if col in df.columns and "Customer_ID" in df.columns:
            df[col] = df.groupby("Customer_ID")[col].transform(
                lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
            )
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    return df


def split_payment_behaviour(df):
    # ahahha split Payment_Behaviour into Spent and Value_Payments like the notebook
    if "Payment_Behaviour" not in df.columns:
        return df

    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

    if "Customer_ID" in df.columns:
        df["Payment_Behaviour"] = df.groupby("Customer_ID")["Payment_Behaviour"].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
        )

    mode_value = df["Payment_Behaviour"].mode(dropna=True)
    if not mode_value.empty:
        df["Payment_Behaviour"] = df["Payment_Behaviour"].fillna(mode_value.iloc[0])

    split_cols = df["Payment_Behaviour"].str.split("_", expand=True)
    df["Spent"] = split_cols.iloc[:, 0]
    df["Value_Payments"] = split_cols.iloc[:, 2]
    df.drop(["Payment_Behaviour"], axis=1, inplace=True, errors="ignore")
    return df


def final_feature_cleanup(df, is_train=True):
    # ahahha this is the last bit before modelling
    # turn text labels into numbers where needed
    if "Credit_Mix" in df.columns:
        df["Credit_Mix"] = df["Credit_Mix"].replace({"Bad": 0, "Standard": 1, "Good": 2})
        df["Credit_Mix"] = pd.to_numeric(df["Credit_Mix"], errors="coerce")
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"Yes": 1, "No": 0})
        df["Payment_of_Min_Amount"] = pd.to_numeric(df["Payment_of_Min_Amount"], errors="coerce")
    if "Spent" in df.columns:
        df["Spent"] = df["Spent"].replace({"High": 1, "Low": 0})
        df["Spent"] = pd.to_numeric(df["Spent"], errors="coerce")
    if "Value_Payments" in df.columns:
        df["Value_Payments"] = df["Value_Payments"].replace({"Small": 0, "Medium": 1, "Large": 2})
        df["Value_Payments"] = pd.to_numeric(df["Value_Payments"], errors="coerce")

    # ahahha this is the notebook choice:
    # drop occupation to reduce dummy complexity
    df = df.drop(["Occupation"], axis=1, errors="ignore")
    df = df.drop(["Customer_ID"], axis=1, errors="ignore")

    # fill leftover numeric nulls with medians
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if col != TARGET:
            df[col] = df[col].fillna(df[col].median())

    # ahahha this makes sure XGBoost only sees numeric columns
    for col in df.columns:
        if col != TARGET:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    if is_train:
        df = df.dropna()

    return df


def process_train(train_df):
    print("2. Cleaning training data...")
    train_df = basic_clean(train_df)
    train_df = create_no_of_loan(train_df)
    train_df = clean_numeric_by_customer(train_df)
    train_df = fill_main_categoricals(train_df)
    train_df = split_payment_behaviour(train_df)
    train_df = final_feature_cleanup(train_df, is_train=True)
    print("Processed train shape:", train_df.shape)
    return train_df


def process_test(test_df, train_columns):
    print("3. Cleaning test data with same steps...")
    test_df = basic_clean(test_df)
    test_df = create_no_of_loan(test_df)
    test_df = clean_numeric_by_customer(test_df)
    test_df = fill_main_categoricals(test_df)
    test_df = split_payment_behaviour(test_df)
    test_df = final_feature_cleanup(test_df, is_train=False)

    # ahahha line train and test columns up so the model sees the same input
    test_df = test_df.reindex(columns=train_columns, fill_value=0)
    print("Processed test shape:", test_df.shape)
    return test_df


def build_models():
    return {
        "Logistic Regression": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
            )),
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


def compare_models(X_train, y_train, X_test, y_test):
    print("4. Running the 4 main models...")
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    y_map = {"Poor": 0, "Standard": 1, "Good": 2}
    y_reverse_map = {0: "Poor", 1: "Standard", 2: "Good"}

    for name, model in models.items():
        print(f"   Running CV for {name}...")
        use_y_train = y_train
        use_y_test = y_test

        # ahahha XGBoost wants the target as numbers, so just for this model
        # we temporarily map Poor/Standard/Good to 0/1/2
        if name == "XGBoost":
            use_y_train = y_train.map(y_map)
            use_y_test = y_test.map(y_map)

        scores = cross_val_score(
            model,
            X_train,
            use_y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        cv_results.append({
            "Model": name,
            "CV Mean Accuracy": scores.mean(),
            "CV Std Accuracy": scores.std(),
        })

        model.fit(X_train, use_y_train)
        y_pred = model.predict(X_test)

        if name == "XGBoost":
            y_pred = pd.Series(y_pred).map(y_reverse_map)
            use_y_test = use_y_test.map(y_reverse_map)

        print(f"\n{name}")
        print("Accuracy:", round(accuracy_score(use_y_test, y_pred), 4))
        print(classification_report(use_y_test, y_pred))
        print(confusion_matrix(use_y_test, y_pred))

    cv_results_df = pd.DataFrame(cv_results).sort_values(
        by="CV Mean Accuracy",
        ascending=False
    )

    print("\nCV Results")
    print(cv_results_df)
    return cv_results_df


def tune_random_forest(X_train, y_train):
    print("5. Tuning Random Forest...")
    search_spaces = {
        "n_estimators": (100, 250),
        "max_depth": (5, 20),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 5),
    }

    bayes_search = BayesSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        search_spaces=search_spaces,
        n_iter=10,
        scoring="roc_auc_ovr",
        cv=3,
        n_jobs=1,
        random_state=42,
    )

    bayes_search.fit(X_train, y_train)
    print("Best RF Parameters:", bayes_search.best_params_)
    print("Best RF AUC:", bayes_search.best_score_)
    return bayes_search


def fit_best_rf_and_predict(X_train, y_train, X_test, y_test, X_submission, best_params):
    print("6. Fitting tuned RF and scoring test data...")

    smote = SMOTE(random_state=42)

    # ahahha first fit on the train split so we can see validation performance
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    best_rf = RandomForestClassifier(**best_params, random_state=42)
    best_rf.fit(X_train_smote, y_train_smote)

    y_pred = best_rf.predict(X_test)
    print("\nTuned Random Forest Validation Results")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # ahahha now fit on all the train data and predict the real test csv
    X_all_smote, y_all_smote = smote.fit_resample(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
    )
    final_rf = RandomForestClassifier(**best_params, random_state=42)
    final_rf.fit(X_all_smote, y_all_smote)

    test_predictions = final_rf.predict(X_submission)
    test_predictions_df = pd.DataFrame({
        "Predicted_Credit_Score": test_predictions
    })
    test_predictions_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved predictions to:", OUTPUT_PATH)
    print(test_predictions_df.head())
    return test_predictions_df


def run_workflow():
    train_raw, test_raw = load_data()

    train = process_train(train_raw)
    X = train.drop(TARGET, axis=1)
    y = train[TARGET]

    test = process_test(test_raw, X.columns)

    print("4a. Splitting train/validation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    compare_models(X_train, y_train, X_test, y_test)
    rf_search = tune_random_forest(X_train, y_train)
    fit_best_rf_and_predict(X_train, y_train, X_test, y_test, test, rf_search.best_params_)


if __name__ == "__main__":
    run_workflow()
