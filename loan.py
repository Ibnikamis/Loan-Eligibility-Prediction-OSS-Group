import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Loan Prediction App", layout="wide")

st.title("🏦 Loan Eligibility Prediction – Streamlit App")

uploaded_file = st.file_uploader("Upload BANK_CUSTOMER_DATA.csv", type=["csv"])

if uploaded_file is not None:
    B = pd.read_csv(uploaded_file)
    B.columns = B.columns.str.strip()

    st.subheader("📌 Dataset Preview")
    st.dataframe(B.head())

    # EDA
    st.header("📊 Exploratory Data Analysis")

    # Loan Status Count
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Loan_Status', data=B, ax=ax)
    ax.set_title("Loan Status Distribution")
    st.pyplot(fig)

    # CIBIL Score
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(B['Cibil_Score'], kde=True, ax=ax)
    ax.set_title("Distribution of CIBIL Score")
    st.pyplot(fig)

    # Income vs Loan
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Loan_Status', y='Annual_Income', data=B, ax=ax)
    ax.set_title("Annual Income vs Loan Status")
    st.pyplot(fig)

    # CIBIL vs Loan
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Loan_Status', y='Cibil_Score', data=B, ax=ax)
    ax.set_title("CIBIL Score vs Loan Status")
    st.pyplot(fig)

    # Heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(B.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # ENCODING
    for col in ['Gender', 'Marital_Status', 'Education', 'Existing_Loan', 'Loan_Status']:
        B[col] = B[col].astype(str).str.strip().str.title()

    label_enc = LabelEncoder()

    for col in ['Existing_Loan', 'Loan_Status']:
        B[col] = label_enc.fit_transform(B[col])

    onehot_enc = OneHotEncoder(drop='first', sparse_output=False)

    onehot_df = pd.DataFrame(
        onehot_enc.fit_transform(B[['Gender', 'Marital_Status']]),
        columns=onehot_enc.get_feature_names_out(['Gender', 'Marital_Status']),
        index=B.index
    )

    B = pd.concat(
        [B.drop(['Gender', 'Marital_Status'], axis=1), onehot_df],
        axis=1
    )

    edu_order = [['High School', 'Graduate', 'Postgraduate', 'Phd']]

    ord_enc = OrdinalEncoder(
        categories=edu_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    B['Education'] = ord_enc.fit_transform(B[['Education']])

    # BASELINE FEATURES (Original Model)
    baseline_features = [
        'Annual_Income',
        'Gold_Assets',
        'Education',
        'Bank_Balance',
        'Land_Assets'
    ]

    # IMPROVED FEATURES (Aliyu's Contribution)
    improved_features = [
        'Annual_Income',
        'Gold_Assets',
        'Education',
        'Bank_Balance',
        'Land_Assets',
        'Cibil_Score',       # Aliyu's contribution
        'Existing_Loan'      # Aliyu's contribution
    ]

    X_baseline = B[baseline_features]
    X_improved = B[improved_features]
    y = B['Loan_Status']

    # TRAIN-TEST SPLIT
    train_idx, test_idx = train_test_split(
        B.index,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_baseline_train = X_baseline.loc[train_idx]
    X_baseline_test = X_baseline.loc[test_idx]

    X_improved_train = X_improved.loc[train_idx]
    X_improved_test = X_improved.loc[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # SMOTE - HANDLE CLASS IMBALANCE
    sm_baseline = SMOTE(random_state=42)
    X_baseline_res, y_baseline_res = sm_baseline.fit_resample(
        X_baseline_train, y_train
    )

    sm_improved = SMOTE(random_state=42)
    X_improved_res, y_improved_res = sm_improved.fit_resample(
        X_improved_train, y_train
    )

    # MODEL PARAMETERS
    model_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 7,
        'class_weight': 'balanced',
        'random_state': 42
    }

    # BASELINE MODEL
    baseline_model = RandomForestClassifier(**model_params)
    baseline_model.fit(X_baseline_res, y_baseline_res)

    baseline_pred = baseline_model.predict(X_baseline_test)

    # IMPROVED MODEL - ALIYU'S CONTRIBUTION
    rf_model = RandomForestClassifier(**model_params)
    rf_model.fit(X_improved_res, y_improved_res)

    y_train_pred = rf_model.predict(X_improved_train)
    y_test_pred = rf_model.predict(X_improved_test)
    y_prob = rf_model.predict_proba(X_improved_test)[:, 1]

    # MODEL COMPARISON
    st.header("🆚 Baseline vs Improved Model")

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)

    improved_accuracy = accuracy_score(y_test, y_test_pred)
    improved_precision = precision_score(y_test, y_test_pred, zero_division=0)
    improved_recall = recall_score(y_test, y_test_pred, zero_division=0)
    improved_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Baseline Model': [
            baseline_accuracy,
            baseline_precision,
            baseline_recall,
            baseline_f1
        ],
        'Improved Model': [
            improved_accuracy,
            improved_precision,
            improved_recall,
            improved_f1
        ]
    })

    st.dataframe(
        comparison_df.style.format({
            'Baseline Model': '{:.4f}',
            'Improved Model': '{:.4f}'
        }),
        use_container_width=True
    )

    # METRICS - IMPROVED MODEL
    st.header("📈 Improved Model Performance")

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    st.write(f"**Accuracy:** {improved_accuracy:.4f}")
    st.write(f"**Precision:** {improved_precision:.4f}")
    st.write(f"**Recall:** {improved_recall:.4f}")
    st.write(f"**F1 Score:** {improved_f1:.4f}")
    st.write(f"**Training Accuracy:** {train_accuracy:.4f}")
    st.write(f"**Testing Accuracy:** {test_accuracy:.4f}")

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Improved Model Confusion Matrix")
    st.pyplot(fig)

    # CLASSIFICATION REPORT
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    # FINAL TABLE
    st.header("📋 Final Loan Predictions")

    pred_df = X_improved_test.copy()

    pred_df['Customer_ID'] = B.loc[test_idx, 'Customer_ID'].values
    pred_df['Predicted_Loan_Status'] = y_test_pred
    pred_df['Approval_Probability'] = y_prob

    pred_df['Predicted_Loan_Status'] = pred_df[
        'Predicted_Loan_Status'
    ].map({1: 'Yes', 0: 'No'})

    final_df = pred_df[
        ['Customer_ID', 'Predicted_Loan_Status', 'Approval_Probability']
    ].sort_values(
        by='Approval_Probability',
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(final_df)

    # SUMMARY REPORT
    st.header("📘 Summary Report")

    approved = (final_df["Predicted_Loan_Status"] == "Yes").sum()
    rejected = (final_df["Predicted_Loan_Status"] == "No").sum()

    st.write(f"**Total Customers Evaluated:** {len(final_df)}")
    st.write(f"**Approved Customers:** {approved}")
    st.write(f"**Rejected Customers:** {rejected}")

else:
    st.info("📂 Please upload a CSV file to begin.")