import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

from data_validation import validate_dataset


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Eligibility Prediction System")
st.write("Upload your loan dataset to analyze and predict loan eligibility.")


# =========================================================
# DATA UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:

    B = pd.read_csv(uploaded_file)

    # Clean column names
    B.columns = B.columns.str.strip()

    # =====================================================
    # DATASET VALIDATION
    # =====================================================

    st.header("🛡️ Dataset Quality Validation")

    validation = validate_dataset(B)

    if validation["errors"]:

        st.error("Dataset validation failed!")

        for error in validation["errors"]:
            st.error(error)

        st.stop()

    if validation["warnings"]:

        st.warning("Dataset validation warnings:")

        for warning in validation["warnings"]:
            st.warning(warning)

    st.success("✅ Dataset validation passed!")

    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    st.subheader("📌 Dataset Preview")

    st.dataframe(B.head())


    # =====================================================
    # EXPLORATORY DATA ANALYSIS
    # =====================================================

    st.header("📊 Exploratory Data Analysis")

    # Loan Status Distribution
    st.subheader("Loan Status Distribution")

    fig, ax = plt.subplots()

    sns.countplot(
        data=B,
        x="Loan_Status",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


    # CIBIL Score Distribution
    if "Cibil_Score" in B.columns:

        st.subheader("CIBIL Score Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            data=B,
            x="Cibil_Score",
            kde=True,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # Annual Income vs Loan Status
    if "Annual_Income" in B.columns:

        st.subheader("Annual Income vs Loan Status")

        fig, ax = plt.subplots()

        sns.boxplot(
            data=B,
            x="Loan_Status",
            y="Annual_Income",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # CIBIL Score vs Loan Status
    if "Cibil_Score" in B.columns:

        st.subheader("CIBIL Score vs Loan Status")

        fig, ax = plt.subplots()

        sns.boxplot(
            data=B,
            x="Loan_Status",
            y="Cibil_Score",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # Correlation Heatmap
    st.subheader("Numerical Correlation Heatmap")

    numeric_data = B.select_dtypes(include=np.number)

    if not numeric_data.empty:

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    # =====================================================
    # ADDITIONAL EDA
    # =====================================================

    display_eda_improvements(B)


    # =====================================================
    # DATA PREPROCESSING
    # =====================================================

    st.header("⚙️ Data Preprocessing")

    # Clean categorical values
    categorical_columns = [
        "Gender",
        "Marital_Status",
        "Education",
        "Existing_Loan",
        "Loan_Status"
    ]

    for column in categorical_columns:

        if column in B.columns:

            B[column] = (
                B[column]
                .astype(str)
                .str.strip()
                .str.title()
            )


    # =====================================================
    # LABEL ENCODING
    # =====================================================

    existing_loan_encoder = LabelEncoder()

    B["Existing_Loan"] = existing_loan_encoder.fit_transform(
        B["Existing_Loan"]
    )


    loan_status_encoder = LabelEncoder()

    B["Loan_Status"] = loan_status_encoder.fit_transform(
        B["Loan_Status"]
    )


    # =====================================================
    # ORDINAL ENCODING FOR EDUCATION
    # =====================================================

    edu_order = [
        ["High School", "Graduate", "Postgraduate", "Phd"]
    ]

    ord_enc = OrdinalEncoder(
        categories=edu_order
    )

    B["Education"] = ord_enc.fit_transform(
        B[["Education"]]
    )


    # =====================================================
    # BASELINE AND IMPROVED FEATURES
    # =====================================================

    # ORIGINAL FEATURES (BASELINE MODEL)

    baseline_features = [
        "Annual_Income",
        "Gold_Assets",
        "Education",
        "Bank_Balance",
        "Land_Assets"
    ]


    # IMPROVED FEATURES (ALIYU'S CONTRIBUTION)

    improved_features = [
        "Annual_Income",
        "Gold_Assets",
        "Education",
        "Bank_Balance",
        "Land_Assets",
        "Cibil_Score",
        "Existing_Loan"
    ]


    X_baseline = B[baseline_features]

    X_improved = B[improved_features]

    y = B["Loan_Status"]


    # =====================================================
    # TRAIN-TEST SPLIT
    # =====================================================

    st.header("🤖 Model Training")

    train_idx, test_idx = train_test_split(
        B.index,
        test_size=0.3,
        random_state=42,
        stratify=y
    )


    # Baseline data split

    X_baseline_train = X_baseline.loc[train_idx]

    X_baseline_test = X_baseline.loc[test_idx]


    # Improved data split

    X_improved_train = X_improved.loc[train_idx]

    X_improved_test = X_improved.loc[test_idx]


    # Target split

    y_train = y.loc[train_idx]

    y_test = y.loc[test_idx]


    # =====================================================
    # SMOTE - HANDLE CLASS IMBALANCE
    # =====================================================

    st.subheader("⚖️ Handling Class Imbalance with SMOTE")


    # SMOTE for baseline model

    sm_baseline = SMOTE(random_state=42)

    X_baseline_res, y_baseline_res = (
        sm_baseline.fit_resample(
            X_baseline_train,
            y_train
        )
    )


    # SMOTE for improved model

    sm_improved = SMOTE(random_state=42)

    X_improved_res, y_improved_res = (
        sm_improved.fit_resample(
            X_improved_train,
            y_train
        )
    )


    # =====================================================
    # BASELINE MODEL
    # =====================================================

    st.subheader("📌 Baseline Random Forest Model")

    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=7,
        class_weight="balanced",
        random_state=42
    )

    rf_baseline.fit(
        X_baseline_res,
        y_baseline_res
    )


    y_baseline_pred = rf_baseline.predict(
        X_baseline_test
    )


    # =====================================================
    # IMPROVED MODEL
    # =====================================================

    st.subheader("🚀 Improved Random Forest Model")

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=7,
        class_weight="balanced",
        random_state=42
    )

    rf_model.fit(
        X_improved_res,
        y_improved_res
    )


    y_test_pred = rf_model.predict(
        X_improved_test
    )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.header("📈 Model Performance Comparison")


    baseline_accuracy = accuracy_score(
        y_test,
        y_baseline_pred
    )

    baseline_precision = precision_score(
        y_test,
        y_baseline_pred,
        zero_division=0
    )

    baseline_recall = recall_score(
        y_test,
        y_baseline_pred,
        zero_division=0
    )

    baseline_f1 = f1_score(
        y_test,
        y_baseline_pred,
        zero_division=0
    )


    improved_accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    improved_precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    improved_recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    improved_f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0
    )


    comparison_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Baseline Model": [
            baseline_accuracy,
            baseline_precision,
            baseline_recall,
            baseline_f1
        ],

        "Improved Model": [
            improved_accuracy,
            improved_precision,
            improved_recall,
            improved_f1
        ]

    })


    st.dataframe(
        comparison_df.style.format({
            "Baseline Model": "{:.2%}",
            "Improved Model": "{:.2%}"
        }),
        use_container_width=True
    )


    # =====================================================
    # IMPROVED MODEL METRICS
    # =====================================================

    st.header("📊 Improved Model Evaluation")


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Accuracy",
        f"{improved_accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{improved_precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{improved_recall:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{improved_f1:.2%}"
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    plt.close(fig)


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_test_pred,
        zero_division=0
    )

    st.text(report)


    # =====================================================
    # LOAN PREDICTIONS
    # =====================================================

    st.header("🔮 Loan Prediction Results")


    pred_df = B.copy()


    pred_df["Predicted_Loan_Status"] = (
        rf_model.predict(X_improved)
    )


    pred_df["Approval_Probability"] = (
        rf_model.predict_proba(X_improved)[:, 1]
    )


    pred_df["Predicted_Loan_Status"] = (
        pred_df["Predicted_Loan_Status"]
        .map({1: "Yes", 0: "No"})
    )


    final_df = pred_df[
        [
            "Customer_ID",
            "Predicted_Loan_Status",
            "Approval_Probability"
        ]
    ].sort_values(
        by="Approval_Probability",
        ascending=False
    ).reset_index(drop=True)


    st.dataframe(
        final_df,
        use_container_width=True
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.header("📋 Prediction Summary")


    approved = (
        pred_df["Predicted_Loan_Status"] == "Yes"
    ).sum()


    rejected = (
        pred_df["Predicted_Loan_Status"] == "No"
    ).sum()


    col1, col2 = st.columns(2)


    col1.metric(
        "Predicted Approved Loans",
        approved
    )


    col2.metric(
        "Predicted Rejected Loans",
        rejected
    )


    st.success(
        "Loan prediction analysis completed successfully!"
    )


else:

    st.info(
        "👆 Please upload a CSV dataset to get started."
    )