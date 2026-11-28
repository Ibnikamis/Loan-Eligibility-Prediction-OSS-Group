import streamlit as st # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import sklearn.ensemble # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore


st.set_page_config(page_title="Loan Prediction App", layout="wide")

st.title("🏦 Loan Eligibility Prediction – Streamlit App")

uploaded_file = st.file_uploader(
    "Upload BANK_CUSTOMER_DATA.csv",
    type=["csv"],
)

if uploaded_file is not None:
    B = pd.read_csv(uploaded_file)
    B.columns = B.columns.str.strip()

    required_columns = [
        "Annual_Income",
        "Gold_Assets",
        "Education",
        "Bank_Balance",
        "Land_Assets",
        "Cibil_Score",
        "Existing_Loan",
        "Loan_Status",
        "Gender",
        "Marital_Status",
        "Customer_ID",
    ]

    missing_columns = [column for column in required_columns if column not in B.columns]
    if missing_columns:
        st.error(
            "The uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    st.subheader("📌 Dataset Preview")
    st.dataframe(B.head())

    st.header("📊 Exploratory Data Analysis")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Loan_Status", data=B, ax=ax)
    ax.set_title("Loan Status Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(B["Cibil_Score"], kde=True, ax=ax)
    ax.set_title("Distribution of CIBIL Score")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Loan_Status", y="Annual_Income", data=B, ax=ax)
    ax.set_title("Annual Income vs Loan Status")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Loan_Status", y="Cibil_Score", data=B, ax=ax)
    ax.set_title("CIBIL Score vs Loan Status")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(B.corr(numeric_only=True), annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    for col in [
        "Gender",
        "Marital_Status",
        "Education",
        "Existing_Loan",
        "Loan_Status",
    ]:
        B[col] = B[col].astype(str).str.strip().str.title()

    label_enc = LabelEncoder()
    for col in ["Existing_Loan", "Loan_Status"]:
        B[col] = label_enc.fit_transform(B[col])

    onehot_enc = OneHotEncoder(drop="first", sparse_output=False)
    onehot_df = pd.DataFrame(
        onehot_enc.fit_transform(B[["Gender", "Marital_Status"]]),
        columns=onehot_enc.get_feature_names_out(["Gender", "Marital_Status"]),
        index=B.index,
    )
    B = pd.concat(
        [B.drop(["Gender", "Marital_Status"], axis=1), onehot_df],
        axis=1,
    )

    edu_order = [["High School", "Graduate", "Postgraduate", "Phd"]]
    ord_enc = OrdinalEncoder(
        categories=edu_order,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    B["Education"] = ord_enc.fit_transform(B[["Education"]])

    numeric_columns = [
        "Annual_Income",
        "Gold_Assets",
        "Bank_Balance",
        "Land_Assets",
        "Cibil_Score",
    ]
    for col in numeric_columns:
        B[col] = pd.to_numeric(B[col], errors="coerce")

    if B[list(numeric_columns + ["Education", "Existing_Loan", "Loan_Status"])].isnull().any().any():
        st.error(
            "The dataset contains missing or invalid numeric values. "
            "Please clean the CSV before training."
        )
        st.stop()

    features = [
        "Annual_Income",
        "Gold_Assets",
        "Education",
        "Bank_Balance",
        "Land_Assets",
        "Cibil_Score",
        "Existing_Loan",
    ]

    X = B[features]
    y = B["Loan_Status"]

    if y.nunique() < 2:
        st.error("Loan_Status must contain at least two different classes.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    rf_model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=7,
        class_weight="balanced",
        random_state=42,
    )
    rf_model.fit(X_train_res, y_train_res)

    st.header("🔍 Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax,
    )
    ax.set_title("Feature Importance in Loan Prediction")
    st.pyplot(fig)
    st.dataframe(importance_df)

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    st.header("📈 Model Performance")

    accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
    precision = sklearn.metrics.precision_score(y_test, y_test_pred, zero_division=0)
    recall = sklearn.metrics.recall_score(y_test, y_test_pred, zero_division=0)
    f1 = sklearn.metrics.f1_score(y_test, y_test_pred, zero_division=0)
    train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**Training Accuracy:** {train_accuracy:.4f}")
    st.write(f"**Testing Accuracy:** {test_accuracy:.4f}")

    cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(sklearn.metrics.classification_report(y_test, y_test_pred, zero_division=0))

    st.header("📋 Final Loan Predictions")

    pred_df = X_test.copy()
    pred_df["Customer_ID"] = B.loc[X_test.index, "Customer_ID"].values
    pred_df["Predicted_Loan_Status"] = y_test_pred
    pred_df["Approval_Probability"] = y_prob
    pred_df["Predicted_Loan_Status"] = pred_df[
        "Predicted_Loan_Status"
    ].map({1: "Yes", 0: "No"})

    final_df = (
        pred_df[
            [
                "Customer_ID",
                "Predicted_Loan_Status",
                "Approval_Probability",
            ]
        ]
        .sort_values(by="Approval_Probability", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(final_df)

    st.header("📘 Summary Report")

    approved = (final_df["Predicted_Loan_Status"] == "Yes").sum()
    rejected = (final_df["Predicted_Loan_Status"] == "No").sum()

    st.write(f"**Total Customers Evaluated:** {len(final_df)}")
    st.write(f"**Approved Customers:** {approved}")
    st.write(f"**Rejected Customers:** {rejected}")

else:
    st.info("📂 Please upload a CSV file to begin.")
