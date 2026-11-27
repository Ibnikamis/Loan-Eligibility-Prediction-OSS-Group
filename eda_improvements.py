import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_eda_improvements(df):

    st.header("📊 Additional Data Analysis")

    # Loan approval by education
    st.subheader("Loan Approval by Education")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Education",
        hue="Loan_Status",
        ax=ax
    )

    ax.set_title("Loan Status by Education Level")
    ax.set_xlabel("Education")
    ax.set_ylabel("Number of Applicants")
    plt.xticks(rotation=30)

    st.pyplot(fig)

    # Loan approval by existing loan
    st.subheader("Loan Approval and Existing Loan")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Existing_Loan",
        hue="Loan_Status",
        ax=ax
    )

    ax.set_title("Loan Status by Existing Loan")
    ax.set_xlabel("Existing Loan")
    ax.set_ylabel("Number of Applicants")

    st.pyplot(fig)

    # Income distribution by loan status
    st.subheader("Income Distribution by Loan Status")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="Annual_Income",
        hue="Loan_Status",
        kde=True,
        ax=ax
    )

    ax.set_title("Income Distribution by Loan Status")
    st.pyplot(fig)

    # Assets comparison
    st.subheader("Applicant Assets by Loan Status")

    asset_columns = [
        "Gold_Assets",
        "Land_Assets",
        "Bank_Balance"
    ]

    melted_df = df.melt(
        id_vars="Loan_Status",
        value_vars=asset_columns,
        var_name="Asset_Type",
        value_name="Asset_Value"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.boxplot(
        data=melted_df,
        x="Asset_Type",
        y="Asset_Value",
        hue="Loan_Status",
        ax=ax
    )

    ax.set_title("Assets Comparison by Loan Status")
    st.pyplot(fig)
