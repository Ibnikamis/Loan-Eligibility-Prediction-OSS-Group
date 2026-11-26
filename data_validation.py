cat > data_validation.py <<'EOF'
import pandas as pd

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Gender",
    "Marital_Status",
    "Education",
    "Annual_Income",
    "Bank_Balance",
    "Cibil_Score",
    "Gold_Assets",
    "Land_Assets",
    "Existing_Loan",
    "Loan_Status",
]

NUMERIC_COLUMNS = [
    "Annual_Income",
    "Bank_Balance",
    "Cibil_Score",
    "Gold_Assets",
    "Land_Assets",
]

ALLOWED_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Marital_Status": ["Single", "Married"],
    "Education": ["High School", "Graduate", "Postgraduate", "Phd"],
    "Existing_Loan": ["Yes", "No"],
    "Loan_Status": ["Yes", "No"],
}


def validate_dataset(df):
    """Validate a loan dataset and return a report."""

    errors = []
    warnings = []

    if df.empty:
        errors.append("The uploaded dataset is empty.")
        return {"valid": False, "errors": errors, "warnings": warnings}

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    if missing_columns:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check missing values
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        warnings.append(
            "Missing values detected: "
            + ", ".join(
                f"{column} ({count})"
                for column, count in missing_values.items()
            )
        )

    # Check duplicate customer IDs
    if df["Customer_ID"].duplicated().any():
        warnings.append("Duplicate Customer_ID values detected.")

    # Check numeric columns
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")

        invalid = converted.isnull() & df[column].notnull()

        if invalid.any():
            errors.append(
                f"{column} contains non-numeric values."
            )

        if (converted < 0).any():
            errors.append(
                f"{column} contains negative values."
            )

    # CIBIL score range
    cibil = pd.to_numeric(df["Cibil_Score"], errors="coerce")

    if ((cibil < 0) | (cibil > 900)).any():
        errors.append("Cibil_Score must be between 0 and 900.")

    # Check categorical values
    for column, allowed_values in ALLOWED_CATEGORIES.items():
        invalid_values = df[column].dropna().unique()

        invalid_values = [
            value for value in invalid_values
            if value not in allowed_values
        ]

        if invalid_values:
            warnings.append(
                f"Unexpected values in {column}: "
                + ", ".join(map(str, invalid_values))
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
EOF