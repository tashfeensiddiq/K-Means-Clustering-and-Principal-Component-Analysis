
# ============================================================
# ECE 8908 / ENGI 9808 - Applied Machine Learning & Neural Networks
# Lab 3 - Principal Component Analysis (PCA)
# ============================================================

# ------------------ IMPORTS ------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


# ------------------ LOAD DATASET ------------------

df = pd.read_csv("car_prices_lab3.csv")

print("Dataset loaded successfully!")
print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# ------------------ SEPARATE FEATURES AND TARGET ------------------

# selling_price is the target variable
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

print("\nTarget variable: selling_price")
print("Original input columns:", X.shape[1])


# ------------------ IDENTIFY FEATURE TYPES ------------------

# Identify categorical features
cat_features = X.select_dtypes(
    include=["object", "category"]
).columns

# Identify numerical features
num_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical features:")
print(list(num_features))

print("\nCategorical features:")
print(list(cat_features))


# ------------------ PREPROCESSING ------------------

# Scale numerical features and encode categorical features

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_features
        )
    ]
)

# Apply preprocessing
X_transformed = preprocess.fit_transform(X)


# ------------------ CONVERT TO DATAFRAME ------------------

# Get the new feature names

new_feature_names = (
    list(num_features) +
    list(
        preprocess.named_transformers_["cat"]
        .get_feature_names_out(cat_features)
    )
)

# Convert transformed data into a DataFrame

X_transformed_df = pd.DataFrame(
    X_transformed.toarray()
    if hasattr(X_transformed, "toarray")
    else X_transformed,
    columns=new_feature_names
)

print("\nPreprocessing completed successfully!")

print(
    "\nNumber of features after initial preprocessing:",
    X_transformed.shape[1]
)

print("\nTransformed dataset preview:")
print(X_transformed_df.head())


# ------------------ PCA HYPERPARAMETER EXPLORATION ------------------

# Convert sparse matrix to dense matrix if necessary

X_pca_input = (
    X_transformed.toarray()
    if hasattr(X_transformed, "toarray")
    else X_transformed
)

# Original number of input features

original_features = X_pca_input.shape[1]

print("\nOriginal number of features:", original_features)


# ------------------ COMPONENT VALUES ------------------

# Integer values: 1 to 31
integer_components = list(range(1, 32))

# Variance thresholds provided in Lab 3 instructions
variance_thresholds = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.72, 0.74, 0.76, 0.78, 0.80,
    0.82, 0.84, 0.86, 0.88, 0.90,
    0.92, 0.94, 0.96, 0.98, 0.99
]

component_values = (
    integer_components + variance_thresholds
)


# ------------------ RUN PCA EXPERIMENTS ------------------

results = []

for n in component_values:

    # Initialize PCA
    pca = PCA(
        n_components=n,
        random_state=42
    )

    # Apply PCA
    X_pca = pca.fit_transform(X_pca_input)

    # Number of retained features
    reduced_features = X_pca.shape[1]

    # Cumulative variance explained
    cumulative_variance = (
        pca.explained_variance_ratio_.cumsum()[-1]
    )

    # Calculate percentage of feature savings
    feature_savings = (
        1 - reduced_features / original_features
    ) * 100

    # Store results
    results.append({
        "n_components": n,
        "Number of features after PCA": reduced_features,
        "Cumulative variance": cumulative_variance,
        "% Savings in number of input features": feature_savings
    })


# ------------------ DISPLAY RESULTS ------------------

results_df = pd.DataFrame(results)

print("\n================================================")
print("       PCA HYPERPARAMETER EXPLORATION RESULTS")
print("================================================")

print("\n", results_df.to_string(index=False))


# ------------------ SAVE RESULTS ------------------

results_df.to_csv(
    "PCA_Results_Table.csv",
    index=False
)

print("\n================================================")
print("PCA EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("================================================")

print("\nOriginal features:", original_features)

print("Total experiments:", len(results_df))

print("\nResults saved to PCA_Results_Table.csv")
