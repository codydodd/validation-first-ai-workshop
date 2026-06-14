import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# 1. DATASET LOADERS — LOCAL FILES
# ---------------------------------------------------------

def load_wine_quality_local():
    """
    Source:
    Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
    Modeling wine preferences by data mining from physicochemical properties.
    Decision Support Systems, 47(4), 547–553.
    UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    """
    path = "predicting_drivers/datasets/winequality.csv"
    df = pd.read_csv(path)
    target = "quality"
    return df, target

def load_airline_satisfaction_local():
    """
    Source:
    Invistico Airline Passenger Satisfaction Dataset.
    Originally published on Kaggle but the authenticity of the data is not verified. Used as an example not as real data.
    Kaggle Dataset URL: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
    """
    path = "predicting_drivers/datasets/Invistico_Airline.csv"
    df = pd.read_csv(path)

    # Convert satisfaction to numeric (0/1)
    df["satisfaction"] = df["satisfaction"].map({
        "dissatisfied": 0,
        "satisfied": 1
    })

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).dropna()

    target = "satisfaction"
    return df, target

# ---------------------------------------------------------
# 2. SELECT DATASET HERE
# ---------------------------------------------------------

df, target_col = load_wine_quality_local()            # <-- swap here
# df, target_col = load_airline_satisfaction_local()      # <-- swap here

print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")
print(f"Predicting target: {target_col}")

# ---------------------------------------------------------
# 3. PREPARE DATA
# ---------------------------------------------------------

X = df.drop(columns=[target_col])
y = df[target_col]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. TRAIN RANDOM FOREST
# ---------------------------------------------------------

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. MODEL PERFORMANCE
# ---------------------------------------------------------

y_pred = rf.predict(X_test)

print("\n=== MODEL PERFORMANCE ===")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCES (GINI)
# ---------------------------------------------------------

gini_importances = pd.Series(
    rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\n=== TOP DRIVERS — Gini Importance ===")
for feature, score in gini_importances.items():
    print(f"{feature:30s}  {score:.4f}")

# ---------------------------------------------------------
# 7. PERMUTATION IMPORTANCE (MORE RELIABLE)
# ---------------------------------------------------------

perm = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_importances = pd.Series(
    perm.importances_mean,
    index=feature_names
).sort_values(ascending=False)

print("\n=== TOP DRIVERS — Permutation Importance ===")
for feature, score in perm_importances.items():
    print(f"{feature:30s}  {score:.4f}")

# ---------------------------------------------------------
# 8. COMBINED DRIVER TABLE
# ---------------------------------------------------------

driver_table = pd.DataFrame({
    "Gini Importance": gini_importances,
    "Permutation Importance": perm_importances
}).sort_values("Permutation Importance", ascending=False)

print("\n=== COMBINED DRIVER TABLE (Ranked) ===")
print(driver_table)






# Example prompt - drivers analysis

'''
You are an expert data scientist specializing in feature importance, regression, classification, and drivers analysis.

I will provide a dataset in CSV or Excel format. Your job is to:

1. Inspect the dataset and identify:
   - The target column (I will tell you which one)
   - The predictor columns
   - Whether the task is regression or classification

2. Clean and prepare the data:
   - Convert categorical variables to numeric (one-hot encoding)
   - Convert the target column to numeric if needed
   - Drop rows with missing target values
   - Keep all predictors unless I explicitly ask you to remove some

3. Fit two models:
   A. A linear model (Linear Regression or Logistic Regression)
   B. A tree-based model (Random Forest Regressor or Classifier)

4. Produce a drivers analysis:
   - For the linear model: output coefficients, ranked from strongest to weakest
   - For the Random Forest: output feature importances, ranked from strongest to weakest
   - For both models: show the top 10 drivers

5. Output format:
   - A clean, structured table
   - Columns: Feature, Linear Coefficient (or Odds Ratio), Random Forest Importance
   - Sorted by Random Forest importance (descending)
   - No prose, no commentary, no markdown formatting
   - Just the table

6. Additional requirements:
   - If the target is categorical, treat it as classification
   - If the target is numeric, treat it as regression
   - Standardize numeric predictors for the linear model
   - Do NOT standardize predictors for the Random Forest
   - Use train/test split to avoid leakage
   - Report model performance (R² or Accuracy)

'''

# example output from Claude 4.6 sonnet for wine quality
'''
Linear Regression R²: 0.3330

Random Forest R²: 0.5578
Feature                Linear Coefficient   Random Forest Importance

alcohol                0.318292             0.244033

volatile acidity       -0.218357            0.127511

free sulfur dioxide    0.112985             0.088677

sulphates              0.118364             0.081292

total sulfur dioxide   -0.152251            0.076724

residual sugar         0.209434             0.070450

pH                     0.064433             0.066413

chlorides              -0.012662            0.065170

citric acid            -0.013526            0.061730

density                -0.156364            0.060495
'''