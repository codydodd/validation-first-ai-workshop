"""
Deterministic Machine Learning Example:
Predicting Binary Outcomes Using a Simple Neural Network

This script demonstrates:
1. Loading and preparing structured data (two dataset options)
2. Encoding + scaling features
3. Splitting data into train/test sets
4. Training a deterministic neural network
5. Evaluating accuracy (TensorFlow + manual)
6. Printing example predictions

Switch between datasets by setting DATASET = "pima" or "bank".
"""

# -----------------------------
# 1. Imports
# -----------------------------
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Deterministic Setup
# -----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------
# 2. Choose Dataset
# -----------------------------
DATASET = "pima"   # <-- change to "bank" or "pima" to switch datasets

# -----------------------------
# 3. Load + Prepare Dataset
# -----------------------------
if DATASET == "pima":
    """
    Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
    Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). 
    IEEE Computer Society Press.
    UCI Machine Learning Repository: hhttps://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    # Pima: all numeric
    dataset = np.loadtxt("predicting_diabetes/datasets/pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, 0:8]
    y = dataset[:, 8]

elif DATASET == "bank":
    """
    Source:
    Moro, S., Cortez, P., & Rita, P. (2014).
    A Data‑Driven Approach to Predict the Success of Bank Telemarketing.
    Decision Support Systems, 62, 22–31.
    UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    """
    # Try semicolon first (official UCI format)
    df = pd.read_csv("predicting_diabetes/datasets/bank-full.csv", sep=";", engine="python")

    # If only one column was parsed, fallback to comma
    if len(df.columns) == 1:
        df = pd.read_csv("predicting_diabetes/datasets/bank-full.csv", sep=",", engine="python")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    print("Detected columns:", df.columns.tolist())

    # Now the label column should exist
    if "y" not in df.columns:
        raise ValueError("Could not find label column 'y' in bank-full.csv")

    # Separate features + label
    y = df["y"].astype(int).values

    X_raw = df.drop(columns=["y"])

    # Identify categorical + numeric columns
    categorical_cols = X_raw.select_dtypes(include=["object"]).columns
    numeric_cols = X_raw.select_dtypes(exclude=["object"]).columns

    # One-hot encode categoricals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X_raw[categorical_cols])

    # Numeric features
    X_num = X_raw[numeric_cols].values

    # Combine
    X = np.hstack([X_num, X_cat])


else:
    raise ValueError("Unknown dataset: choose 'pima' or 'bank'")

# -----------------------------
# 4. Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Define Neural Network
# -----------------------------
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# -----------------------------
# 7. Train Model (Deterministic)
# -----------------------------
es = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=10,
    callbacks=[es],
    verbose=0,
    shuffle=False
)

# -----------------------------
# 8. Evaluate
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"TensorFlow accuracy on test set: {accuracy * 100:.2f}%")

# -----------------------------
# 9. Manual Accuracy
# -----------------------------
predictions = (model.predict(X_test) > 0.5).astype(int)

correct = 0
total = len(y_test)

for i in range(total):
    pred = int(predictions[i].item())
    actual = int(y_test[i].item() if hasattr(y_test[i], "item") else y_test[i])
    if pred == actual:
        correct += 1

manual_accuracy = (correct / total) * 100.0
print(f"Manual accuracy: {manual_accuracy:.2f}% ({correct}/{total} correct)")

# -----------------------------
# 10. Sample Predictions
# -----------------------------
print("\nSample predictions (first 50 rows):")
for i in range(min(50, total)):
    pred = int(predictions[i].item())
    actual = int(y_test[i].item() if hasattr(y_test[i], "item") else y_test[i])
    print(f"Row {i}: Predicted={pred}, Actual={actual}")

# -----------------------------
# 11. Notes
# -----------------------------
# Even sophisticated models rarely exceed 78–82% accuracy without heavy feature engineering.


# Example prompt - initial model

'''
You are a binary classifier. Use which ever model you think is appropriate.
For each row of numeric data in the attached csv, add a new column and output 0 or 1.
Do not explain.
Do not summarize.
Do not add commentary.
Do not change the format.
Only output predictions in a list.
Export as xlsx.
'''

# Example prompt - validating results

'''
You are evaluating a binary classification task.
I am uploading two files:
– File A: GPT‑generated predictions (contains a prediction column).
– File B: the ground‑truth dataset (contains the true diabetes label).

Your tasks are:

1. Load both files and align rows by their original order.  
Do not reorder, sort, or drop rows unless absolutely required.

2. Compare the GPT predictions to the true labels.  
Treat predictions as 0/1 integers.

3. Calculate and output:  
– total number of rows
– number of correct predictions
– number of incorrect predictions
– overall accuracy percentage

4. Produce a row‑by‑row comparison for the first 10 mismatches, in this format:
Row N: predicted X, actual Y, input = [list of feature values]

5. Summarize the pattern of errors.  
Identify whether the GPT model:
– over‑predicts 0 or 1
– misses edge cases
– shows inconsistent logic
– appears to rely on simplistic rules

6. Do not retrain a model.
Do not invent new rules.
Do not change the dataset.
Only evaluate the predictions that already exist.

Begin when both files are uploaded.
'''
# - the above produced both 'bank-full-claud_predictions.xlxs' and 'prima-indians-diabetes-claude_predictions.xlsx'
