import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Load human‑validated dataset
# ---------------------------------------------------------
"""
Source: Service Canada - Client Insights and Impact Measurement team
899 comments tagged by the team according to themes identified by the client insights team
"""

df = pd.read_excel("predicting_themes/datasets/text_analysis_online_social_security_application-with_results.xlsx",     
    sheet_name=0,
    header=0,
    dtype=str,              # prevents dtype inference that drops rows
    keep_default_na=False   # keeps empty cells as ""
)

# Remove rows that are truly empty (all columns blank)
df = df.dropna(how="all")
print("Rows loaded:", len(df))


# Column B = response text
TEXT_COL = df.columns[1]
responses = df[TEXT_COL].astype(str).fillna("")

# Theme columns = everything after column B
theme_cols = df.columns[2:]
human_labels = df[theme_cols].astype(int)

# ---------------------------------------------------------
# 2. Load keyword dictionary
# ---------------------------------------------------------
"""
Source: Service Canada - Client Insights and Impact Measurement team & PigeonLine's ResearchAI
A dictionary made using PigeonLine's ResearchAI for an online social security application, developed by the client insights team
"""
dict_df = pd.read_excel("predicting_themes/dictionaries/custom_topic_model-online_application.xlsx")

# Each row = theme, columns = keywords
keyword_dict = {
    row["Theme"]: [
        str(row[col]).strip().lower()
        for col in dict_df.columns
        if col != "Theme" and pd.notna(row[col])
    ]
    for _, row in dict_df.iterrows()
}

# ---------------------------------------------------------
# 3. Load pre-modelled data (for comparison) - OPTIONAL
# ---------------------------------------------------------
"""
Source: Service Canada - Client Insights and Impact Measurement team & PigeonLine's ResearchAI
All 899 comments tagged according to the ResearchAI advanced machine learning tool, which lets users choose weights and snapshots that will define thematic modelling.
"""
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
advanced_ml_path = os.path.join(base_dir, "modelled_data", "text_analysis-advanced_ml.xlsx")

# ---------------------------------------------------------
# 3. Normalize text
# ---------------------------------------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9À-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

responses_norm = responses.apply(normalize)

# ---------------------------------------------------------
# 4. Deterministic Keyword‑Dictionary Classifier
# ---------------------------------------------------------
keyword_predictions = pd.DataFrame(0, index=df.index, columns=theme_cols)

for theme in theme_cols:
    keywords = keyword_dict.get(theme, [])
    for kw in keywords:
        if kw:
            keyword_predictions[theme] |= responses_norm.str.contains(re.escape(kw))

keyword_predictions = keyword_predictions.astype(int)

# ---------------------------------------------------------
# 5. Deterministic Machine‑Learning Classifier
# ---------------------------------------------------------
ml_predictions = pd.DataFrame(0, index=df.index, columns=theme_cols)

# TF‑IDF vectorizer (deterministic)
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(responses_norm)

# Train one classifier per theme
for theme in theme_cols:
    y = human_labels[theme]

    # Train/test split (deterministic)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(
        max_iter=200,
        solver="liblinear",
        random_state=42
    )

    # If the theme has fewer than 2 positive samples in the entire dataset,
    # OR if the training split has no positives, skip ML training.
    if y.sum() < 2 or y_train.sum() < 1:
        ml_predictions[theme] = 0
        continue

    clf.fit(X_train, y_train)

    # Predict on full dataset (not just test)
    ml_predictions[theme] = clf.predict(X)





# ---------------------------------------------------------
# 6. Evaluation Function
# ---------------------------------------------------------
def evaluate(name, preds, truth):
    print("\n" + "="*60)
    print(f"RESULTS FOR: {name}")
    print("="*60)
    print(classification_report(truth, preds, zero_division=0))

# ---------------------------------------------------------
# 7. Run Evaluations
# ---------------------------------------------------------
evaluate("Keyword Dictionary Classifier", keyword_predictions, human_labels)
evaluate("Machine Learning Classifier", ml_predictions, human_labels)

# ---------------------------------------------------------
# 8. Optional: Compare keyword vs ML vs human
# ---------------------------------------------------------
summary = pd.DataFrame({
    "Theme": theme_cols,
    "Human Positives": human_labels.sum(),
    "Keyword Positives": keyword_predictions.sum(),
    "ML Positives": ml_predictions.sum(),
})

# % of human positives found
summary["Keyword % of Human"] = summary["Keyword Positives"] / summary["Human Positives"]
summary["ML % of Human"] = summary["ML Positives"] / summary["Human Positives"]

# % of total dataset
total_rows = len(df)
summary["Human % of Dataset"] = summary["Human Positives"] / total_rows
summary["Keyword % of Dataset"] = summary["Keyword Positives"] / total_rows
summary["ML % of Dataset"] = summary["ML Positives"] / total_rows


# ---------------------------------------------------------
# 9. Load external model results for extra comparison
# ---------------------------------------------------------

print("Checking previously modelled data file at:", advanced_ml_path)

modelled_data_predictions = None

if os.path.exists(advanced_ml_path):
    print("Modelled data file detected — loading...")

    adv_df = pd.read_excel(advanced_ml_path)

    # Ensure all theme columns exist — fill missing ones with 0
    for col in theme_cols:
        if col not in adv_df.columns:
            adv_df[col] = 0

    # Now safely extract aligned predictions
    adv_preds = adv_df[theme_cols].astype(int)
    modelled_data_predictions = adv_preds

    # Add to summary
    summary["Modelled data Positives"] = modelled_data_predictions.sum()
    summary["Modelled data % of Human"] = (
        summary["Modelled data Positives"] / summary["Human Positives"]
    )
    summary["Modelled data % of Dataset"] = (
        summary["Modelled data Positives"] / total_rows
    )

    print("Modelled data metrics added to summary.")
else:
    print("No modelled data file found — skipping.")



print(summary)


# ---------------------------------------------------------
# 10. Closeness-to-Human metric (0 = far, 1 = perfect match)
# ---------------------------------------------------------

def closeness(model_pos, human_pos):
    if human_pos == 0:
        return 0  # cannot compute closeness for themes with no human positives
    return max(0, 1 - abs(model_pos - human_pos) / human_pos)

summary["Keyword Closeness"] = summary.apply(
    lambda row: closeness(row["Keyword Positives"], row["Human Positives"]), axis=1
)

summary["ML Closeness"] = summary.apply(
    lambda row: closeness(row["ML Positives"], row["Human Positives"]), axis=1
)

if modelled_data_predictions is not None:
    summary["Modelled data Closeness"] = summary.apply(
        lambda row: closeness(row["Modelled data Positives"], row["Human Positives"]), axis=1
    )



print("\n=== BEST THEMES — KEYWORDS (Closest to Human) ===")
print(summary.sort_values("Keyword Closeness", ascending=False).head(20)[[
    "Theme",
    "Human Positives",
    "Keyword Positives",
    "Keyword Closeness"
]])

print("\n=== BEST THEMES — ML (Closest to Human) ===")
print(summary.sort_values("ML Closeness", ascending=False).head(20)[[
    "Theme",
    "Human Positives",
    "ML Positives",
    "ML Closeness"
]])

if modelled_data_predictions is not None:
    print("\n=== BEST THEMES — Modelled data (Closest to Human) ===")
    print(summary.sort_values("Modelled data Closeness", ascending=False).head(20)[[
        "Theme",
        "Human Positives",
        "Modelled data Positives",
        "Modelled data Closeness"
    ]])




## Notes:

# Example prompt without guidance

'''
You are an expert in qualitative analysis, service design, and thematic coding.

Your task is to analyze a dataset of client responses about their experience with government online social security applications.

You will:

1. Read all responses I provide.
2. Identify a set of discrete, non-overlapping, service‑experience themes.
3. Create a final theme list that:
   - Is concise (15–40 themes)
   - Uses short, clear labels
   - Avoids duplication or overlap
   - Represents *service experience*, not sentiment or emotion
   - Avoids overly granular themes unless necessary
4. For each response, assign a 1 or 0 for each theme:
   - 1 = the response expresses that theme
   - 0 = it does not
   - A response may have multiple themes
   - Do NOT infer meaning beyond what is written
5. Produce a final output table in a format that can be saved directly to Excel:
   - First column: “Response”
   - One column per theme
   - Each row corresponds to one response
   - Values are only 0 or 1
   - No extra commentary
   - No nested structures
   - No markdown formatting
   - Use plain CSV-like text or a clean table

Important constraints:
- Do NOT summarize responses.
- Do NOT rewrite responses.
- Do NOT add sentiment labels.
- Do NOT add explanations.
- Only output the final table, , as xlsx


'''
# - the above produced 'text_analysis-claude_no_dict.xlsx


# Example prompt with previous dictionary

'''
You are an expert in qualitative analysis, service design, and thematic classification.

I am uploading two files:
– File A - custom topic model: A predefined list of themes and related keywords.
– File B - text responses: A dataset of client responses about their experience with government online social security applications.

Your task is to produce a complete, Excel‑ready theme‑assignment matrix.

-----------------------------------------
YOUR OBJECTIVES
-----------------------------------------

1. Read the theme list I provide from the custom topic model.
   - These themes MUST be included in your final output.
   - You MUST attempt to detect these themes in every response.
   - You MAY add new themes if the dataset contains concepts not covered by the predefined list.
   - Any new themes must be:
       • service‑experience related
       • concise
       • non‑overlapping
       • clearly named

2. Read all rows (responses) I provide.

3. For each response, assign a 1 or 0 for every theme:
   - 1 = the response expresses that theme
   - 0 = it does not
   - A response may have multiple themes
   - Do NOT infer meaning beyond what is written
   - Do NOT summarize or rewrite responses

4. Produce a final output table suitable for Excel:
   - First column: “Response”
   - One column per theme (predefined themes first, then any new themes)
   - Each row corresponds to one response
   - Values must be only 0 or 1
   - No markdown formatting
   - No commentary
   - No nested structures
   - Export as xlsx

5. Theme detection rules:
   - Use the keyword lists as hints, not strict rules.
   - Use semantic understanding to detect themes even when keywords are not present.
   - If a response clearly expresses a theme but uses different wording, still mark it as 1.
   - If a response does NOT express a theme, mark it as 0 even if a keyword appears in an unrelated context.

'''




# Example prompt with ML pre-loaded

# - Save a logistic regression model and add to a prompt