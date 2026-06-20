# AI Workshop — Deterministic vs Generative Systems

This repository contains the hands‑on activities for the 3‑hour workshop.
Each activity demonstrates the difference between deterministic AI (regression, ML, rules) and non‑deterministic AI (GPT/LLMs).

1. Create and activate a virtual environment

Ubuntu:
```
python3 -m venv .workshop-venv3-12
source .workshop-venv3-12/bin/activate
```

Windows (PowerShell):
```
python -m venv .workshop-venv3-12
.workshop-venv3-12\Scripts\activate
```

2. Install dependencies

From the project root:

`pip install -r requirements.txt`


3. Run the activities

Each folder contains its own script(s). Run them directly from the project root.



## Activity 1 — Predicting Diabetes (Deterministic ML)

`python predicting_diabetes/machine_learning.py`

This activity introduces deterministic ML using the Pima Indians Diabetes dataset.

What to observe:

- Deterministic ML produces stable, reproducible predictions.

- Compare Python results to an LLM asked to predict diabetes without labels.

- Then compare again when the LLM is given labeled data.

- Try the same with the larger bank dataset.

- Notice that some LLMs (e.g., Claude) get close only because they run Python tools, not because they “understand” the data.

This demonstrates the difference between simulation and computation.

## Activity 2 — Drivers Analysis with Random Forests

`python predicting_drivers/random_forest.py`

Start with the Wine Quality dataset, then the Airline Satisfaction dataset.

What to observe:

- Random Forests provide ranked feature importances.

- Compare these to LLM‑generated “drivers.”

- Some LLMs get close only when they execute real Python, not when they guess.

- Others hallucinate drivers entirely.

This shows why deterministic systems are essential for drivers analysis.

## Activity 3 — Predicting Themes (Dictionary vs ML vs LLM)

`python predicting_themes/predicting_themes.py`

This activity uses a dataset of ~900 human‑tagged comments.

What to observe:

- Run a dictionary‑based topic model using custom_topic_model-online_application.xlsx.

- Then run a machine learning classifier trained on human tags.

- Compare both to a pre‑tagged dataset from a more complex ML model.

- Finally, ask an LLM to replicate the tagging.

Participants will see:

- Deterministic dictionary methods are transparent but brittle.

- Generic ML models are not necessarily as accurate as keywords, without extensive coding.

- More advanced / managed ML models can get remarkably accurate with limited tagging.

- Without python, LLMs are inconsistent, often drifting or hallucinating categories.

This demonstrates the limits of generative systems for structured classification.

## Workshop Learning Outcomes

By the end of the workshop, participants will understand:

- Why deterministic systems are essential for reliability, auditability, and reproducibility.

- Why LLMs are non‑deterministic and can deviate from instructions.

- Why some LLMs (e.g., Claude) appear “better” only because they rely on deterministic tools behind the scenes.

- Why validation‑first AI architectures combine deterministic ML with LLMs as narrators, or sometimes orchestrators, but not predictors.

## Other reflections

What Participants Learned (and felt)
1. Deterministic vs Generative is not a theory — it’s observable

We saw that:
- Derministic Python model produced the same answer every time
- LLM attempts produced 30 different answers from 30 people
- Claude only “got close” because it used agentic Python execution
- Copilot sometimes guessed, sometimes drifted, sometimes tried Python, sometimes didn’t

This is the perfect demonstration of LLM non‑determinism.

2. We discovered contamination and leakage firsthand

Because we used PIMA — a public dataset — we saw:

- Claude and Copilot “finding” the headers online

- Meaning: LLMs don’t just use the file — they use the internet

- Which means predictions can be polluted, biased, or inconsistent

This is the best possible demonstration of ungrounded inference.

3. We learned that LLMs don’t “do math” — they simulate it

Your diabetes activity showed:

- LLMs simulate regression

- Claude delegates to Python

- Copilot sometimes tries, sometimes doesn’t

- is a substitute for deterministic computation

This is the exact distinction we wanted to internalize.

4. We saw that “drivers analysis” is a decision tree, not a prompt

Every LLM:

- Chose different preprocessing
- Chose different encodings
- Chose different models
- Chose different hyperparameters
- Produced different importances

While your Random Forest:

- Produced the same importances
- Every time
- For everyone

This is the perfect demonstration of model reproducibility.

5. We learned that LLMs struggle with structured classification

Our theme‑tagging activity was impactful because it showed:

- Dictionaries = transparent, surprisingly strong for some themes
- Naive ML = weak without feature engineering
- Weighted ML = excellent, fast, reliable
- Claude = powerful but slow, expensive, and fragile
- Copilot = winging it

This is the exact lesson of validation‑first AI.

6. We saw the future: ensemble systems

- We didn’t just learn “LLMs are flawed.”
- We learned:
- “LLMs are one component in a larger architecture — and the deterministic components do heavy lifting.”

This is the core of AI ensemble design. Because we didn’t just learn the thesis —
we collectively experimented on it.

- We felt the instability.
- We felt the drift.
- We felt the inconsistency.
- We felt the determinism of Python.
- We felt the reliability of ML.
- We felt the fragility of LLMs.

And once someone feels the architecture, they never forget it.

### What we achieved

- A mental model for when to trust LLMs
- A mental model for when not to
- A clear understanding of deterministic vs generative
- A lived experience of reproducibility vs drift
- A practical sense of how to build hybrid AI systems
- A realistic view of Claude as “the optimistic middle ground”
- A sense of empowerment — not fear

This is exactly what the field needs right now.

