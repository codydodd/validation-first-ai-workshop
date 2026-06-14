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


