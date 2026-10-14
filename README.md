# PartnerLens AI

PartnerLens AI is a privacy-first relationship self-discovery and compatibility project for adults (18+).

## Phase 1 features
- Relationship-preference questionnaire
- Personal partner-quality blueprint
- Synthetic profile dataset
- Explainable compatibility scoring
- Streamlit web app

## Planned portfolio features
- K-Means relationship-style discovery
- NLP communication-pattern analyzer
- AI conversation/scenario practice
- Two-person alignment mode
- Privacy controls and delete-data flow
- Model evaluation and visual analytics

## Run locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install packages:
```bash
pip install -r requirements.txt
```

Generate synthetic data:
```bash
python src/generate_profiles.py
```

Run the app:
```bash
streamlit run app.py
```
