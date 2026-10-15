# Suggested B.Tech Project-Report Outline

1. **Abstract** – problem, data, models, key result, limitation (synthetic vs real data).
2. **Introduction** – flood impact in India, motivation, objectives, scope.
3. **Literature survey** – ML for flood prediction (RF, SVM, LSTM), hydrological forecasting, imbalanced classification.
4. **Problem statement & objectives** – next-day 3-class flood-risk classification.
5. **Data description** – variables, sources (NASA POWER, CWC/India-WRIS, flood archives), the synthetic sample and why it was used, schema table.
6. **Methodology** – cleaning, imputation, labelling rule, feature engineering, chronological validation, class imbalance handling (see README §4; cite `src/preprocessing.py`, `src/train.py`).
7. **System design** – architecture diagram (README workflow), module diagram, dashboard screenshots.
8. **Implementation** – tools (Python, pandas, scikit-learn, Matplotlib, Seaborn, Streamlit), code walkthrough.
9. **Results & discussion** – metric table, confusion matrices, feature importance, test timeline, comparison with the majority baseline; discuss errors on the *High* class.
10. **Testing** – `pytest` results, dashboard smoke tests, input-range checks.
11. **Limitations & ethics** – synthetic data, labelling assumptions, no guarantee of real flood events, responsible use.
12. **Conclusion & future scope.**
13. **References** – NASA POWER, CWC, India-WRIS, scikit-learn, Streamlit documentation, papers surveyed.

**Viva questions to prepare:** why chronological split? why macro-F1? why is accuracy misleading here? what is leakage and where did you avoid it? how would you validate on real data? how do Random Forest and Logistic Regression differ?
