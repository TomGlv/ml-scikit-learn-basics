# ml-scikit-learn-basics

> Machine-learning lab with scikit-learn: KNN, decision tree, linear & logistic regression, with an auto-generated PDF report.

## Content
- **Iris classification**: train/test split, **KNN (k=3)**, accuracy, confusion matrix, classification report, prediction on a new sample
- **Decision tree** vs KNN comparison
- **E-learning dataset** (study hours, logins, quiz score):
  - **Linear regression** to predict the grade (MSE, R²)
  - **Logistic regression** to predict pass/fail
- Feature scaling with `StandardScaler`
- Results exported automatically into a **PDF report** (`fpdf`) → `rapport_tp_ml.pdf`

## Stack
Python · pandas · NumPy · scikit-learn · matplotlib · fpdf

## Run
```bash
pip install pandas numpy scikit-learn matplotlib fpdf
python ml_tp.py
```

---
Made by [Tom Galvan](https://tomglv.dev) · Bachelor in AI, ECE Paris
