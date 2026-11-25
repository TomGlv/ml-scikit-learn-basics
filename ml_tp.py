# ==========================================
# Partie 0 – Imports
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fpdf import FPDF

# --- Classe pour générer le PDF proprement ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, 'Rapport Automatique - TP Machine Learning', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDFReport()
pdf.add_page()
pdf.set_font('helvetica', '', 11)

def ajouter_texte(texte):
    print(texte)
    # On nettoie les caractères spéciaux pour éviter les erreurs d'encodage de base du PDF
    safe_text = texte.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, safe_text)
    pdf.ln(2)

ajouter_texte("=== INTRODUCTION ===")
ajouter_texte("Ce rapport contient les resultats du TP de Machine Learning.\n")

# ==========================================
# Partie 1 & 2 – Chargement et Exploration Iris
# ==========================================
ajouter_texte("--- 1. Exploration du jeu de donnees Iris ---")
iris = load_iris(as_frame=True)
df = iris.frame

ajouter_texte(f"Dimensions du DataFrame : {df.shape}")
res_classes = df['target'].value_counts().to_string()
ajouter_texte(f"Repartition des classes :\n{res_classes}\n")

# ==========================================
# Partie 4 & 5 – Séparation des données
# ==========================================
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# Partie 6 & 7 – Modèle KNN
# ==========================================
ajouter_texte("--- 2. Evaluation du modele KNN (k=3) ---")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred)
ajouter_texte(f"Accuracy du KNN (k=3) : {acc_knn:.4f}")
cm = str(confusion_matrix(y_test, y_pred))
ajouter_texte(f"Matrice de confusion :\n{cm}\n")

# ==========================================
# Partie 8 – Prédire une nouvelle fleur
# ==========================================
ajouter_texte("--- 3. Prediction d'une nouvelle fleur ---")
nouvelle_fleur = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
prediction = model.predict(nouvelle_fleur)
fleur_predite = iris.target_names[prediction[0]]
ajouter_texte(f"Mesures testees : [5.1, 3.5, 1.4, 0.2]")
ajouter_texte(f"Classe predite : {fleur_predite}\n")

# ==========================================
# Partie 10 à 12 – Dataset e-learning
# ==========================================
ajouter_texte("--- 4. Dataset E-Learning & Modeles Predictifs ---")
np.random.seed(42)
n = 200

heures_etude = np.random.normal(10, 3, n).clip(1, 20)
connexions = np.random.poisson(15, n)
quiz = np.random.randint(0, 101, n)

note = (
    0.5 * heures_etude
    + 0.2 * connexions
    + 0.15 * quiz
    + np.random.normal(0, 2, n)
)
note = np.clip(note, 0, 20)

df_elearning = pd.DataFrame({
    "heures_etude": heures_etude,
    "connexions": connexions,
    "quiz": quiz,
    "note": note
})
df_elearning["reussite"] = (df_elearning["note"] >= 10).astype(int)

# Régression Linéaire
X_el = df_elearning[["heures_etude", "connexions", "quiz"]]
y_el_reg = df_elearning["note"]
X_train_el, X_test_el, y_train_el, y_test_el = train_test_split(X_el, y_el_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_el, y_train_el)
y_pred_reg = reg.predict(X_test_el)

ajouter_texte(f"Regression Lineaire (Note) -> MSE : {mean_squared_error(y_test_el, y_pred_reg):.4f} | R2 : {r2_score(y_test_el, y_pred_reg):.4f}")

# Régression Logistique
y_el_clf = df_elearning["reussite"]
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_el, y_el_clf, test_size=0.2, random_state=42, stratify=y_el_clf)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_clf, y_train_clf)
y_pred_clf = clf.predict(X_test_clf)

ajouter_texte(f"Regression Logistique (Reussite) -> Accuracy : {accuracy_score(y_test_clf, y_pred_clf):.4f}\n")

# ==========================================
# Partie 13 – Exercices Bonus
# ==========================================
ajouter_texte("--- 5. Resultats des Exercices Bonus ---")

# Comparaison des k
ajouter_texte("Comparaison de k (KNN) :")
for k in [1, 3, 5, 7]:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    knn_test.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn_test.predict(X_test))
    ajouter_texte(f" - k = {k} -> Accuracy : {acc:.4f}")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))
ajouter_texte(f"KNN avec StandardScaler -> Accuracy : {acc_scaled:.4f}")

# Arbre de décision
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
acc_tree = accuracy_score(y_test, tree_model.predict(X_test))
ajouter_texte(f"DecisionTreeClassifier -> Accuracy : {acc_tree:.4f}")

# Prédiction nouvel étudiant
nouvel_etudiant = pd.DataFrame({"heures_etude": [12], "connexions": [20], "quiz": [80]})
note_predite = reg.predict(nouvel_etudiant)[0]
reussite_predite = clf.predict(nouvel_etudiant)[0]

ajouter_texte(f"\nPrediction pour l'etudiant (12h, 20 connexions, 80 quiz) :")
ajouter_texte(f" - Note predite : {note_predite:.2f} / 20")
ajouter_texte(f" - Reussite predite (1=Oui, 0=Non) : {reussite_predite}")

# ==========================================
# Sauvegarde du fichier PDF
# ==========================================
nom_fichier_pdf = "rapport_tp_ml.pdf"
pdf.output(nom_fichier_pdf)

print(f"\n==================================================")
print(f" SUCCÈS : Le PDF a été généré : '{nom_fichier_pdf}' !")
print(f"==================================================")