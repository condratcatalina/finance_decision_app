import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DATA_PATH = os.path.join(ROOT_DIR, "data", "credit_eligibility_data_v2.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"📂 Căutăm datele la: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    print("✅ Date încărcate cu succes!")
except FileNotFoundError:
    print(f"❌ EROARE: Nu am găsit fișierul la {DATA_PATH}.")
    exit()

categorical_cols = ['gen', 'educatie', 'tip_locuinta']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"🔹 Mapare {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

coloane_antrenare = [
    'varsta', 'gen', 'educatie', 'venit_anual', 'vechime',
    'tip_locuinta', 'valoare_credit', 'rata_dobanzii',
    'procent_credit_din_venit', 'ani_istoric_credit', 'istoric_neplata'
]

X = df[coloane_antrenare]
y = df['status_credit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Antrenăm modelul cu parametrii optimizați...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n Acuratețe test: {accuracy_score(y_test, y_pred):.2f}")
print("\nRaport de clasificare:\n", classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(MODEL_DIR, "model_credit_rf.pkl"))

joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

print(f"\n GATA! Modelul a fost salvat în: {MODEL_DIR}")