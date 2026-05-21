import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# 1. Configurarea căilor către datele pregătite
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'credit_prep_data.csv')
models_dir = os.path.join(base_path, 'models')


# 2. Incarcare date
df = pd.read_csv(data_path)

# 3. Separare variabile
# x = toate col de la varsta la istoric_neplata
# y = coloana status credit (target)
x = df.drop('status_credit', axis=1)
y = df['status_credit']


# 4. Impartire train-test (80%-20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=42)

# 5. Scalare date (media sa fie 0 si deviatia standard 1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 6. Antrenare model (max_iter = 1000 pt ca alg sa aiba timp sa gaseasca sol optima)
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)


# 7. Evaluare performanta model
y_pred = model.predict(x_test_scaled)
print("--- Raport Final de Clasificare (Credit Risk) ---")
print(classification_report(y_test, y_pred))


# 8. Salvarea modelului și a scalerului în folderul 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(model, os.path.join(models_dir, 'model_credit_rf.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler_credit.pkl'))
