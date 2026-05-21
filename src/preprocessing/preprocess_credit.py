import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Stabilim căile către fișiere
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/preprocessing
src_dir = os.path.dirname(current_dir) # src
base_path = os.path.dirname(src_dir) # Folderul principal (D:/Master/...)

input_path = os.path.join(base_path, 'data', 'credit_eligibility_data.csv')
output_path = os.path.join(base_path, 'data', 'credit_prep_data.csv')
encoder_path = os.path.join(base_path, 'models', 'encoder_credit.pkl')


# 2. Incarcare date
df = pd.read_csv(input_path)


# 3. Tratare missing values - mediana pt numerice si mod pt text
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 4. LabelEncoding (text -> nr) : gen, educaatie, tip_locuinta
categorical_cols = ['gen', 'educatie', 'tip_locuinta']
encoders = {}

for col in categorical_cols:
    le =LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"✅ Coloana '{col}' a fost encodată.")

# 5. Salvare date
df.to_csv(output_path, index=False)

# 6. Salvarea encoderelor
if not os.path.exists(os.path.dirname(encoder_path)):
    os.makedirs(os.path.dirname(encoder_path))
joblib.dump(encoders, encoder_path)
print(f"📂 Encoderele au fost salvate în: {encoder_path}")







