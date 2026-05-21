import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cluster_customer_data.csv')

cols_monetare = [
    'BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'
]

cols_frecventa = [
    'BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
    'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT'
]

for col in cols_monetare:
    upper_limit = df[col].quantile(0.95)
    df[col] = df[col].clip(upper=upper_limit)

print("✅ Outlier-ii au fost plafonați la percentila 95.")

scaler = StandardScaler()

toate_coloanele = cols_monetare + cols_frecventa
df_scaled = df.copy()
df_scaled[toate_coloanele] = scaler.fit_transform(df[toate_coloanele])

print("✅ Datele au fost standardizate (Z-score).")

df_scaled.to_csv('date_prep_clustering.csv', index=False)

print("📂 Fișierul 'date_prep_clustering.csv' a fost salvat.")

print(df_scaled.head())