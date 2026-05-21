import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import joblib
import os

# 1. Stabilirea căilor dinamice
# __file__ este calea către acest script (src/training/train_clustering_model.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Mergem două niveluri sus pentru a ajunge la rădăcina proiectului
base_path = os.path.dirname(os.path.dirname(current_dir))

data_input_path = os.path.join(base_path, 'data', 'date_prep_clustering.csv')
model_output_path = os.path.join(base_path, 'models', 'model_clustering_kmeans.pkl')

# 2. Încărcare date
if not os.path.exists(data_input_path):
    print(f" Eroare: Nu s-a găsit fișierul de date la: {data_input_path}")
else:
    df_scaled = pd.read_csv(data_input_path)

    # 3. Elbow Method pentru găsirea nr. de K optim
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    # 4. Vizualizare grafic Elbow (Opțional pentru debugging)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Metoda Elbow pentru Segmentare Clienți')
    plt.xlabel('Nr. clustere')
    plt.ylabel('WCSS (Inerția)')
    plt.show()

    # 5. Antrenare model final (Alegem 4 clustere conform analizei tale)
    n_clusters_optim = 4
    kmeans_final = KMeans(n_clusters=n_clusters_optim, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(df_scaled)

    # --- SECȚIUNE NOUĂ: TESTARE PERFORMANȚĂ MODEL ---
    print("\n--- Evaluarea Performanței Modelului ---")

    # A. Scorul Silhouette (Măsoară cât de similar este un punct cu propriul cluster față de alte clustere)
    # Valori între -1 și 1. Cu cât e mai aproape de 1, cu atât clusterele sunt mai bine separate.
    sil_score = silhouette_score(df_scaled, clusters)
    print(f"Scorul Silhouette: {sil_score:.4f}")

    # B. Indicele Davies-Bouldin (Măsoară separarea dintre clustere)
    # Cu cât valoarea este mai mică (mai aproape de 0), cu atât segmentarea este mai bună.
    db_index = davies_bouldin_score(df_scaled, clusters)
    print(f"Indicele Davies-Bouldin: {db_index:.4f}")

    # C. Inerția finală (WCSS)
    print(f"Inerția finală (WCSS): {kmeans_final.inertia_:.2f}")
    # -----------------------------------------------

    # 6. Adăugăm eticheta de cluster la date
    df_scaled['Cluster_ID'] = clusters

    # 7. Salvare model în folderul 'models'
    if not os.path.exists(os.path.dirname(model_output_path)):
        os.makedirs(os.path.dirname(model_output_path))

    joblib.dump(kmeans_final, model_output_path)

    print(f" Modelul K-Means a fost antrenat cu {n_clusters_optim} clustere.")
    print(f" Model salvat în: {model_output_path}")

    # Afișăm distribuția utilizatorilor
    print("\nNumăr de clienți în fiecare cluster:")
    print(df_scaled['Cluster_ID'].value_counts())