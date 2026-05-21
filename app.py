import os
import jwt
import re
import io
import csv
import json
import pandas as pd
import numpy as np
import joblib
import time
import hashlib
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, render_template
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps
from bson import ObjectId
from groq import Groq
import requests
from bs4 import BeautifulSoup
import urllib.request
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from database import db

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
CORS(app)
bcrypt = Bcrypt(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHED_BNR_RATE = None
LAST_BNR_FETCH = None

GOOGLE_CLIENT_ID = "726233842111-lb660egqoidifi85encp4e1dk4baqn2e.apps.googleusercontent.com"


# --- ÎNCĂRCARE MODELE ---
def load_model(path, name):
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f" Modelul {name} a fost încărcat!")
            return model
        print(f" Fișierul {name} lipsește la: {path}")
    except Exception as e:
        print(f" Eroare la încărcarea {name}: {e}")
    return None

rf_model = load_model(os.path.join(BASE_DIR, "models", "model_credit_rf.pkl"), "Random Forest")
kmeans_model = load_model(os.path.join(BASE_DIR, "models", "model_clustering_kmeans.pkl"), "K-Means")


# --- DECORATOR SECURITATE ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({"message": "Token-ul lipsește!"}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = db.users.find_one({"_id": ObjectId(data['user_id'])})
            if not current_user:
                return jsonify({"message": "Utilizator invalid!"}), 401
        except Exception as e:
            return jsonify({"message": "Eroare sesiune!", "error": str(e)}), 401
        return f(current_user, *args, **kwargs)
    return decorated


# --- FUNCȚII HELPER ---
def anonymize_details(text):
    if not text: return ""

    # 1. Protecție IBAN (Format generic: 2 litere + 2 cifre + 10-30 caractere alfanumerice)
    text = re.sub(r'[A-Z]{2}\d{2}[A-Z\d]{10,30}', '[IBAN_PROTEJAT]', text)

    # 2. Protecție Numere Card (Ex: **** 1234 sau 4242 **** 1234)
    text = re.sub(r'(\d{4}\s?\*+\s?\d{4})|(\*+\s?\d{4})', '[CARD_PROTEJAT]', text)

    # 3. Protecție Nume Persoane după cuvinte cheie specifice extraselor bancare
    keywords = ['Ordonator:', 'Beneficiar:', 'Titular cont:', 'Din contul:', 'In contul:', 'Catre:', 'De la:']
    for kw in keywords:
        pattern = rf"(?i){kw}\s*[^,|;\n]+"
        text = re.sub(pattern, f"{kw} [NUME_PROTEJAT]", text)

    return text


def classify_with_ai(details, tip):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""Ești un asistent financiar care extrage date STRICT în formatul: CATEGORIE | NUME_CURAT
NU scrie explicații. NU folosi cuvântul 'CATEGORIE' în răspuns.

### LOGICĂ VENITURI (Dacă TIP = VENIT):
1. SALARIU: Verifică dacă în detalii apar: 'SALARIU', 'SALARY', 'CIM', 'VENIT LUNAR'.
   - Dacă DA: Venit | Salariu
   - Dacă NU: Venit | Incasare

### LOGICĂ CHELTUIELI (Dacă TIP = CHELTUIALĂ):
- MÂNCARE: Supermarket (Lidl, Mega Image, Kaufland, Carrefour, Profi, Penny, Auchan).
- IEȘIRI: Restaurante, Livrări (Glovo, Tazz, Starbucks, McDonald's, KFC, Bolt Food).
- SHOPPING: Haine, Mall, Electronice (Emag, Zara, H&M, Dedeman, Ikea, Fashion Days).
- UTILITĂȚI: Facturi (Enel, Digi, Orange, Apa Nova, Vodafone).
- TRANSPORT: Benzinarie, Uber, Bolt, Taxi, Metrorex (OMV, Petrom, Rompetrol).
- LIFESTYLE: Abonamente (Netflix, Spotify, Gym, Youtube).
- TRANSFERURI: Transferuri bancare catre persoane, Revolut P2P.
- ALTELE: Comisioane, Taxe, ATM.

### EXEMPLE:
- Detalii: SALARIU LUNA DECEMBRIE | Tip: VENIT => Venit | Salariu
- Detalii: Incasare Revolut de la [NUME_PROTEJAT] | Tip: VENIT => Venit | Incasare
- Detalii: LIDL BUCURESTI | Tip: CHELTUIALĂ => Mâncare | Lidl

Tranzacție de procesat:
DETALII: {details}
TIP: {tip}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        rezultat = completion.choices[0].message.content.strip().split('\n')[0]
        rezultat = rezultat.replace('"', '').replace('.', '').replace('CATEGORIE | ', '')
        return rezultat
    except Exception as e:
        print(f" Eroare AI: {e}")
        return "Altele | Tranzacție"


def get_live_bnr_rate():
    global CACHED_BNR_RATE, LAST_BNR_FETCH

    # 1. Verificare Cache (valabilitate 24 de ore)
    if CACHED_BNR_RATE and LAST_BNR_FETCH and (datetime.now() - LAST_BNR_FETCH) < timedelta(hours=24):
        return CACHED_BNR_RATE

    print("🔍 [BNR] Interogăm agregatorul financiar pentru rata de politică monetară...")
    try:
        req = urllib.request.Request(
            "https://www.curs.ro/",
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')

        soup = BeautifulSoup(html, 'html.parser')
        text_complet = soup.get_text().lower()

        # Căutăm mențiunea dobânzii BNR și extragem primul procent aflat în proximitate (max 150 caractere)
        match = re.search(r'(?:doband[aă]\s+bnr|politic[aă]\s+monetar[aă])[^\d%]{0,150}(\d+([.,]\d+)?)\s*%',
                          text_complet)

        if match:
            num_str = match.group(1).replace(',', '.')
            live_rate = float(num_str)

            CACHED_BNR_RATE = live_rate
            LAST_BNR_FETCH = datetime.now()
            print(f" [BNR] SUCCES! Rata oficială a fost preluată dinamic: {live_rate}%")
            return live_rate
        else:
            print(" [BNR] Indicatorul numeric nu a putut fi extras din structura statică.")

    except Exception as e:
        print(f" [BNR] Conexiunea externă a eșuat sau a expirat ({e}).")

    # Valoarea oficială la zi stabilită de consiliul BNR
    valoare_oficiala = 6.50
    print(f" [BNR] Se aplică valoarea de referință oficială: {valoare_oficiala}%")
    return valoare_oficiala


def calculate_dynamic_interest_rate(current_user, dti):
    base_rate = get_live_bnr_rate()

    if int(current_user.get('istoric_neplata', 0)) == 1: base_rate += 5.0
    vechime = int(current_user.get('vechime', 0))
    if vechime < 2:
        base_rate += 2.5
    elif vechime > 10:
        base_rate -= 1.0
    if int(current_user.get('educatie', 0)) >= 3: base_rate -= 1.5
    if dti > 0.35: base_rate += 3.0
    return round(base_rate, 2)


def generate_repayment_schedule(suma, dobanda_anuala, luni):
    dobanda_lunara = (dobanda_anuala / 100) / 12
    rata_lunar = suma * (dobanda_lunara * (1 + dobanda_lunara) ** luni) / ((1 + dobanda_lunara) ** luni - 1)
    schedule = []
    sold_ramas = suma
    data_start = datetime.now()
    for l in range(1, luni + 1):
        dobanda_luna = sold_ramas * dobanda_lunara
        principal_luna = rata_lunar - dobanda_luna
        sold_ramas -= principal_luna
        schedule.append({
            "luna": l,
            "data_plata": (data_start + timedelta(days=30 * l)).strftime("%d-%m-%Y"),
            "rata_totala": round(rata_lunar, 2),
            "principal": round(principal_luna, 2),
            "dobanda": round(dobanda_luna, 2),
            "sold_ramas": max(0, round(sold_ramas, 2))
        })
    return schedule


def safe_int(val, default=0):
    if val is None or str(val).strip().lower() == 'none' or str(val).strip() == '':
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# --- RUTE NAVIGARE ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/auth')
def auth_page(): return render_template('auth.html')

@app.route('/transactions')
def transactions_page(): return render_template('transactions.html')

@app.route('/simulator')
def simulator_page(): return render_template('simulator.html')

@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')

@app.route('/profile')
def profile_page(): return render_template('profile.html')


# --- RUTE API PROFIL & AUTH ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if db.users.find_one({"email": data['email']}):
        return jsonify({"error": "Email deja înregistrat"}), 400
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user_doc = {
        "username": data['username'], "email": data['email'], "password": hashed_password,
        "varsta": int(data.get('varsta', 30)), "gen": int(data.get('gen', 0)),
        "educatie": int(data.get('educatie', 2)), "vechime": int(data.get('vechime', 5)),
        "tip_locuinta": int(data.get('tip_locuinta', 1)), "ani_istoric_credit": int(data.get('ani_istoric_credit', 3)),
        "istoric_neplata": int(data.get('istoric_neplata', 0)), "data_creare": datetime.now(timezone.utc)
    }
    db.users.insert_one(user_doc)
    return jsonify({"message": "Utilizator creat!"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = db.users.find_one({"email": data.get('email')})
    if user and bcrypt.check_password_hash(user['password'], data.get('password')):
        token = jwt.encode({'user_id': str(user['_id']), 'exp': datetime.now(timezone.utc) + timedelta(hours=24)},
                           app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({"token": token, "message": "Login reușit!"}), 200
    return jsonify({"error": "Date invalide"}), 401


@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    data = request.get_json()
    token_frontend = data.get('id_token')

    if not token_frontend:
        return jsonify({"error": "ID Token lipsă"}), 400

    try:
        # Validare securizată direct prin serverele asincrone Google
        id_info = id_token.verify_oauth2_token(token_frontend, google_requests.Request(), GOOGLE_CLIENT_ID)

        email_utilizator = id_info.get('email')
        nume_utilizator = id_info.get('name')

        if not email_utilizator:
            return jsonify({"error": "Imposibil de extras adresa email din structura Google"}), 400

        # Verificăm dacă profilul există deja în MongoDB
        user = db.users.find_one({"email": email_utilizator})

        # Dacă utilizatorul se conectează pentru prima dată, îi construim profilul automat pe loc (Înregistrare Automată)
        if not user:
            user_doc = {
                "username": nume_utilizator,
                "email": email_utilizator,
                "password": "autentificare_google_oauth2",
                "varsta": 30,
                "gen": 0,
                "educatie": 2,
                "vechime": 5,
                "tip_locuinta": 1,
                "ani_istoric_credit": 3,
                "istoric_neplata": 0,
                "data_creare": datetime.now(timezone.utc)
            }
            result = db.users.insert_one(user_doc)
            user = db.users.find_one({"_id": result.inserted_id})
            print(f" [OAuth] Cont nou creat automat pentru: {email_utilizator}")
        else:
            print(f"🔑 [OAuth] Autentificare reușită pentru contul: {email_utilizator}")

        # Generăm token-ul tău JWT nativ pentru a-i acorda acces în paginile protejate ale aplicației
        token_local = jwt.encode(
            {'user_id': str(user['_id']), 'exp': datetime.now(timezone.utc) + timedelta(hours=24)},
            app.config['SECRET_KEY'],
            algorithm="HS256"
        )

        return jsonify({"token": token_local, "message": "Autentificare Google realizată cu succes!"}), 200

    except ValueError:
        return jsonify({"error": "Token-ul primit de la Google este invalid sau expirat"}), 401
    except Exception as e:
        print(f" EROARE OAUTH CRITICĂ: {str(e)}")
        return jsonify({"error": f"Eroare server: {str(e)}"}), 500


@app.route('/get-profile', methods=['GET'])
@token_required
def get_profile(current_user):
    user_data = {
        "username": current_user.get('username'), "email": current_user.get('email'),
        "varsta": current_user.get('varsta', 30), "gen": current_user.get('gen', 0),
        "educatie": current_user.get('educatie', 2), "vechime": current_user.get('vechime', 5),
        "tip_locuinta": current_user.get('tip_locuinta', 1),
        "ani_istoric_credit": current_user.get('ani_istoric_credit', 3),
        "istoric_neplata": current_user.get('istoric_neplata', 0)
    }
    return jsonify(user_data), 200


@app.route('/update-profile', methods=['POST'])
@token_required
def update_profile(current_user):
    data = request.get_json()
    update_data = {k: (int(v) if k != 'username' else str(v)) for k, v in data.items() if
                   k in ["varsta", "gen", "educatie", "vechime", "tip_locuinta", "ani_istoric_credit",
                         "istoric_neplata", "username"]}
    db.users.update_one({"_id": current_user['_id']}, {"$set": update_data})
    return jsonify({"message": "Profil actualizat!"}), 200


# --- RUTE TRANSFORMARE EXTRAS IN CSV CLEAN
@app.route('/converter')
def convert_extras_page():
    return render_template('converter.html')

@app.route('/api/convert-ing-csv', methods=['POST'])
@token_required
def convert_ing_csv(current_user):
    if 'file' not in request.files: return jsonify({"error": "Fără fișier"}), 400

    file = request.files['file']
    try:
        content = file.read().decode('utf-8').replace('\0', '')
        lines = content.splitlines()
        cleaned_transactions = []
        current_tx = None
        luni_ro = ["ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie"]

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith('"') and line.endswith('"'): line = line[1:-1]
            line = line.replace('""', '"')
            f_line = io.StringIO(line)
            parts = next(csv.reader(f_line))
            if not parts: continue

            first_cell = str(parts[0]).lower()
            is_date = any(l in first_cell for l in luni_ro) and any(c.isdigit() for c in first_cell)

            if is_date:
                if current_tx: cleaned_transactions.append(current_tx)
                def get_val(idx):
                    if len(parts) > idx and parts[idx]:
                        v = parts[idx].replace('.', '').replace(',', '.')
                        try: return float(v)
                        except: return 0.0
                    return 0.0

                val_debit = get_val(5)
                val_credit = get_val(6)

                current_tx = {
                    "Data": parts[0],
                    "Suma": val_credit if val_credit != 0 else -abs(val_debit),
                    "Detalii": parts[3] if len(parts) > 3 else ""
                }
            elif current_tx and len(parts) > 3 and parts[3]:
                txt = parts[3].strip()
                if txt and txt not in ["Detalii tranzactie", "Data"]:
                    current_tx["Detalii"] += " " + txt

        if current_tx: cleaned_transactions.append(current_tx)
        df = pd.DataFrame(cleaned_transactions)

        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')

        return jsonify({
            "csv": output.getvalue(),
            "filename": "extras_curatat.csv"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/import-standard-csv', methods=['POST'], strict_slashes=False)
@token_required
def import_standard_csv(current_user):
    if 'file' not in request.files:
        return jsonify({"error": "Fără fișier"}), 400

    file = request.files['file']
    try:
        content = file.read().decode('utf-8')
        f = io.StringIO(content)
        reader = csv.reader(f)
        rows = list(reader)

        if not rows or len(rows) < 2:
            return jsonify({"error": "Fișierul este gol sau are doar antetul."}), 400

        tranzactii_de_procesat = rows[1:]
        count = 0

        for row in tranzactii_de_procesat:
            if not row or len(row) < 3: continue

            data_str = row[0]  # Coloana 0: Data
            try:
                suma_val = float(row[1])  # Coloana 1: Suma
            except:
                continue
            detalii_brute = row[2]  # Coloana 2: Detalii

            fingerprint = f"{current_user['_id']}_{data_str}_{suma_val}_{detalii_brute}"
            tx_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

            if db.transactions.find_one({"user_id": current_user['_id'], "tx_hash": tx_hash}):
                continue

            info_securizata = anonymize_details(detalii_brute)

            tip = "VENIT" if suma_val > 0 else "CHELTUIALĂ"
            ai_res = classify_with_ai(info_securizata, tip)

            if " | " in ai_res:
                cat, clean_det = ai_res.split(" | ", 1)
            else:
                cat, clean_det = "Altele", info_securizata

            db.transactions.insert_one({
                "user_id": current_user['_id'],
                "tx_hash": tx_hash,
                "data": data_str,
                "suma": suma_val,
                "detalii": clean_det.strip(),
                "categorie": cat.strip(),
                "tip": tip,
                "data_import": datetime.now(timezone.utc)
            })
            count += 1

        db['forecasts'].delete_many({"user_id": current_user['_id']})

        return jsonify({
            "status": "success",
            "message": f"Import finalizat cu succes! {count} tranzacții noi au fost clasificate și adăugate."
        }), 200

    except Exception as e:
        print(f" Eroare Import Standard: {str(e)}")
        return jsonify({"error": str(e)}), 500



# --- RUTE TRANZACȚII & IMPORT ---
@app.route('/preview-csv', methods=['POST'])
@token_required
def preview_csv(current_user):
    if 'file' not in request.files: return jsonify({"error": "Fără fișier"}), 400
    file = request.files['file']
    try:
        content = file.read().decode('utf-8').replace('\0', '')
        lines = content.splitlines()

        final_rows = []
        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]

            line = line.replace('""', '"')

            f_line = io.StringIO(line)
            reader = csv.reader(f_line, delimiter=',')
            try:
                row = next(reader)
                if any(row):
                    final_rows.append(row)
            except:
                continue

        if not final_rows:
            return jsonify({"error": "Fișierul nu a putut fi parsat corect"}), 400

        max_cols = max([len(row) for row in final_rows])
        cols = [f"Coloana {i}" for i in range(max_cols)]

        return jsonify({"columns": cols, "preview_rows": final_rows}), 200
    except Exception as e:
        print(f" Eroare Preview: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/add-manual-transaction', methods=['POST'], strict_slashes=False)
@token_required
def add_manual_transaction(current_user):
    data = request.get_json()
    if not data:
        return jsonify({"error": "Nu s-au primit date"}), 400

    try:
        data_tranzactie = data.get('data') or datetime.now().strftime("%d %B %Y")
        detalii = data.get('detalii') or 'Tranzacție manuală'
        categorie = data.get('categorie') or 'Altele'

        suma_raw = str(data.get('suma', '0')).strip().replace(',', '.')
        if suma_raw in ["", "None", "null"]:
            suma_raw = "0"

        try:
            valoare = float(suma_raw)
        except ValueError:
            return jsonify({"error": "Suma introdusă nu este un număr valid"}), 400

        seed = f"{current_user['_id']}_{data_tranzactie}_{valoare}_{detalii}_{time.time()}"
        tx_hash = hashlib.sha256(seed.encode()).hexdigest()

        db.transactions.insert_one({
            "user_id": current_user['_id'],
            "tx_hash": tx_hash,
            "data": data_tranzactie,
            "suma": valoare,
            "detalii": detalii,
            "categorie": categorie,
            "tip": "VENIT" if valoare > 0 else "CHELTUIALĂ",
            "data_import": datetime.now(timezone.utc)
        })

        db['forecasts'].delete_many({"user_id": current_user['_id']})
        return jsonify({"message": "OK"}), 201

    except Exception as e:
        print(f" EROARE ADD-MANUAL: {str(e)}")
        return jsonify({"error": f"Eroare internă: {str(e)}"}), 500


@app.route('/get-transactions', methods=['GET'])
@token_required
def get_all_transactions(current_user):
    filter_month = request.args.get('month')
    filter_year = request.args.get('year')

    transactions = list(db.transactions.find({"user_id": current_user['_id']}).sort("data_import", -1))

    filtered_list = []
    for t in transactions:
        date_lower = str(t.get('data', '')).lower()
        match = True

        if filter_month and filter_month.lower() not in date_lower:
            match = False
        if filter_year and filter_year not in date_lower:
            match = False

        if match:
            filtered_list.append({
                "id": str(t['_id']),
                "data": t['data'],
                "suma": t['suma'],
                "detalii": t.get('detalii'),
                "categorie": t.get('categorie')
            })

    return jsonify(filtered_list), 200


@app.route('/delete-transaction/<t_id>', methods=['DELETE'])
@token_required
def delete_transaction(current_user, t_id):
    db.transactions.delete_one({"_id": ObjectId(t_id), "user_id": current_user['_id']})
    db['forecasts'].delete_many({"user_id": current_user['_id']})

    return jsonify({"message": "Sters"}), 200


@app.route('/dashboard-stats', methods=['GET'])
@token_required
def get_dashboard_stats(current_user):
    filter_month = request.args.get('month')
    filter_year = request.args.get('year')

    all_trans = list(db.transactions.find({"user_id": current_user['_id']}))
    if not all_trans:
        return jsonify({
            "total_venit": 0,
            "total_cheltuieli": 0,
            "rata_economisire": 0,
            "top_comercianti": [],
            "pie_chart": {"labels": [], "values": []},
            "selected_month": "",
            "selected_year": ""
        }), 200

    df = pd.DataFrame(all_trans)

    # Mapare pentru sortare luni
    luni_ordine = {
        "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5, "iunie": 6,
        "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10, "noiembrie": 11, "decembrie": 12
    }

    # Extragem lista de ani și luni disponibile pentru a determina "ultima încărcare"
    df['data_lower'] = df['data'].str.lower()

    # Dacă nu avem filtre, determinăm automat ultima lună/an
    if not filter_year or not filter_month:
        # Găsim ultimul an (numeric)
        years_found = []
        for d in df['data_lower']:
            parts = d.split()
            if len(parts) >= 3:
                try:
                    years_found.append(int(parts[2]))
                except:
                    pass

        if years_found:
            last_year = str(max(years_found))
            filter_year = last_year if not filter_year else filter_year

            # Găsim ultima lună din acel an
            months_in_year = []
            for d in df[df['data_lower'].str.contains(filter_year)]['data_lower']:
                for m in luni_ordine.keys():
                    if m in d:
                        months_in_year.append(m)

            if months_in_year:
                # Sortăm lunile după ordinea calendaristică
                filter_month = max(set(months_in_year), key=lambda m: luni_ordine[m])

    # Aplicăm filtrarea finală
    def match_filter(row_date):
        date_lower = str(row_date).lower()
        return (filter_month.lower() in date_lower) and (filter_year in date_lower)

    df_filtered = df[df['data'].apply(match_filter)]

    if df_filtered.empty:
        return jsonify({
            "total_venit": 0,
            "total_cheltuieli": 0,
            "rata_economisire": 0,
            "top_comercianti": [],
            "pie_chart": {"labels": [], "values": []},
            "selected_month": filter_month,
            "selected_year": filter_year
        }), 200

    total_venit = df_filtered[df_filtered['suma'] > 0]['suma'].sum()
    total_cheltuieli = df_filtered[df_filtered['suma'] < 0]['suma'].abs().sum()

    # --- CALCULARE STATISTICĂ RATĂ ECONOMISIRE ---
    rata_economisire = 0.0
    if total_venit > 0:
        balanta = total_venit - total_cheltuieli
        rata_economisire = (balanta / total_venit) * 100

    cheltuieli_df = df_filtered[df_filtered['suma'] < 0].copy()
    top_cats = cheltuieli_df.groupby('categorie')['suma'].sum().abs().to_dict() if not cheltuieli_df.empty else {}

    # --- CALCULARE TOP 3 COMERCIANȚI (EXCLUZÂND TRANSFERURILE P2P / REVOLUT) ---
    top_comercianti_list = []
    if not cheltuieli_df.empty:
        # Filtru critic: Eliminăm tranzacțiile din categoria 'TRANSFERURI' sau care conțin cuvinte de rebalansare în detalii
        filtru_excludere = (cheltuieli_df['categorie'].str.lower().str.contains('transfer', na=False)) | \
                           (cheltuieli_df['detalii'].str.lower().str.contains(
                               'revolut p2p|schimb valutar|cont economii', na=False))

        cheltuieli_comerciale = cheltuieli_df[~filtru_excludere]

        if not cheltuieli_comerciale.empty:
            comercianti_grouped = cheltuieli_comerciale.groupby('detalii')['suma'].sum().abs().nlargest(3)
            for nume, suma_totala in comercianti_grouped.items():
                top_comercianti_list.append({
                    "nume": str(nume).strip(),
                    "suma": round(float(suma_totala), 2)
                })

    return jsonify({
        "pie_chart": {
            "labels": list(top_cats.keys()),
            "values": [round(v, 2) for v in top_cats.values()]
        },
        "total_venit": round(float(total_venit), 2),
        "total_cheltuieli": round(float(total_cheltuieli), 2),
        "rata_economisire": round(float(rata_economisire), 2),
        "top_comercianti": top_comercianti_list,
        "selected_month": filter_month,
        "selected_year": filter_year
    }), 200


@app.route('/api/get-estimated-income', methods=['GET'])
@token_required
def get_estimated_income(current_user):
    try:
        trans = list(db['transactions'].find({"user_id": current_user['_id'], "categorie": "Salariu"}))
        df = pd.DataFrame(trans)
        venit_ron = (df['suma'].sum() / len(df)) * 12 if not df.empty else 0
        return jsonify({"venit_anual_ron": round(venit_ron, 2)}), 200
    except Exception as e:
        print(f"Eroare calcul venit: {str(e)}")
        return jsonify({"venit_anual_ron": 0, "error": str(e)}), 500


@app.route('/predict-credit', methods=['POST'])
@token_required
def predict_credit(current_user):
    if not rf_model: return jsonify({"error": "Model ML indisponibil"}), 500
    data = request.get_json()
    try:
        suma_ron = float(data.get('suma_credit', 0))
        ani = int(data.get('perioada_ani', 5))
        v_manual = data.get('venit_anual_manual')

        if v_manual is not None and str(v_manual).strip() != "" and str(v_manual).strip() != "null":
            venit_ron = float(str(v_manual).replace(',', '.'))
        else:
            trans = list(db['transactions'].find({"user_id": current_user['_id'], "categorie": "Salariu"}))
            if not trans:
                return jsonify({
                    "venit_anual_ron": 0,
                    "error": "Introdu venitul manual pentru simulare."}), 200

            df = pd.DataFrame(trans)
            venit_ron = (df['suma'].sum() / len(df)) * 12 if not df.empty else 0

        v_model = venit_ron / 5.0
        s_model = suma_ron / 5.0

        if v_model <= 0:
            return jsonify({"error": "Venitul trebuie să fie mai mare de 0 pentru analiză."}), 400

        dti = s_model / v_model
        dobanda_real = calculate_dynamic_interest_rate(current_user, dti)

        input_features = [
            safe_int(current_user.get('varsta'), 30),
            safe_int(current_user.get('gen'), 0),
            safe_int(current_user.get('educatie'), 2),
            float(v_model),
            safe_int(current_user.get('vechime'), 5),
            safe_int(current_user.get('tip_locuinta'), 1),
            float(s_model),
            float(dobanda_real),
            round(dti, 2),
            safe_int(current_user.get('ani_istoric_credit'), 3),
            safe_int(current_user.get('istoric_neplata'), 0) ]

        #prob = rf_model.predict_proba([input_features])[0][1]
        features_df = pd.DataFrame([input_features], columns=rf_model.feature_names_in_)
        prob = rf_model.predict_proba(features_df)[0][1]
        status = "APROBAT" if prob >= 0.5 else "RESPINS"

        ai_advice = "Analiza AI nu a putut fi generată."
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            prompt_sfat = f"Consilier bancar: Explică scurt de ce creditul de {suma_ron} RON este {status}. Venit anual: {venit_ron}, Grad îndatorare: {round(dti * 100, 2)}%."
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt_sfat}]
            )
            ai_advice = completion.choices[0].message.content
        except Exception as e:
            print(f" Eroare Groq: {e}")

        db['credit_simulations'].insert_one({
            "user_id": current_user['_id'],
            "suma_solicitata": suma_ron,
            "perioada_ani": ani,
            "scor_probabilitate": round(prob, 4),
            "decizie": status,
            "recomandare_ai": ai_advice,
            "venit_calculat": round(venit_ron, 2),
            "dti_calculat": round(dti, 4),
            "timestamp": datetime.now(timezone.utc)
        })

        return jsonify({
            "status_credit": status,
            "probabilitate": f"{round(prob * 100, 2)}%",
            "explicatie_ai": ai_advice,
            "scadentar": generate_repayment_schedule(suma_ron, dobanda_real, ani * 12)[:12],
            "venit_anual_ron": round(venit_ron, 2),
            "dobanda_aplicata": f"{dobanda_real}%"
        }), 200

    except Exception as e:
        print(f" EROARE CRITICĂ PREDICT: {str(e)}")
        return jsonify({"error": f"Eroare server: {str(e)}"}), 500


@app.route('/api/get-forecast-data', methods=['GET'])
@token_required
def get_forecast_detailed(current_user):
    all_trans = list(db.transactions.find({"user_id": current_user['_id']}))
    if not all_trans:
        return jsonify({"error": "Nu există tranzacții pentru analiză."}), 400

    df = pd.DataFrame(all_trans)
    df['suma'] = df['suma'].abs()
    df_exp = df[df['tip'] == 'CHELTUIALĂ'].copy()

    luni_map = {
        "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5, "iunie": 6,
        "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10, "noiembrie": 11, "decembrie": 12
    }

    def parse_to_date(d_str):
        p = d_str.lower().split()
        return datetime(int(p[2]), luni_map[p[1]], 1)

    df_exp['dt_obj'] = df_exp['data'].apply(parse_to_date)
    monthly = df_exp.groupby('dt_obj')['suma'].sum().reset_index().sort_values('dt_obj')

    if len(monthly) < 3:
        return jsonify({"error": "Sunt necesare minim 3 luni de date pentru un trend."}), 400

    y_raw = monthly['suma'].values
    q3 = np.percentile(y_raw, 75)
    iqr = q3 - np.percentile(y_raw, 25)
    upper_limit = q3 + 2.0 * iqr
    y_clean_for_trend = np.clip(y_raw, None, upper_limit)

    x = np.arange(len(monthly))
    weights = np.linspace(1, 2, len(x))
    slope, intercept = np.polyfit(x, y_clean_for_trend, 1, w=weights)
    y_trend = slope * x + intercept

    monthly['month_num'] = monthly['dt_obj'].dt.month
    monthly['seasonal_ratio'] = y_raw / y_trend
    seasonal_map = monthly.groupby('month_num')['seasonal_ratio'].mean().to_dict()

    correlation_matrix = np.corrcoef(x, y_raw)
    r_squared = correlation_matrix[0, 1] ** 2

    last_date = monthly['dt_obj'].max()
    forecast_values = []
    forecast_labels = []

    luni_ro_list = ["Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie", "Iulie", "August", "Septembrie",
                    "Octombrie", "Noiembrie", "Decembrie"]

    for i in range(1, 4):
        next_dt = last_date + pd.DateOffset(months=i)
        m_num = next_dt.month

        future_x = len(monthly) + i - 1
        base_trend_val = slope * future_x + intercept

        factor = seasonal_map.get(m_num, 1.0)
        factor = max(0.6, min(1.8, factor))

        forecast_values.append(base_trend_val * factor)
        forecast_labels.append(f"{luni_ro_list[m_num - 1]} {next_dt.year}")

    history_labels = [f"{luni_ro_list[d.month - 1]} {d.year}" for d in monthly['dt_obj']]

    return jsonify({
        "history": {
            "labels": history_labels,
            "values": y_raw.tolist()
        },
        "forecast": {
            "labels": forecast_labels,
            "values": [round(v, 2) for v in forecast_values],
            "slope": slope,
            "intercept": intercept,
            "next_index_start": len(monthly)
        },
        "metrics": {
            "r_squared": round(r_squared * 100, 2),
            "data_points_count": len(monthly),
            "is_outdated": (datetime.now(timezone.utc).replace(tzinfo=None) - last_date).days > 45
        }
    }), 200


@app.route('/financial-persona', methods=['GET'])
@token_required
def get_financial_persona(current_user):
    if not kmeans_model:
        return jsonify({"error": "Modelul K-Means este indisponibil"}), 500

    try:
        # 1. Preluăm lista exactă de 13 coloane pe care le așteaptă modelul tău K-Means
        expected_features = kmeans_model.feature_names_in_
        print(f"🔍 [K-Means] Coloanele așteptate de modelul tău: {list(expected_features)}")

        # 2. Extragem datele brute ale utilizatorului curent
        all_trans = list(db.transactions.find({"user_id": current_user['_id']}))

        total_venituri = 0.0
        total_cheltuieli = 0.0
        numar_tranzactii = 0
        cat_mancare = 0.0
        cat_shopping = 0.0
        cat_utilitati = 0.0

        if all_trans:
            df = pd.DataFrame(all_trans)
            numar_tranzactii = len(df)
            total_venituri = float(df[df['suma'] > 0]['suma'].sum())
            total_cheltuieli = float(df[df['suma'] < 0]['suma'].abs().sum())

            # Calculăm sumele pe categoriile principale
            cat_mancare = float(df[df['categorie'].str.lower() == 'mâncare']['suma'].abs().sum())
            cat_shopping = float(df[df['categorie'].str.lower() == 'shopping']['suma'].abs().sum())
            cat_utilitati = float(df[df['categorie'].str.lower() == 'utilități']['suma'].abs().sum())

        # 3. Creăm un dicționar cu TOATE denumirile posibile (bilingv - română/engleză)
        user_data_map = {
            "varsta": safe_int(current_user.get('varsta'), 30),
            "gen": safe_int(current_user.get('gen'), 0),
            "educatie": safe_int(current_user.get('educatie'), 2),
            "vechime": safe_int(current_user.get('vechime'), 5),
            "tip_locuinta": safe_int(current_user.get('tip_locuinta'), 1),
            "ani_istoric_credit": safe_int(current_user.get('ani_istoric_credit'), 3),
            "istoric_neplata": safe_int(current_user.get('istoric_neplata'), 0),
            "total_venituri": total_venituri,
            "total_cheltuieli": total_cheltuieli,
            "numar_tranzactii": numar_tranzactii,
            "mancare": cat_mancare,
            "shopping": cat_shopping,
            "utilitati": cat_utilitati,

            "age": safe_int(current_user.get('varsta'), 30),
            "gender": safe_int(current_user.get('gen'), 0),
            "education": safe_int(current_user.get('educatie'), 2),
            "experience": safe_int(current_user.get('vechime'), 5),
            "home_type": safe_int(current_user.get('tip_locuinta'), 1),
            "credit_history_years": safe_int(current_user.get('ani_istoric_credit'), 3),
            "missed_payments": safe_int(current_user.get('istoric_neplata'), 0),
            "total_income": total_venituri,
            "total_expenses": total_cheltuieli,
            "transaction_count": numar_tranzactii,
            "food": cat_mancare,
            "utilities": cat_utilitati
        }

        # 4. Construim vectorul final mapat și ordonat EXACT după cerințele K-Means
        aligned_features = [user_data_map.get(col, 0.0) for col in expected_features]

        # Transformăm lista într-un DataFrame cu numele corecte
        features_df = pd.DataFrame([aligned_features], columns=expected_features)

        # 5. Rulăm predicția
        cluster_id = int(kmeans_model.predict(features_df)[0])
        print(f" [K-Means] Utilizatorul a fost repartizat corect în Clusterul: {cluster_id}")

        persona_mapping = {
            0: {
                "nume": "Profil Prudent-Conservator",
                "desc": "Utilizatorul manifestă o conduită financiară caracterizată prin aversiune structurală față de risc, prioritizarea fondurilor de rezervă și un control riguros asupra fluxurilor de ieșire non-esențiale."
            },
            1: {
                "nume": "Profil Echilibrat-Strategic",
                "desc": "Comportamentul relevă o gestionare optimă și planificată a resurselor, menținând o corelație stabilă între veniturile realizate și cheltuielile operaționale, cu o capacitate ridicată de optimizare a bugetului."
            },
            2: {
                "nume": "Profil Expansiv (Orientat spre Consum)",
                "desc": "Analiza indică o frecvență ridicată a tranzacțiilor în sectoare de consum discreționar și o viteză mare de circulație a banilor, evidențiind o preferință pentru utilitatea imediată a lichidităților."
            },
            3: {
                "nume": "Profil Expus (Restricționat Bugetar)",
                "desc": "Acest segment prezintă o elasticitate redusă a bugetului individual, unde cheltuielile totale converg spre plafonul veniturilor disposable, generând o vulnerabilitate ridicată în fața unor eventuale șocuri de lichiditate."
            }
        }

        profil_determinat = persona_mapping.get(cluster_id, persona_mapping[0])

        raspuns = {
            "nume": profil_determinat["nume"],
            "desc": profil_determinat["desc"],
            "cluster_id": cluster_id
        }

        return jsonify(raspuns), 200

    except Exception as e:
        print(f" EROARE K-MEANS PERSONA: {str(e)}")
        return jsonify({"error": f"Eroare server la determinarea profilului: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)