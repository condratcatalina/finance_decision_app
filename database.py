import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.client = MongoClient(self.uri)
        self.db = self.client['FinanceAppDB']

        self.users = self.db['users']
        self.profiles = self.db['user_profiles']
        self.transactions = self.db['transactions']
        self.ml_results = self.db['ml_results']
        self.credit_requests = self.db['credit_requests']

    def __getitem__(self, collection_name):
        return self.db[collection_name]

    def test_connection(self):
        try:
            self.client.admin.command('ping')
            print(" Conexiunea la MongoDB Atlas a fost realizată cu succes!")
        except Exception as e:
            print(f" Eroare la conectare: {e}")

db = Database()