import os

db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
api_key = os.getenv("OPENAI_TOKEN")
base_url = os.getenv("BASE_URL", "http://localhost:8000")