import os

db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
api_key = os.getenv("OPENAI_TOKEN")
base_url = os.getenv("BASE_URL", "http://localhost:8000")
ssl_verify = os.environ.get("SSL_VERIFY")
s3_region = os.environ.get('S3_REGION')
access_key = os.environ.get('ACCESS_KEY')
secret_key = os.environ.get('SECRET_KEY')
s3_endpoint_url = os.environ.get('S3_ENDPOINT_URL')
signed_url_expiration = os.environ.get('SIGNED_URL_EXPIRATION', 3600)