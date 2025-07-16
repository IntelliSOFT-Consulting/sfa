import os
from dotenv import load_dotenv

# loading environment variables from a .env file
load_dotenv()

# environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.getenv("langchain_api_key")
OPENAI_API_KEY = os.getenv("openai_api_key")
PINECONE_API_KEY = os.getenv("pinecone_api_key")
HUGGINGFACE_TOKEN = os.getenv("huggingface_access_token")
SECRET_KEY = os.getenv("secret_key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "api", "auth_module", "users.db")