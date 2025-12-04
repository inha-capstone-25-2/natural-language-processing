import os
from dotenv import load_dotenv

# Load environment variables from backend-secret submodule
# Default to prod environment
env_path = os.path.join(os.path.dirname(__file__), "..", "backend-secret", "prod", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    # Fallback to local .env
    load_dotenv()
    print("Loaded environment variables from local .env")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://rsrs-root:KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo@35.87.92.19:27017/arxiv?authSource=admin")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "arxiv")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "papers")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Model Configuration
BIGBIRD_MODEL_PATH = os.getenv("BIGBIRD_MODEL_PATH", "models/bigbird")
M2M100_MODEL_PATH = os.getenv("M2M100_MODEL_PATH", "models/m2m100_cs_finetuned")
