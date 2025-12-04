import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Model Configuration
BIGBIRD_MODEL_PATH = os.getenv("BIGBIRD_MODEL_PATH", "models/bigbird")
M2M100_MODEL_PATH = os.getenv("M2M100_MODEL_PATH", "models/m2m100_cs_finetuned")
