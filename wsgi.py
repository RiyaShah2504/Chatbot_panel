import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before importing app
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

from app import app

if __name__ == "__main__":
    app.run()