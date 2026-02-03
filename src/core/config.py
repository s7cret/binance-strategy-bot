from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    db_path: str = os.getenv("DB_PATH", "data/bot.db")

settings = Settings()
