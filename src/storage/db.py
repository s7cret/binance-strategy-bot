import aiosqlite
from pathlib import Path
from src.core.config import settings

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

async def init_db():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.db_path) as db:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            await db.executescript(f.read())
        await db.commit()

async def get_db():
    db = await aiosqlite.connect(settings.db_path)
    db.row_factory = aiosqlite.Row
    return db
