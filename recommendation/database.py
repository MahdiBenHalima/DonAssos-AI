"""MongoDB adapter for the original DonAssos AI recommendation pipeline.

Only the two collections consumed by the AI are exposed; no authentication,
payment, campaign-management or Flutter code is included.
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient

uri = os.environ.get("MONGODB_URI")
if not uri:
    raise RuntimeError("MONGODB_URI is required for the live AI recommendation API")

_client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
_database = _client[os.environ.get("MONGODB_DB_NAME", "donassos")]
campaigns_collection = _database["campaigns"]
events_collection = _database["events"]
