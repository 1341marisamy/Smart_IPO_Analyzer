import os

from dotenv import load_dotenv

load_dotenv()


def get_mongo_uri() -> str | None:
    """Return the preferred MongoDB URI, with support for the legacy alias."""
    return os.environ.get("MONGO_URI") or os.environ.get("MONGODB_STR")
