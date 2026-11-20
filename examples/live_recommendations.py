"""Call the REAL DonAssos AI recommendation endpoint backed by MongoDB.

First start the AI API with an authorized MONGODB_URI:
    uvicorn app:app --reload
Then run:
    python -m examples.live_recommendations YOUR_EXISTING_USER_ID

This script creates no fake profiles, events, rankings or campaign records.
"""
import json
import sys
from urllib.parse import urlencode
from urllib.request import urlopen


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m examples.live_recommendations <user_id>"
        )
    query = urlencode({"user_id": sys.argv[1], "k": 10})
    url = "http://127.0.0.1:8000/reco/recommendations?" + query
    with urlopen(url, timeout=30) as response:
        result = json.load(response)
    for rank, item in enumerate(result.get("items", []), start=1):
        print(f"{rank}. {item.get('title', '<untitled>')} (id={item['id']})")
    if not result.get("items"):
        print("No eligible recommendations for this user.")


if __name__ == "__main__":
    main()
