"""Integration test: original AI pipeline against a real, isolated MongoDB test database.

The fixtures are controlled test records, NOT user data or model quality evidence.
Run with MONGODB_URI pointing to a disposable MongoDB instance.
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def ai_modules(monkeypatch):
    """Bind the real pipeline to a separate, randomly named test database."""
    import motor.motor_asyncio
    from recommendation import reco

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MongoDB integration requires MONGODB_URI")

    client = motor.motor_asyncio.AsyncIOMotorClient(
        uri, serverSelectionTimeoutMS=5000
    )
    database_name = "donassos_ai_test_" + uuid.uuid4().hex
    db = client[database_name]

    monkeypatch.setattr(reco, "campaigns_collection", db["campaigns"])
    monkeypatch.setattr(reco, "events_collection", db["events"])
    reco.CONTENT.item_matrix = None
    reco.CONTENT.lang_cached = None
    return reco, client, db


async def test_original_hybrid_pipeline_reads_mongodb_and_ranks(ai_modules):
    from bson import ObjectId

    reco, client, db = ai_modules
    try:
        await client.admin.command("ping")
        campaign_ids = [ObjectId(), ObjectId(), ObjectId()]
        await db.campaigns.insert_many([
            {
                "_id": campaign_ids[0], "approved": True, "lang": "fr",
                "title": "Aide scolaire",
                "description": "Fournitures scolaires pour les enfants",
                "tags": ["education", "ecole"],
            },
            {
                "_id": campaign_ids[1], "approved": True, "lang": "fr",
                "title": "Soutien alimentaire",
                "description": "Repas pour des familles",
                "tags": ["alimentation", "solidarite"],
            },
            {
                "_id": campaign_ids[2], "approved": False, "lang": "fr",
                "title": "Campagne non approuvee",
                "description": "Cette campagne ne doit pas apparaitre",
                "tags": ["exclure"],
            },
        ])
        await db.events.insert_many([
            {
                "user_id": "integration-user",
                "item_id": str(campaign_ids[0]),
                "type": "favorite",
            },
            {
                "user_id": "neighbor-user",
                "item_id": str(campaign_ids[0]),
                "type": "don",
            },
            {
                "user_id": "neighbor-user",
                "item_id": str(campaign_ids[1]),
                "type": "don",
            },
        ])

        await reco.CONTENT.build(reco.campaigns_collection)
        await reco.CF.build(reco.events_collection)

        # Execute the actual production hybrid ranking function, not a replica.
        result = await reco.recommendations(
            user_id="integration-user", k=10, lang="fr",
            exclude_seen=True,
        )
        assert isinstance(result["items"], list)
        assert [item["id"] for item in result["items"]] == [
            str(campaign_ids[1])
        ]
        assert result["items"][0]["title"] == "Soutien alimentaire"
        assert "_id" not in result["items"][0]

        # End-to-end event writing, reindex and reranking
        from recommendation.schemas import EventIn

        recorded = await reco.log_event(EventIn(
            user_id="integration-user",
            item_id=str(campaign_ids[1]),
            type="favorite",
        ))
        assert recorded["ok"] is True
        assert await db.events.count_documents({
            "user_id": "integration-user", "item_id": str(campaign_ids[1])
        }) == 1

        await reco.rebuild_index(lang="fr")
        result = await reco.recommendations(
            user_id="integration-user", k=10, lang="fr",
            exclude_seen=True,
        )
        assert result["items"] == []
    finally:
        await client.drop_database(db.name)
        client.close()
