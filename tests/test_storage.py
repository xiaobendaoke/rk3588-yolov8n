from app.storage.events import EventStore
from app.types import Detection, RiskEventCandidate


def test_insert_and_query_event(tmp_path):
    db = tmp_path / "events.db"
    store = EventStore(str(db))
    store.init_schema()

    d = Detection(ts_ms=1712000000000, class_id=0, class_name="cup", conf=0.9, bbox_xyxy=(1, 2, 3, 4))
    event = RiskEventCandidate(
        ts_ms=1712000000000,
        risk_type="liquid_near_electronics",
        severity="high",
        confidence=0.8,
        reason="unit test",
        objects=[d],
    )

    event_id = store.insert_event(event, "/tmp/s.jpg")
    assert event_id >= 1

    items = store.list_events(page=1, size=10)
    assert len(items) == 1
    assert items[0]["risk_type"] == "liquid_near_electronics"
