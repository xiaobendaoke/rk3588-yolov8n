from app.rules.engine import RuleConfig, RuleEngine
from app.types import Detection


def _d(ts, cls, conf, box):
    names = ["cup", "phone", "keyboard", "scissors"]
    return Detection(ts_ms=ts, class_id=cls, class_name=names[cls], conf=conf, bbox_xyxy=box)


def _engine(hold=1, cooldown=0):
    return RuleEngine(
        RuleConfig(
            risk_distance_px=120,
            risk_hold_frames=hold,
            event_cooldown_sec=cooldown,
            danger_roi=(430, 260, 640, 640),
            dense_count_threshold=4,
            dense_iou_sum_threshold=1.2,
        )
    )


def test_liquid_near_electronics_trigger():
    e = _engine()
    ds = [
        _d(1, 0, 0.9, (100, 100, 170, 170)),
        _d(1, 1, 0.8, (180, 120, 240, 180)),
    ]
    out = e.evaluate(ds)
    assert len(out) == 1
    assert out[0].risk_type == "liquid_near_electronics"


def test_sharp_tool_roi_trigger():
    e = _engine()
    ds = [_d(1, 3, 0.9, (500, 400, 580, 500))]
    out = e.evaluate(ds)
    assert any(x.risk_type == "sharp_tool_misplaced" for x in out)


def test_dense_trigger_count_threshold():
    e = _engine()
    ds = [
        _d(1, 0, 0.9, (10, 10, 90, 90)),
        _d(1, 1, 0.8, (100, 20, 180, 100)),
        _d(1, 2, 0.7, (190, 30, 270, 110)),
        _d(1, 3, 0.6, (280, 40, 360, 120)),
    ]
    out = e.evaluate(ds)
    assert any(x.risk_type == "desk_overcrowded" for x in out)


def test_dense_trigger_iou_sum_without_count_threshold():
    e = _engine()
    ds = [
        _d(1, 0, 0.9, (100, 100, 220, 220)),
        _d(1, 1, 0.8, (110, 110, 230, 230)),
        _d(1, 2, 0.7, (120, 120, 240, 240)),
    ]
    out = e.evaluate(ds)
    assert any(x.risk_type == "desk_overcrowded" for x in out)


def test_hold_and_cooldown():
    e = _engine(hold=2, cooldown=100)
    ds = [
        _d(1, 0, 0.9, (100, 100, 170, 170)),
        _d(1, 1, 0.8, (180, 120, 240, 180)),
    ]
    out1 = e.evaluate(ds)
    out2 = e.evaluate(ds)
    out3 = e.evaluate(ds)
    assert len(out1) == 0
    assert len(out2) == 1
    assert len(out3) == 0
