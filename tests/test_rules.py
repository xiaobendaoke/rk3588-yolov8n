from app.rules.engine import RuleConfig, RuleEngine
from app.types import Detection


def _d(ts, cls, conf, box):
    """创建带有预定义类别名称的 Detection 对象。

    Args:
        ts: 毫秒级时间戳。
        cls: 类别索引（0=cup, 1=phone, 2=keyboard, 3=scissors）。
        conf: 置信度分数。
        box: (x1, y1, x2, y2) 格式的边界框。

    Returns:
        Detection 实例。
    """
    names = ["cup", "phone", "keyboard", "scissors"]
    return Detection(ts_ms=ts, class_id=cls, class_name=names[cls], conf=conf, bbox_xyxy=box)


def _engine(hold=1, cooldown=0):
    """创建带有测试配置的 RuleEngine。

    Args:
        hold: 发出事件前需要持续的帧数。
        cooldown: 同类型事件之间的冷却时间（秒）。

    Returns:
        配置好的 RuleEngine 实例。
    """
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
    """测试杯子靠近手机时触发液体靠近电子设备规则。"""
    e = _engine()
    ds = [
        _d(1, 0, 0.9, (100, 100, 170, 170)),
        _d(1, 1, 0.8, (180, 120, 240, 180)),
    ]
    out = e.evaluate(ds)
    assert len(out) == 1
    assert out[0].risk_type == "liquid_near_electronics"


def test_sharp_tool_roi_trigger():
    """测试剪刀在危险区域内时触发尖锐工具误放规则。"""
    e = _engine()
    ds = [_d(1, 3, 0.9, (500, 400, 580, 500))]
    out = e.evaluate(ds)
    assert any(x.risk_type == "sharp_tool_misplaced" for x in out)


def test_dense_trigger_count_threshold():
    """测试4个或更多目标通过数量阈值触发桌面拥挤规则。"""
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
    """测试高 IoU 总和即使目标数量较少也触发桌面拥挤规则。"""
    e = _engine()
    ds = [
        _d(1, 0, 0.9, (100, 100, 220, 220)),
        _d(1, 1, 0.8, (110, 110, 230, 230)),
        _d(1, 2, 0.7, (120, 120, 240, 240)),
    ]
    out = e.evaluate(ds)
    assert any(x.risk_type == "desk_overcrowded" for x in out)


def test_hold_and_cooldown():
    """测试持续帧计数和冷却机制抑制重复事件。"""
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