from callshield.cdr.features import FEATURE_NAMES, Event, RollingHistory, features_from_history


def test_empty_history_has_all_features():
    f = features_from_history([], now=1000.0)
    assert list(f) == FEATURE_NAMES


def test_no_lookahead():
    hist = [Event(100, "out", "a", 10), Event(5000, "out", "b", 10)]
    assert features_from_history(hist, now=1000.0)["out_calls_24h"] == 1  # future call ignored


def test_unfinished_call_ignored():
    # a call that started before `now` but is still running has no known duration yet
    assert features_from_history([Event(990, "out", "a", 60)], now=1000.0)["out_calls_24h"] == 0


def test_simbox_pattern():
    hist = [Event(i * 60, "out", f"n{i}", 20) for i in range(30)]
    f = features_from_history(hist, now=30 * 60 + 30)
    assert f["distinct_ratio_24h"] == 1.0
    assert f["short_call_share_24h"] == 1.0
    assert f["data_sessions_24h"] == 0 and f["sms_24h"] == 0


def test_event_kinds_counted():
    hist = [Event(1, "data"), Event(2, "sms"), Event(3, "in", "x", 5), Event(4, "out", "y", 5)]
    f = features_from_history(hist, now=100)
    assert (f["data_sessions_24h"], f["sms_24h"], f["in_calls_24h"], f["out_calls_24h"]) == (1, 1, 1, 1)


def test_rolling_history_window():
    h = RollingHistory()
    h.add("x", Event(0, "out", "a", 1))
    h.add("x", Event(90000, "out", "b", 1))
    assert len(h.get("x", 90001)) == 1
