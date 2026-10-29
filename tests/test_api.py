"""API tests. The key one: features computed ONLINE through the API equal the
features computed OFFLINE for training (no train/serve skew)."""
import pytest

from callshield.common import MODELS, PROCESSED

pytestmark = pytest.mark.skipif(not (MODELS / "serving" / "stage1.json").exists(), reason="serving bundle not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from callshield.serve.app import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_score_unknown_caller(client):
    r = client.post("/v1/calls/score", json={"caller": "+000", "callee": "+111", "ts": 1e9}).json()
    assert 0 <= r["fraud_probability"] <= 1 and r["action"] in ("warn", "allow")


def test_online_features_match_offline(client):
    """Replay the first hours of one scenario through the API and compare with the offline table."""
    from callshield.cdr.data import CDR_ROOT, replay_scenario

    folder = CDR_ROOT / "all_naive" / "all_naive_12%_200sims"
    offline = replay_scenario(folder).set_index(["ts", "caller", "callee"])
    import pandas as pd

    cols = ["date", "time", "type", "src", "a", "b", "src_cell", "dst", "c", "d", "dst_cell", "duration", "size"]
    cdr = pd.read_csv(folder / "Op_1_CDRTrace.txt", sep=r"\s+", skiprows=1, names=cols, dtype=str).head(15000)
    from callshield.cdr.data import KIND, _ts

    checked = 0
    for r in cdr.itertuples(index=False):
        if r.type not in KIND:
            continue
        ts, sim = _ts(r.date, r.time), "T" + r.src  # prefix keeps test SIMs apart from other tests
        if KIND[r.type] == "out":
            online = client.post("/v1/calls/score", json={"caller": sim, "callee": "T" + r.dst, "ts": ts}).json()["features"]
            row = offline.loc[(ts, r.src, r.dst)]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            for k, v in online.items():
                assert v == pytest.approx(row[k], rel=1e-6, abs=1e-6), k
            checked += 1
            client.post("/v1/calls/complete", json={"caller": sim, "callee": "T" + r.dst, "ts": ts,
                                                    "duration": float(r.duration), "cell": r.src_cell,
                                                    "callee_cell": r.dst_cell})
        else:
            client.post("/v1/events", json={"sim": sim, "kind": KIND[r.type], "ts": ts, "cell": r.src_cell})
    assert checked > 50
