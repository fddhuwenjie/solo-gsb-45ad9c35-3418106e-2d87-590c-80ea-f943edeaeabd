"""排练批注：在当前领取进度登记文字批注并反推乐谱位置、按登记先后查询。

覆盖：
* 未定位/已定位起点下的反推（含“刚定位的同一秒落在目标位置”）；
* 秒数不等于当前进度 -> 409 且不写入；秒数非有限/负、文字为空 -> 422；
* 位置量化到最近 tick（含线性段半 tick 四舍五入）；
* 按登记先后查询；同进度可登记多条；
* 发布/再次定位后既有批注的位置与修订号保持登记时的值（冻结）；
* 登记不推进进度、不发放提示；
* 登记与发布并发：进度、起点、速度图取自同一已提交状态（修订与位置一致）；
* SQLite 持久化（模拟 Compose 重启）后批注仍可查询。
"""

import math
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """每个测试用独立 SQLite，并重新初始化模块级 store。"""
    db_path = tmp_path / "annotations.db"
    monkeypatch.setenv("TEMPO_DB_PATH", str(db_path))

    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))
    yield db_path


from app.main import app  # noqa: E402

client = TestClient(app)

# 120 BPM、4/4、PPQ=480：一拍 0.5s、一小节 2.0s
# 1:1:0 -> 0.0s，1:2:0 -> 0.5s，2:1:0 -> 2.0s，4:1:0 -> 6.0s
SCORE_BODY = {
    "ppq": 480,
    "time_signatures": [],
    "tempo_points": [{"position": "1:1:0", "bpm": 120, "curve": "step"}],
    "anchors": [{"position": "1:1:0", "locked": True}],
}
# 替换表：整体降到 60 BPM（1s 一拍；2.0s 仅走到 1:3:0）
SLOW_POINTS = [{"position": "1:1:0", "bpm": 60, "curve": "step"}]


def create_score(body=SCORE_BODY):
    resp = client.post("/scores", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def create_rehearsal(score_id, cues=None):
    cues = cues if cues is not None else [{"position": "2:1:0", "text": "提示"}]
    resp = client.post(
        "/rehearsals", json={"score_id": score_id, "cues": cues}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def claim(rid, elapsed):
    return client.post(
        f"/rehearsals/{rid}/claims", json={"elapsed_seconds": elapsed}
    )


def locate(rid, elapsed, position):
    return client.post(
        f"/rehearsals/{rid}/locate",
        json={"elapsed_seconds": elapsed, "position": position},
    )


def annotate(rid, elapsed, text):
    return client.post(
        f"/rehearsals/{rid}/annotations",
        json={"elapsed_seconds": elapsed, "text": text},
    )


def annotations(rid):
    return client.get(f"/rehearsals/{rid}/annotations")


# ---------------------------------------------------------------------- #
# 基本登记：未定位时从乐谱起点按绝对秒数反推
# ---------------------------------------------------------------------- #
def test_annotation_at_score_start_before_any_claim():
    score = create_score()
    r = create_rehearsal(score["id"])
    resp = annotate(r["id"], 0.0, "开头注意")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert isinstance(data["id"], str) and len(data["id"]) == 32
    assert data["text"] == "开头注意"
    assert data["elapsed_seconds"] == 0.0
    assert data["position"] == "1:1:0"
    assert (data["measure"], data["beat"], data["tick"]) == (1, 1, 0)
    assert data["total_ticks"] == 0
    assert data["revision"] == 0


def test_annotation_maps_current_progress_to_score_position():
    score = create_score()
    r = create_rehearsal(score["id"])
    assert claim(r["id"], 2.0).status_code == 200  # 进度推进到 2.0s

    resp = annotate(r["id"], 2.0, "这里弱了")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    # 120 BPM 下 2.0s = 一小节 -> 2:1:0 / 1920 tick
    assert data["position"] == "2:1:0"
    assert (data["measure"], data["beat"], data["tick"]) == (2, 1, 0)
    assert data["total_ticks"] == 1920
    assert data["revision"] == 0
    assert data["elapsed_seconds"] == 2.0


def test_annotation_does_not_advance_progress_or_deliver_cues():
    score = create_score()
    r = create_rehearsal(
        score["id"], [{"position": "2:1:0", "text": "到副歌"}]
    )
    resp = annotate(r["id"], 0.0, "批注不发提示")
    assert resp.status_code == 201

    got = client.get(f"/rehearsals/{r['id']}").json()
    assert got["last_elapsed_seconds"] == 0.0
    assert got["cues"][0]["status"] == "pending"


# ---------------------------------------------------------------------- #
# 秒数不等于当前已领取进度 -> 409 且不写入
# ---------------------------------------------------------------------- #
def test_annotation_with_stale_elapsed_is_409_and_not_written():
    score = create_score()
    r = create_rehearsal(score["id"])
    claim(r["id"], 2.0)

    resp = annotate(r["id"], 1.5, "落后进度")
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "elapsed_seconds"]
    assert detail["elapsed_seconds"] == 1.5
    assert detail["last_elapsed_seconds"] == 2.0

    # 超前同样 409（必须精确等于）
    ahead = annotate(r["id"], 2.5, "超前进度")
    assert ahead.status_code == 409

    # 不写入：批注列表为空，排练进度/起点不变
    assert annotations(r["id"]).json() == []
    got = client.get(f"/rehearsals/{r['id']}").json()
    assert got["last_elapsed_seconds"] == 2.0
    assert got["origin"]["total_ticks"] == 0

    # 按正确秒数仍可登记
    ok = annotate(r["id"], 2.0, "补上")
    assert ok.status_code == 201
    assert len(annotations(r["id"]).json()) == 1


def test_annotation_rejects_negative_nonfinite_elapsed_and_empty_text():
    score = create_score()
    r = create_rehearsal(score["id"])

    assert annotate(r["id"], -0.1, "负秒数").status_code == 422
    empty = annotate(r["id"], 0.0, "   ")
    assert empty.status_code == 422
    assert empty.json()["detail"][0]["loc"] == ["body", "text"]

    # 两个错误一次性聚合返回
    both = client.post(
        f"/rehearsals/{r['id']}/annotations",
        json={"elapsed_seconds": -1.0, "text": ""},
    )
    assert both.status_code == 422
    locs = [tuple(e["loc"]) for e in both.json()["detail"]]
    assert ("body", "elapsed_seconds") in locs
    assert ("body", "text") in locs

    for raw in ("NaN", "Infinity", "-Infinity"):
        resp = client.post(
            f"/rehearsals/{r['id']}/annotations",
            content=f'{{"elapsed_seconds": {raw}, "text": "x"}}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw

    assert annotations(r["id"]).json() == []


def test_annotation_unknown_rehearsal_is_404():
    resp = annotate("deadbeef", 0.0, "x")
    assert resp.status_code == 404
    assert resp.json()["detail"][0]["loc"] == ["path", "rehearsal_id"]
    got = annotations("deadbeef")
    assert got.status_code == 404
    assert got.json()["detail"][0]["loc"] == ["path", "rehearsal_id"]


# ---------------------------------------------------------------------- #
# 已定位：按当前定位起点反推；刚定位的同一秒落在目标位置
# ---------------------------------------------------------------------- #
def test_annotation_right_after_locate_lands_on_target_position():
    score = create_score()
    r = create_rehearsal(
        score["id"],
        [{"position": f"{m}:1:0", "text": f"c{m}"} for m in (2, 3, 5)],
    )
    rid = r["id"]
    claim(rid, 2.0)
    # 定位到 4:1:0（绝对 6.0s），提交累计秒数 2.0
    loc = locate(rid, 2.0, "4:1:0")
    assert loc.status_code == 200, loc.text

    # 同一秒登记：应精确落在定位目标 4:1:0 / 5760 tick
    resp = annotate(rid, 2.0, "跳过来的位置")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["position"] == "4:1:0"
    assert data["total_ticks"] == 5760
    assert data["elapsed_seconds"] == 2.0
    assert data["revision"] == 0

    # 再领 2.0s（累计 4.0）：起点 4:1:0 + 一小节(2.0s) -> 5:1:0 / 7680
    assert claim(rid, 4.0).status_code == 200
    later = annotate(rid, 4.0, "一小节后").json()
    assert later["position"] == "5:1:0"
    assert later["total_ticks"] == 7680


def test_annotation_quantizes_to_nearest_tick_on_linear_segment():
    """线性段中累计秒数落在两 tick 之间时，量化到最近 tick。"""
    from app.tempo import build_tempo_map

    body = {
        "ppq": 480,
        "time_signatures": [],
        "tempo_points": [
            {"position": "1:1:0", "bpm": 60, "curve": "linear"},
            {"position": "1:2:0", "bpm": 120, "curve": "step"},
        ],
        "anchors": [{"position": "1:1:0", "locked": True}],
    }
    score = create_score(body)
    r = create_rehearsal(score["id"])
    rid = r["id"]

    # 取线性段内 tick 240.5 处的秒数：未量化 tick 距 241 更近（半 tick
    # 四舍五入），登记后应量化为 241。
    tmap = build_tempo_map(
        score["ppq"], [], body["tempo_points"]
    )
    elapsed = tmap.seconds_at_tick(240.5)
    raw = tmap.tick_at_seconds(elapsed)
    assert math.isclose(raw, 240.5, rel_tol=0, abs_tol=1e-9)
    claim(rid, elapsed)

    data = annotate(rid, elapsed, "半 tick").json()
    assert data["total_ticks"] == 241
    assert data["position"] == "1:1:241"
    assert (data["measure"], data["beat"], data["tick"]) == (1, 1, 241)


# ---------------------------------------------------------------------- #
# 按登记先后查询；同进度可登记多条
# ---------------------------------------------------------------------- #
def test_annotations_listed_in_registration_order_including_same_elapsed():
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]

    a0 = annotate(rid, 0.0, "第一条").json()
    a1 = annotate(rid, 0.0, "同进度第二条").json()  # 同秒可重复登记
    claim(rid, 2.0)
    a2 = annotate(rid, 2.0, "第三条").json()

    ids = [a0["id"], a1["id"], a2["id"]]
    assert len(set(ids)) == 3
    listing = annotations(rid).json()
    assert [a["id"] for a in listing] == ids
    assert [a["text"] for a in listing] == ["第一条", "同进度第二条", "第三条"]
    assert [a["position"] for a in listing] == ["1:1:0", "1:1:0", "2:1:0"]
    assert [a["total_ticks"] for a in listing] == [0, 0, 1920]
    assert all(a["revision"] == 0 for a in listing)


def test_annotations_empty_list_for_new_rehearsal():
    score = create_score()
    r = create_rehearsal(score["id"])
    assert annotations(r["id"]).json() == []


# ---------------------------------------------------------------------- #
# 冻结：速度图发布或再次定位后，既有批注的位置与修订号保持登记时的值
# ---------------------------------------------------------------------- #
def test_existing_annotations_frozen_after_publish_and_relocate():
    score = create_score()
    sid = score["id"]
    r = create_rehearsal(sid)
    rid = r["id"]

    # rev0 / 120 BPM：进度 2.0s -> 2:1:0 / 1920
    claim(rid, 2.0)
    first = annotate(rid, 2.0, "放慢前").json()
    assert first["revision"] == 0 and first["total_ticks"] == 1920

    # 发布 60 BPM（rev1）：同一进度 2.0s 现在只走到 1:3:0 / 960
    pub = client.post(
        f"/scores/{sid}/revisions/0/publish",
        json={"tempo_points": SLOW_POINTS},
    )
    assert pub.status_code == 200, pub.text
    second = annotate(rid, 2.0, "放慢后同秒").json()
    assert second["revision"] == 1
    assert second["position"] == "1:3:0"
    assert second["total_ticks"] == 960

    # 再次定位到 3:1:0 并在同一秒登记：落在新目标（rev1 下 3:1:0 = 4.0s）
    loc = locate(rid, 2.0, "3:1:0")
    assert loc.status_code == 200
    third = annotate(rid, 2.0, "重定位处").json()
    assert third["revision"] == 1
    assert third["position"] == "3:1:0"
    assert third["total_ticks"] == 3840

    # 既有批注全部保持登记时的位置与修订号
    listing = annotations(rid).json()
    by_id = {a["id"]: a for a in listing}
    assert by_id[first["id"]]["position"] == "2:1:0"
    assert by_id[first["id"]]["total_ticks"] == 1920
    assert by_id[first["id"]]["revision"] == 0
    assert by_id[second["id"]]["position"] == "1:3:0"
    assert by_id[second["id"]]["total_ticks"] == 960
    assert by_id[second["id"]]["revision"] == 1
    assert by_id[third["id"]]["position"] == "3:1:0"
    assert by_id[third["id"]]["total_ticks"] == 3840
    assert by_id[third["id"]]["revision"] == 1
    assert [a["id"] for a in listing] == [
        first["id"], second["id"], third["id"]
    ]


# ---------------------------------------------------------------------- #
# 并发：登记与发布交错时，进度/起点/速度图必须取自同一已提交状态
# ---------------------------------------------------------------------- #
def test_concurrent_annotations_against_publish_use_one_committed_revision(
    fresh_db,
):
    """多连接并发：一条发布可能在批注之间提交。

    每条批注的修订号与位置必须自洽：rev0（120 BPM）下 10.0s 对应
    9600 tick，rev1（60 BPM）下 10.0s 对应 4800 tick；seq 必须连续唯一。
    """
    from app.db import Store
    from app.main import _annotate_position

    score = create_score()
    sid = score["id"]
    r = create_rehearsal(sid)
    rid = r["id"]
    db_path = fresh_db
    assert claim(rid, 10.0).status_code == 200  # 先把进度固定在 10.0s

    n_workers = 8
    stores = [Store(str(db_path)) for _ in range(n_workers)]
    barrier = threading.Barrier(n_workers + 1)
    errors: list[Exception] = []
    results: list[dict] = []

    def worker(store):
        try:
            barrier.wait(timeout=10)
            res = store.add_annotation(
                rid,
                10.0,
                "并发批注",
                lambda s, elapsed: _annotate_position(s, elapsed),
            )
            results.append(res)
        except Exception as exc:  # pragma: no cover - 失败时下面断言暴露
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    barrier.wait(timeout=10)
    # 与批注并发提交一次发布（可能在任意批注前后落盘）。
    pub = client.post(
        f"/scores/{sid}/revisions/0/publish",
        json={"tempo_points": SLOW_POINTS},
    )
    assert pub.status_code == 200, pub.text
    for t in threads:
        t.join(timeout=30)
    assert not errors

    assert len(results) == n_workers
    seqs = sorted(a["seq"] for a in results)
    assert seqs == list(range(n_workers))  # 登记先后连续、不重号
    for a in results:
        if a["revision"] == 0:  # 120 BPM：1s = 960 tick
            assert a["total_ticks"] == 9600
            assert a["position"] == "6:1:0"
        else:  # 60 BPM：1s = 480 tick
            assert a["revision"] == 1
            assert a["total_ticks"] == 4800
            assert a["position"] == "3:3:0"

    listing = annotations(rid).json()
    assert len(listing) == n_workers
    assert [a["text"] for a in listing] == ["并发批注"] * n_workers


# ---------------------------------------------------------------------- #
# 持久化：新 Store 实例（模拟 Compose 重启）后批注仍可查询
# ---------------------------------------------------------------------- #
def test_annotations_survive_store_restart(fresh_db, monkeypatch):
    db_path = fresh_db
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)
    before = annotate(rid, 2.0, "重启前").json()

    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))

    listing = annotations(rid).json()
    assert len(listing) == 1
    got = listing[0]
    assert got["id"] == before["id"]
    assert got["text"] == "重启前"
    assert got["elapsed_seconds"] == 2.0
    assert got["position"] == "2:1:0"
    assert got["total_ticks"] == 1920
    assert got["revision"] == 0

    # 重启后仍可继续登记，seq 接续
    after = annotate(rid, 2.0, "重启后").json()
    ids = [a["id"] for a in annotations(rid).json()]
    assert ids == [before["id"], after["id"]]
