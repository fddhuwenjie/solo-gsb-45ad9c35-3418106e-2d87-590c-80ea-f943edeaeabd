"""排练批注：登记（反推乐谱位置）/ 按登记先后查询 / 冻结与并发 端到端测试。"""

import json
import math
import sqlite3
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

# 120 BPM、4/4、PPQ=480：一拍 0.5s、一小节 2.0s、一 tick = 1/960 s
# 1:1:0 -> 0.0s，2:1:0 -> 2.0s，3:1:0 -> 4.0s，4:1:0 -> 6.0s
SCORE_BODY = {
    "ppq": 480,
    "time_signatures": [],
    "tempo_points": [{"position": "1:1:0", "bpm": 120, "curve": "step"}],
    "anchors": [{"position": "1:1:0", "locked": True}],
}
# 替换表：整体降到 60 BPM（所有图内秒数翻倍）
SLOW_POINTS = [{"position": "1:1:0", "bpm": 60, "curve": "step"}]


def create_score():
    resp = client.post("/scores", json=SCORE_BODY)
    assert resp.status_code == 201, resp.text
    return resp.json()


def create_rehearsal(score_id, cues=({"position": "2:1:0", "text": "B"},)):
    resp = client.post("/rehearsals", json={"score_id": score_id, "cues": cues})
    assert resp.status_code == 201, resp.text
    return resp.json()


def claim(rid, elapsed):
    return client.post(f"/rehearsals/{rid}/claims", json={"elapsed_seconds": elapsed})


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


def list_annotations(rid):
    return client.get(f"/rehearsals/{rid}/annotations")


# ---------------------------------------------------------------------- #
# 登记：位置反推与响应字段
# ---------------------------------------------------------------------- #
def test_create_annotation_returns_position_ticks_and_revision():
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)  # 进度推进到 2.0s（120 BPM 下即 2:1:0）

    resp = annotate(rid, 2.0, "这里进副歌")
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert isinstance(data["id"], str) and len(data["id"]) == 32
    assert data["text"] == "这里进副歌"
    assert data["elapsed_seconds"] == 2.0
    assert data["position"] == "2:1:0"
    assert data["total_ticks"] == 1920
    assert data["revision"] == 0

    # 查询返回同一条批注（登记时冻结的值）
    got = list_annotations(rid)
    assert got.status_code == 200
    body = got.json()
    assert body["id"] == rid
    assert body["annotations"] == [data]


def test_annotation_at_fresh_rehearsal_maps_from_score_start():
    # 未领取、未定位：进度 0.0，从乐谱起点算 -> 1:1:0 / tick 0
    score = create_score()
    r = create_rehearsal(score["id"])
    data = annotate(r["id"], 0.0, "开场笔记").json()
    assert data["position"] == "1:1:0"
    assert data["total_ticks"] == 0
    assert data["elapsed_seconds"] == 0.0


def test_annotation_same_second_as_locate_lands_on_target():
    # 刚定位的同一秒登记：批注应落在定位目标位置
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)
    locate(rid, 2.0, "4:1:0")  # 起点 (4:1:0 = 6.0s, 累计 2.0s)

    data = annotate(rid, 2.0, "跳到这里的笔记").json()
    assert data["position"] == "4:1:0"
    assert data["total_ticks"] == 5760
    assert data["elapsed_seconds"] == 2.0


def test_annotation_after_locate_maps_relative_to_origin():
    # 定位到 4:1:0（图内 6.0s）时累计 2.0s；进度推进到 3.0s 后登记：
    # 图内秒数 = 6.0 + (3.0 - 2.0) = 7.0s -> tick 6720 -> 4:3:0
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)
    locate(rid, 2.0, "4:1:0")
    claim(rid, 3.0)

    data = annotate(rid, 3.0, "定位后一秒").json()
    assert data["position"] == "4:3:0"
    assert data["total_ticks"] == 6720


def test_annotation_quantizes_to_nearest_tick():
    # 120 BPM 下一 tick = 1/960 s：落在两 tick 之间时量化到最近 tick
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]

    claim(rid, 2.0 + 0.4 / 960)  # 1920.4 tick -> 1920
    down = annotate(rid, 2.0 + 0.4 / 960, "偏下").json()
    assert down["total_ticks"] == 1920
    assert down["position"] == "2:1:0"

    claim(rid, 2.0 + 0.6 / 960)  # 1920.6 tick -> 1921
    up = annotate(rid, 2.0 + 0.6 / 960, "偏上").json()
    assert up["total_ticks"] == 1921
    assert up["position"] == "2:1:1"


def test_annotation_does_not_advance_progress_or_touch_cues():
    score = create_score()
    r = create_rehearsal(score["id"], [{"position": "3:1:0", "text": "C"}])
    rid = r["id"]
    claim(rid, 2.0)
    annotate(rid, 2.0, "批注不推进进度")

    got = client.get(f"/rehearsals/{rid}").json()
    assert got["last_elapsed_seconds"] == 2.0
    assert got["cues"][0]["status"] == "pending"
    # 同一秒重复领取不受影响；倒退仍 409
    assert claim(rid, 2.0).json()["cues"] == []
    assert claim(rid, 1.0).status_code == 409


# ---------------------------------------------------------------------- #
# 校验：秒数须等于当前进度（409）、有限非负（422）、文字非空（422）
# ---------------------------------------------------------------------- #
def test_annotation_requires_elapsed_equal_to_progress():
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)

    for bad in (1.5, 2.5):
        resp = annotate(rid, bad, "进度不匹配")
        assert resp.status_code == 409, bad
        detail = resp.json()["detail"][0]
        assert detail["loc"] == ["body", "elapsed_seconds"]
        assert detail["elapsed_seconds"] == bad
        assert detail["last_elapsed_seconds"] == 2.0

    # 不写入：没有任何批注；进度与提示状态不变
    assert list_annotations(rid).json()["annotations"] == []
    got = client.get(f"/rehearsals/{rid}").json()
    assert got["last_elapsed_seconds"] == 2.0
    # 随后按正确秒数仍可登记
    assert annotate(rid, 2.0, "补登记").status_code == 201


def test_annotation_rejects_invalid_elapsed_and_blank_text():
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]

    assert annotate(rid, -0.5, "负秒数").status_code == 422
    # NaN/Infinity 不是标准 JSON，但 Python json.loads 会接受：直接发原始请求体
    for raw in ("NaN", "Infinity", "-Infinity"):
        resp = client.post(
            f"/rehearsals/{rid}/annotations",
            content=f'{{"elapsed_seconds": {raw}, "text": "x"}}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw

    for blank in ("", "   "):
        resp = annotate(rid, 0.0, blank)
        assert resp.status_code == 422, blank
        assert resp.json()["detail"][0]["loc"] == ["body", "text"]

    # 全部拒绝：不写入任何批注
    assert list_annotations(rid).json()["annotations"] == []


def test_annotation_unknown_rehearsal_is_404():
    resp = annotate("deadbeef", 0.0, "x")
    assert resp.status_code == 404
    assert resp.json()["detail"][0]["loc"] == ["path", "rehearsal_id"]
    assert list_annotations("deadbeef").status_code == 404


# ---------------------------------------------------------------------- #
# 按登记先后查询
# ---------------------------------------------------------------------- #
def test_annotations_listed_in_registration_order_not_position_order():
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]

    claim(rid, 2.0)
    a = annotate(rid, 2.0, "A").json()  # 2:1:0
    claim(rid, 4.0)
    b = annotate(rid, 4.0, "B").json()  # 3:1:0
    locate(rid, 4.0, "1:2:0")  # 回跳到 1:2:0（图内 0.5s）
    c = annotate(rid, 4.0, "C").json()  # 1:2:0（位置在 A/B 之前）

    assert (a["position"], b["position"], c["position"]) == (
        "2:1:0",
        "3:1:0",
        "1:2:0",
    )
    got = list_annotations(rid).json()["annotations"]
    # 按登记先后（A, B, C），而非乐谱位置先后
    assert [x["id"] for x in got] == [a["id"], b["id"], c["id"]]
    assert [x["text"] for x in got] == ["A", "B", "C"]


# ---------------------------------------------------------------------- #
# 冻结：发布或再次定位后，既有批注的位置与修订号保持登记时的值
# ---------------------------------------------------------------------- #
def test_annotations_frozen_after_publish_and_relocate():
    score = create_score()
    sid = score["id"]
    r = create_rehearsal(sid)
    rid = r["id"]
    claim(rid, 2.0)
    first = annotate(rid, 2.0, "旧图批注").json()  # 2:1:0 / 1920 / rev0
    assert first["revision"] == 0

    # 发布 60 BPM（rev1）：图内秒数翻倍；再定位到 3:1:0
    pub = client.post(
        f"/scores/{sid}/revisions/0/publish", json={"tempo_points": SLOW_POINTS}
    )
    assert pub.status_code == 200
    locate(rid, 2.0, "3:1:0")  # 起点 (3:1:0 = 8.0s@60BPM, 累计 2.0s)
    claim(rid, 3.0)

    # 既有批注保持登记时的位置与修订号
    got = list_annotations(rid).json()["annotations"]
    assert got == [first]

    # 新批注按新修订、新起点计算：图内 8.0 + (3.0-2.0) = 9.0s
    # 60 BPM 下 9.0s = 4320 tick -> 3:2:0
    second = annotate(rid, 3.0, "新图批注").json()
    assert second["position"] == "3:2:0"
    assert second["total_ticks"] == 4320
    assert second["revision"] == 1

    got = list_annotations(rid).json()["annotations"]
    assert [x["id"] for x in got] == [first["id"], second["id"]]
    assert got[0]["position"] == "2:1:0" and got[0]["revision"] == 0


# ---------------------------------------------------------------------- #
# SQLite 持久化：新 Store 实例（模拟 Compose 重启）后批注仍可读、可继续登记
# ---------------------------------------------------------------------- #
def test_annotations_survive_store_restart(fresh_db, monkeypatch):
    db_path = fresh_db
    score = create_score()
    r = create_rehearsal(score["id"])
    rid = r["id"]
    claim(rid, 2.0)
    first = annotate(rid, 2.0, "重启前").json()

    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))

    got = list_annotations(rid).json()["annotations"]
    assert got == [first]
    # 进度保留，可继续登记；新批注排在既有批注之后
    claim(rid, 4.0)
    second = annotate(rid, 4.0, "重启后").json()
    assert second["position"] == "3:1:0"
    got = list_annotations(rid).json()["annotations"]
    assert [x["text"] for x in got] == ["重启前", "重启后"]


# ---------------------------------------------------------------------- #
# 并发：登记与领取交错时，进度/起点/速度图取自同一已提交状态
# ---------------------------------------------------------------------- #
def test_concurrent_annotations_and_claims_never_use_stale_state(fresh_db):
    """多连接并发：一半领取（推进进度到 1000），一半按旧进度 0.0 登记。

    登记与领取共用写锁串行化：批注要么在某个领取提交前登记成功
    （elapsed=0.0、位置 1:1:0、rev0），要么因进度已前进而 409 不写入；
    不会出现“读到旧进度却按新状态写入”的拼接。
    """
    from app.db import AnnotationMismatch, Store
    from app.main import _annotation_score_position, _due_cues

    score = create_score()
    r = create_rehearsal(
        score["id"],
        [{"position": f"{m}:1:0", "text": f"cue{m}"} for m in range(1, 4)],
    )
    rid = r["id"]
    db_path = fresh_db

    n_workers = 8
    stores = [Store(str(db_path)) for _ in range(n_workers)]
    barrier = threading.Barrier(n_workers)
    errors: list[Exception] = []
    rejected: list[str] = []
    registered: list[dict] = []

    def worker(i, store):
        try:
            barrier.wait(timeout=10)
            if i % 2 == 0:
                store.claim_cues(
                    rid, 1000.0, lambda s, p: _due_cues(s, p, 1000.0)
                )
            else:
                try:
                    registered.append(
                        store.create_annotation(
                            rid, 0.0, f"note{i}", _annotation_score_position
                        )
                    )
                except AnnotationMismatch:
                    # 领取先到时进度已是 1000，按 0.0 登记必须被拒绝且不写入
                    rejected.append("annotate")
        except Exception as exc:  #  pragma: no cover - 失败时下面断言暴露
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(i, s))
        for i, s in enumerate(stores)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors

    # 成功的登记全部基于进度 0.0 的同一已提交状态（起点 0、rev0 速度图）
    for a in registered:
        assert a["elapsed_seconds"] == 0.0
        assert a["position"] == "1:1:0"
        assert a["total_ticks"] == 0
        assert a["revision"] == 0
    # 落库的批注与成功登记一一对应，序号连续（按登记先后 0..n-1）
    got = client.get(f"/rehearsals/{rid}/annotations").json()["annotations"]
    assert sorted(a["id"] for a in registered) == sorted(a["id"] for a in got)
    # 进度最终推进到 1000；之后再按 0.0 登记必须 409
    assert client.get(f"/rehearsals/{rid}").json()["last_elapsed_seconds"] == 1000.0
    assert annotate(rid, 0.0, "太迟了").status_code == 409


# ---------------------------------------------------------------------- #
# 旧版库迁移：无 annotations 表的库打开后自动建表，可正常登记
# ---------------------------------------------------------------------- #
def test_old_schema_db_gets_annotations_table(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE scores (id TEXT PRIMARY KEY, ppq INTEGER NOT NULL,
            time_signatures TEXT NOT NULL, tempo_points TEXT NOT NULL,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE anchors (score_id TEXT NOT NULL, position_index INTEGER NOT NULL,
            position TEXT NOT NULL, locked INTEGER NOT NULL, total_ticks INTEGER NOT NULL,
            original_seconds REAL NOT NULL, current_seconds REAL NOT NULL,
            tick_seconds REAL NOT NULL, PRIMARY KEY (score_id, position_index));
        CREATE TABLE rehearsals (id TEXT PRIMARY KEY, score_id TEXT NOT NULL,
            last_elapsed REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE rehearsal_cues (rehearsal_id TEXT NOT NULL, cue_index INTEGER NOT NULL,
            position TEXT NOT NULL, text TEXT NOT NULL, total_ticks INTEGER NOT NULL,
            PRIMARY KEY (rehearsal_id, cue_index));
        CREATE TABLE cue_deliveries (rehearsal_id TEXT NOT NULL, cue_index INTEGER NOT NULL,
            trigger_seconds REAL NOT NULL, revision INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            delivered_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (rehearsal_id, cue_index));
        """
    )
    points = json.dumps(
        [{"position": "1:1:0", "bpm": 120, "curve": "step"}], ensure_ascii=False
    )
    conn.execute(
        "INSERT INTO scores VALUES ('s1', 480, '[]', ?, 0, datetime('now'))",
        (points,),
    )
    conn.execute(
        "INSERT INTO anchors VALUES ('s1', 0, '1:1:0', 1, 0, 0.0, 0.0, 0.001)"
    )
    conn.execute(
        "INSERT INTO rehearsals (id, score_id, last_elapsed) VALUES ('r1','s1', 2.0)"
    )
    conn.execute("INSERT INTO rehearsal_cues VALUES ('r1', 0, '2:1:0', 'B', 1920)")
    conn.commit()
    conn.close()

    from app.db import Store
    from app.main import _annotation_score_position

    # 打开旧库：自动补定位起点列、cue_skips 表与 annotations 表
    store = Store(str(db_path))
    out = store.create_annotation("r1", 2.0, "旧库批注", _annotation_score_position)
    assert out["position"] == "2:1:0"
    assert out["total_ticks"] == 1920
    assert out["revision"] == 0
    got = store.get_annotations("r1")
    assert [a["text"] for a in got] == ["旧库批注"]
