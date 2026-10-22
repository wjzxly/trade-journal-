"""API 集成测试：用 httpx 跑真实路由，覆盖录入 → 统计的完整链路。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trade_journal import repository as repo
from trade_journal.app import app
from trade_journal.db import init_db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个测试用独立的临时数据库，互不污染。"""
    db = tmp_path / "test.db"
    init_db(db)

    import trade_journal.db as dbmod

    monkeypatch.setattr(dbmod, "DEFAULT_DB", db)
    # repository 里用的是 session()，它读 DEFAULT_DB，同样需要指向临时库
    monkeypatch.setattr(repo, "session", lambda: dbmod.session(db))

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_rule_and_list(client):
    r = client.post("/api/rules", json={
        "name": "均线多头",
        "condition": "close > ma20 and vol_ratio > 1.5",
        "hypothesis": "放量突破有惯性",
    })
    assert r.status_code == 201
    rule_id = r.json()["id"]

    items = client.get("/api/rules").json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "均线多头"
    assert items[0]["version"] == "v1"
    assert rule_id == items[0]["id"]


def test_invalid_rule_rejected(client):
    r = client.post("/api/rules", json={"name": "坏规则", "condition": "close >> 10"})
    assert r.status_code == 400
    assert "规则条件有误" in r.json()["detail"]


def test_rule_condition_change_bumps_version(client):
    rid = client.post("/api/rules", json={"name": "A", "condition": "close > ma20"}).json()["id"]
    client.patch(f"/api/rules/{rid}", json={"condition": "close > ma60"})
    rule = client.get("/api/rules").json()["items"][0]
    assert rule["version"] == "v2"


def test_scan_detects_hit(client):
    client.post("/api/rules", json={
        "name": "放量突破",
        "condition": "close > ma20 and vol_ratio > 1.5",
    })
    r = client.post("/api/signals/scan", json={"rows": [
        {"symbol": "600522", "close": 12.4, "ma20": 11.8, "vol_ratio": 1.9},
        {"symbol": "000001", "close": 10.1, "ma20": 10.6, "vol_ratio": 0.7},
    ]})
    assert r.status_code == 200
    data = r.json()
    assert data["scanned_rows"] == 2
    assert len(data["hits"]) == 1
    assert data["hits"][0]["symbol"] == "600522"


def test_scan_with_no_rules(client):
    r = client.post("/api/signals/scan", json={"rows": [{"symbol": "X", "close": 1}]})
    assert r.json()["hits"] == []
    assert r.json()["rules_evaluated"] == 0


def test_trade_flow_produces_stats(client):
    rid = client.post("/api/rules", json={"name": "测试规则", "condition": "close > ma20"}).json()["id"]

    client.post("/api/trades", json={
        "symbol": "600522", "side": "buy", "price": 10.0, "qty": 1000,
        "trade_date": "2026-01-05", "rule_id": rid, "mood": "平静",
    })
    client.post("/api/trades", json={
        "symbol": "600522", "side": "sell", "price": 11.0, "qty": 1000,
        "trade_date": "2026-01-10", "rule_id": rid, "mood": "平静",
    })

    stats = client.get("/api/stats").json()
    assert stats["overview"]["round_trips"] == 1
    assert stats["overview"]["total_pnl"] == 1000.0
    assert len(stats["by_rule"]) == 1
    assert stats["by_rule"][0]["rule_name"] == "测试规则"


def test_signal_links_trade_automatically(client):
    rid = client.post("/api/rules", json={"name": "R", "condition": "close > 1"}).json()["id"]
    sid = client.post("/api/signals", json={
        "rule_id": rid, "symbol": "600522", "price": 10.0,
    }).json()["id"]

    client.post("/api/trades", json={
        "symbol": "600522", "side": "buy", "price": 10.0, "qty": 100, "signal_id": sid,
    })

    trades = client.get("/api/trades").json()["items"]
    assert trades[0]["rule_id"] == rid  # 从信号自动带出规则

    signals = client.get("/api/signals").json()["items"]
    assert signals[0]["acted"] == 1  # 自动标记已出手


def test_trade_validation(client):
    r = client.post("/api/trades", json={"symbol": "X", "side": "buy", "price": -1, "qty": 100})
    assert r.status_code == 422  # 价格必须为正

    r = client.post("/api/trades", json={"symbol": "X", "side": "hold", "price": 10, "qty": 100})
    assert r.status_code == 422  # 方向只能 buy/sell


def test_delete_trade(client):
    tid = client.post("/api/trades", json={
        "symbol": "X", "side": "buy", "price": 10, "qty": 100,
    }).json()["id"]
    client.delete(f"/api/trades/{tid}")
    assert client.get("/api/trades").json()["items"] == []


def test_review_and_attribution(client):
    client.post("/api/reviews", json={
        "symbol": "600522", "outcome": "win", "pnl_pct": 5.2,
        "attribution": "rule", "what_went_right": "等到回踩才买",
    })
    client.post("/api/reviews", json={
        "symbol": "000001", "outcome": "loss", "pnl_pct": -3.1,
        "attribution": "execution", "what_to_fix": "追高了",
    })

    reviews = client.get("/api/reviews").json()["items"]
    assert len(reviews) == 2

    stats = client.get("/api/stats").json()
    assert stats["attribution"] == {"rule": 1, "execution": 1}


def test_rule_retire(client):
    rid = client.post("/api/rules", json={"name": "临时", "condition": "close > 1"}).json()["id"]
    client.delete(f"/api/rules/{rid}")
    rule = client.get("/api/rules").json()["items"][0]
    assert rule["active"] == 0

    active = client.get("/api/rules?active_only=true").json()["items"]
    assert active == []


def test_trade_stop_above_buy_rejected(client):
    """止损价填得比买入价高，说明填反了，必须拒绝而不是静默算出负风险。"""
    r = client.post("/api/trades", json={
        "symbol": "600522", "side": "buy", "price": 10.0, "qty": 100, "stop_price": 11.0,
    })
    assert r.status_code == 400
    assert "止损" in r.json()["detail"]


def test_sell_trade_stop_ignored(client):
    """卖出单不该带止损，即使传了也应被清空，避免污染 R 倍数。"""
    client.post("/api/trades", json={
        "symbol": "600522", "side": "sell", "price": 10.0, "qty": 100, "stop_price": 9.0,
    })
    trade = client.get("/api/trades").json()["items"][0]
    assert trade["stop_price"] is None


def test_stop_price_flows_into_r_multiple(client):
    rid = client.post("/api/rules", json={
        "name": "带止损的规则", "condition": "close > ma20", "stop_loss_pct": 5.0,
    }).json()["id"]

    # 买 10 止损 9.5（5% 风险），卖 11（+10%）→ 2R
    client.post("/api/trades", json={
        "symbol": "600522", "side": "buy", "price": 10.0, "qty": 1000,
        "trade_date": "2026-01-05", "rule_id": rid, "stop_price": 9.5,
    })
    client.post("/api/trades", json={
        "symbol": "600522", "side": "sell", "price": 11.0, "qty": 1000,
        "trade_date": "2026-01-10", "rule_id": rid,
    })

    stats = client.get("/api/stats").json()
    trip = stats["round_trips"][0]
    assert trip["risk_pct"] == 5.0
    assert trip["risk_source"] == "actual"
    assert trip["r_multiple"] == 2.0
    assert stats["risk_coverage"]["actual"] == 1


def test_rule_stop_pct_persisted(client):
    client.post("/api/rules", json={
        "name": "R", "condition": "close > 1", "stop_loss_pct": 4.5,
    })
    rule = client.get("/api/rules").json()["items"][0]
    assert rule["stop_loss_pct"] == 4.5


def test_rule_stop_pct_out_of_range_rejected(client):
    r = client.post("/api/rules", json={
        "name": "R", "condition": "close > 1", "stop_loss_pct": 0.1,
    })
    assert r.status_code == 422  # 过窄的止损没有意义


def test_migration_adds_columns_to_old_db(tmp_path):
    """老版本数据库升级后应自动补列，不丢数据。"""
    import sqlite3

    from trade_journal.db import init_db

    db = tmp_path / "old.db"
    # 手工造一个缺列的老表
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'entry',
            condition TEXT NOT NULL, version TEXT NOT NULL DEFAULT 'v1',
            hypothesis TEXT DEFAULT '', notes TEXT DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            retired_at TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL,
            qty INTEGER NOT NULL, trade_date TEXT NOT NULL,
            signal_id INTEGER, rule_id INTEGER,
            followed INTEGER NOT NULL DEFAULT 1, deviation TEXT DEFAULT '',
            mood TEXT DEFAULT '', position_pct REAL DEFAULT 0,
            fee REAL DEFAULT 0, notes TEXT DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO rules (name, condition) VALUES ('老规则', 'close > 1');
    """)
    conn.commit()
    conn.close()

    init_db(db)  # 触发迁移

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rule_cols = {r["name"] for r in conn.execute("PRAGMA table_info(rules)")}
    trade_cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)")}
    old = conn.execute("SELECT name FROM rules").fetchone()
    conn.close()

    assert "stop_loss_pct" in rule_cols
    assert "stop_price" in trade_cols
    assert old["name"] == "老规则"  # 数据还在


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Trade Journal" in r.text
