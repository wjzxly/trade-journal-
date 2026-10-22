"""数据访问层：所有 SQL 集中在这里，路由层不写裸 SQL。"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any, Sequence

from .db import session


# ---------------------------------------------------------------- rules

def list_rules(active_only: bool = False) -> list[sqlite3.Row]:
    sql = "SELECT * FROM rules"
    if active_only:
        sql += " WHERE active = 1"
    sql += " ORDER BY active DESC, id DESC"
    with session() as conn:
        return conn.execute(sql).fetchall()


def get_rule(rule_id: int) -> sqlite3.Row | None:
    with session() as conn:
        return conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()


def create_rule(
    name: str,
    condition: str,
    kind: str = "entry",
    hypothesis: str = "",
    notes: str = "",
    version: str = "v1",
    stop_loss_pct: float | None = None,
) -> int:
    with session() as conn:
        cur = conn.execute(
            """INSERT INTO rules (name, kind, condition, version, hypothesis, notes, stop_loss_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, kind, condition, version, hypothesis, notes, stop_loss_pct),
        )
        return int(cur.lastrowid)


def update_rule(rule_id: int, **fields: Any) -> None:
    """修改规则。若条件变了但没升版本号，自动把版本推到 v2/v3……

    强制升版是为了保护历史样本的可解释性——改过的规则和新样本不是同一件事。
    """
    allowed = {
        "name", "kind", "condition", "hypothesis", "notes",
        "active", "version", "stop_loss_pct",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    with session() as conn:
        old = conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
        if old is None:
            return
        if "condition" in updates and updates["condition"] != old["condition"] and "version" not in updates:
            updates["version"] = _next_version(old["version"])
        sets = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE rules SET {sets} WHERE id = ?", (*updates.values(), rule_id))


def retire_rule(rule_id: int) -> None:
    with session() as conn:
        conn.execute(
            "UPDATE rules SET active = 0, retired_at = datetime('now','localtime') WHERE id = ?",
            (rule_id,),
        )


def _next_version(current: str) -> str:
    try:
        return f"v{int(current.lstrip('vV')) + 1}"
    except ValueError:
        return "v2"


def rules_map() -> dict[int, sqlite3.Row]:
    return {r["id"]: r for r in list_rules()}


# ---------------------------------------------------------------- signals

def create_signal(
    rule_id: int,
    symbol: str,
    trade_date: str,
    price: float | None = None,
    hit: bool = True,
    acted: bool | None = None,
    skip_reason: str = "",
) -> int:
    with session() as conn:
        rule = conn.execute("SELECT version FROM rules WHERE id = ?", (rule_id,)).fetchone()
        if rule is None:
            raise ValueError(f"规则 {rule_id} 不存在")
        cur = conn.execute(
            """INSERT INTO signals (rule_id, rule_version, symbol, trade_date, price, hit, acted, skip_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (rule_id, rule["version"], symbol, trade_date, price, int(hit),
             None if acted is None else int(acted), skip_reason),
        )
        return int(cur.lastrowid)


def list_signals(limit: int = 100, acted: str = "all") -> list[sqlite3.Row]:
    sql = """SELECT s.*, r.name AS rule_name FROM signals s
             LEFT JOIN rules r ON r.id = s.rule_id"""
    params: list[Any] = []
    if acted == "pending":
        sql += " WHERE s.acted IS NULL"
    elif acted == "acted":
        sql += " WHERE s.acted = 1"
    elif acted == "skipped":
        sql += " WHERE s.acted = 0"
    sql += " ORDER BY s.trade_date DESC, s.id DESC LIMIT ?"
    params.append(limit)
    with session() as conn:
        return conn.execute(sql, params).fetchall()


def mark_signal(signal_id: int, acted: bool, skip_reason: str = "") -> None:
    with session() as conn:
        conn.execute(
            "UPDATE signals SET acted = ?, skip_reason = ? WHERE id = ?",
            (int(acted), skip_reason, signal_id),
        )


# ---------------------------------------------------------------- trades

def create_trade(
    symbol: str,
    side: str,
    price: float,
    qty: int,
    trade_date: str | None = None,
    signal_id: int | None = None,
    rule_id: int | None = None,
    followed: bool = True,
    deviation: str = "",
    mood: str = "",
    position_pct: float = 0.0,
    fee: float = 0.0,
    notes: str = "",
    stop_price: float | None = None,
) -> int:
    td = trade_date or date.today().isoformat()
    with session() as conn:
        # 传了信号就自动带上规则和「已出手」标记，减少手工录入成本
        if signal_id is not None:
            sig = conn.execute("SELECT rule_id FROM signals WHERE id = ?", (signal_id,)).fetchone()
            if sig and rule_id is None:
                rule_id = sig["rule_id"]
            conn.execute("UPDATE signals SET acted = 1 WHERE id = ?", (signal_id,))
        # 卖出单不该带止损价，避免上游误传污染 R 倍数统计
        if side == "sell":
            stop_price = None
        cur = conn.execute(
            """INSERT INTO trades (symbol, side, price, qty, trade_date, signal_id, rule_id,
                                   followed, deviation, mood, position_pct, fee, notes, stop_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol.strip().upper(), side, price, qty, td, signal_id, rule_id,
             int(followed), deviation, mood, position_pct, fee, notes, stop_price),
        )
        return int(cur.lastrowid)


def list_trades(limit: int = 200, symbol: str | None = None) -> list[sqlite3.Row]:
    sql = """SELECT t.*, r.name AS rule_name FROM trades t
             LEFT JOIN rules r ON r.id = t.rule_id"""
    params: list[Any] = []
    if symbol:
        sql += " WHERE t.symbol = ?"
        params.append(symbol.upper())
    sql += " ORDER BY t.trade_date DESC, t.id DESC LIMIT ?"
    params.append(limit)
    with session() as conn:
        return conn.execute(sql, params).fetchall()


def all_trades() -> list[sqlite3.Row]:
    with session() as conn:
        return conn.execute("SELECT * FROM trades ORDER BY trade_date, id").fetchall()


def delete_trade(trade_id: int) -> None:
    with session() as conn:
        conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))


# ---------------------------------------------------------------- reviews

def create_review(
    trade_id: int | None,
    symbol: str,
    outcome: str,
    pnl: float = 0.0,
    pnl_pct: float = 0.0,
    attribution: str = "rule",
    what_went_right: str = "",
    what_to_fix: str = "",
    review_date: str | None = None,
) -> int:
    with session() as conn:
        cur = conn.execute(
            """INSERT INTO reviews (trade_id, symbol, review_date, outcome, pnl, pnl_pct,
                                    attribution, what_went_right, what_to_fix)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade_id, symbol.strip().upper(), review_date or date.today().isoformat(),
             outcome, pnl, pnl_pct, attribution, what_went_right, what_to_fix),
        )
        return int(cur.lastrowid)


def list_reviews(limit: int = 100) -> list[sqlite3.Row]:
    with session() as conn:
        return conn.execute(
            "SELECT * FROM reviews ORDER BY review_date DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()


def attribution_breakdown() -> dict[str, int]:
    with session() as conn:
        rows = conn.execute(
            "SELECT attribution, COUNT(*) AS n FROM reviews GROUP BY attribution"
        ).fetchall()
    return {r["attribution"]: r["n"] for r in rows}


# ---------------------------------------------------------------- 辅助

def streak_days() -> int:
    """连续记录天数。用连续打卡对抗「懒一天就荒废」的人性。"""
    with session() as conn:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM trades ORDER BY trade_date DESC LIMIT 400"
        ).fetchall()
    days = {r["trade_date"] for r in rows}
    if not days:
        return 0
    today = date.today()
    cursor = today if today.isoformat() in days else today - timedelta(days=1)
    count = 0
    while cursor.isoformat() in days:
        count += 1
        cursor -= timedelta(days=1)
    return count
