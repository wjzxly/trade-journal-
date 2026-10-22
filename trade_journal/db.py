"""数据层：SQLite schema 与访问封装。

四张表构成闭环：

    rules      规则（假设）
      ↓
    signals    信号（规则触发的证据，只记录，不代表成交）
      ↓
    trades     操作（真实成交，含是否按规则执行 ← 归因的基石）
      ↓
    reviews    复盘（事后回看，只写事实不写辩解）

刻意不做的事：不存行情、不调接口、不预测价格。行情属于外部世界，
本系统只关心「我做了什么、我为什么这么做、结果如何」。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "journal.db"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 规则：一条可判定的交易假设
CREATE TABLE IF NOT EXISTS rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    -- 规则分类：entry 入场 / exit 离场 / risk 风控 / filter 过滤
    kind        TEXT    NOT NULL DEFAULT 'entry',
    -- 可判定的触发条件，见 rule_engine.py 的 DSL 说明
    condition   TEXT    NOT NULL,
    -- 版本号：规则一旦改动必须升版，历史样本归属于旧版本
    version     TEXT    NOT NULL DEFAULT 'v1',
    hypothesis  TEXT    DEFAULT '',      -- 这条规则押注的是什么市场规律
    notes       TEXT    DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    retired_at  TEXT
);

-- 信号：规则被触发的记录。信号 ≠ 成交
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id     INTEGER NOT NULL REFERENCES rules(id),
    rule_version TEXT   NOT NULL,        -- 冗余存储，规则升版后仍可追溯
    symbol      TEXT    NOT NULL,
    trade_date  TEXT    NOT NULL,
    price       REAL,
    hit         INTEGER NOT NULL DEFAULT 1, -- 1=条件成立 0=未成立（作为对照样本）
    acted       INTEGER,                 -- NULL=未决定 / 1=出手 / 0=放弃
    skip_reason TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 操作：真实成交。这是统计的唯一事实来源
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL CHECK (side IN ('buy', 'sell')),
    price       REAL    NOT NULL,
    qty         INTEGER NOT NULL,
    trade_date  TEXT    NOT NULL,
    -- 关联信号；为 NULL 表示「计划外操作」，这本身就是重要样本
    signal_id   INTEGER REFERENCES signals(id),
    rule_id     INTEGER REFERENCES rules(id),
    -- 是否按规则执行：统计时按此分组，区分「规则行不行」和「我执不执行」
    followed    INTEGER NOT NULL DEFAULT 1,
    deviation   TEXT    DEFAULT '',      -- 偏离规则的动机，必须用第一人称写实话
    -- 情绪：下单那一刻的状态，用于长期挖掘亏损模式
    mood        TEXT    DEFAULT '',
    position_pct REAL   DEFAULT 0,       -- 占仓位比例 0~100
    fee         REAL    DEFAULT 0,
    notes       TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 复盘：事后归因。只写事实，禁止写「要不是……」
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id    INTEGER REFERENCES trades(id),
    symbol      TEXT,
    review_date TEXT    NOT NULL,
    -- 结果：win / loss / flat
    outcome     TEXT,
    pnl         REAL    DEFAULT 0,
    pnl_pct     REAL    DEFAULT 0,
    -- 归因：rule 规则有效 / execution 执行偏差 / market 系统性风险 / luck 运气
    attribution TEXT    DEFAULT 'rule',
    what_went_right TEXT DEFAULT '',
    what_to_fix     TEXT DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_signals_date  ON signals(trade_date);
CREATE INDEX IF NOT EXISTS idx_signals_rule  ON signals(rule_id);
CREATE INDEX IF NOT EXISTS idx_trades_date   ON trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_rule   ON trades(rule_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_reviews_date  ON reviews(review_date);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开连接，行以 dict 方式访问。"""
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def session(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """事务上下文：正常结束提交，异常回滚。"""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
