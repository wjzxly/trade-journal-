"""生成演示数据：让你在没记录之前就能看到统计长什么样。

明确声明：这里的数字是**模拟的**，用来验证系统能跑，不能当成任何业绩。
真实使用时删掉 data/journal.db 即可从零开始。
"""

from __future__ import annotations

import random
import sys
from datetime import date, timedelta

from . import repository as repo
from .db import init_db

RULES = [
    {
        "name": "放量站上20日线",
        "condition": "close > ma20 and vol_ratio > 1.5 and macd_hist > 0",
        "kind": "entry",
        "hypothesis": "放量突破中期均线，短期有惯性，中短线可吃 3~5 个点",
        "notes": "核心规则。必须同时满足量能，否则容易假突破。",
    },
    {
        "name": "缩量回踩不破",
        "condition": "close > ma60 and pct_chg > -2 and vol_ratio < 0.8",
        "kind": "entry",
        "hypothesis": "上升趋势中的缩量回调是洗盘，不是转势",
        "notes": "胜率不高但盈亏比好，靠赚大的覆盖亏小的。",
    },
    {
        "name": "破位离场",
        "condition": "close < ma10 and pnl_pct < -5",
        "kind": "exit",
        "hypothesis": "跌破10日线且亏损超5%，趋势已坏，不留恋",
        "notes": "这条规则的价值不是赚钱，是限制单笔亏损。",
    },
    {
        "name": "情绪追高",
        "condition": "pct_chg > 7 and vol_ratio > 2.5",
        "kind": "entry",
        "hypothesis": "涨停板附近追入，赌次日溢价",
        "notes": "反面案例。故意留着，用数据看它到底亏多少。",
    },
]

# 模拟三只票的典型行为：赚的、亏的
# 注意元组顺序：(规则序号, 平均收益率%, 波动率(标准差), 样本数, 是否按规则, 情绪)
# 曾经把这里写成 (mean, vol) 却按 (vol, mean) 传给 gauss，导致收益被放大十几倍——
# 演示数据失真会让人误判系统能力，所以这里是必须钉死的地方。
PATTERNS = [
    (0, 3.8, 4.2, 14, True, "平静"),
    (1, 2.4, 3.6, 9, True, "笃定"),
    (2, -4.1, 2.0, 11, True, "平静"),
    (3, -3.2, 5.5, 7, True, "贪婪"),
    (0, -1.8, 6.5, 5, False, "焦虑"),
    (1, -2.6, 4.0, 4, False, "恐惧"),
    (None, -1.2, 5.0, 6, False, "报复性"),
]


def seed(seed_value: int = 20261023) -> None:
    init_db()
    existing = repo.list_rules()
    if existing:
        print("数据库已有规则，跳过播种以免污染真实数据。")
        print("如需重来，请删除 data/journal.db 后再执行。")
        return

    rng = random.Random(seed_value)
    rule_ids = []
    for r in RULES:
        rule_ids.append(repo.create_rule(
            r["name"], r["condition"], r["kind"], r["hypothesis"], r["notes"]
        ))
    print(f"已创建 {len(rule_ids)} 条规则")

    symbols = ["600522", "000001", "300750", "002594", "601318", "600036"]
    start = date.today() - timedelta(days=120)
    day = start
    count = 0
    # 每只股票记录「下一个可买入日期」，保证同一只票的持仓区间互不重叠。
    # 否则会出现「同一天既买入又卖出同一只票」的交叉数据，
    # FIFO 配对会把收益算错——这是前面踩过的坑。
    free_from = {s: start for s in symbols}

    for rule_idx, mean, vol, n, followed, mood in PATTERNS:
        rid = rule_ids[rule_idx] if rule_idx is not None else None
        for _ in range(n):
            sym = rng.choice(symbols)
            hold = rng.randint(2, 12)
            entry = max(day, free_from[sym]) + timedelta(days=rng.randint(0, 8))
            exit_ = entry + timedelta(days=hold)
            if exit_ > date.today():
                continue
            free_from[sym] = exit_ + timedelta(days=1)
            buy = round(rng.uniform(8, 60), 2)
            pct = rng.gauss(mean, vol)
            # 单笔中短线交易的收益不可能超出这个范围，超了就是采样异常，直接钳住
            pct = max(-12.0, min(15.0, pct))
            sell = round(buy * (1 + pct / 100), 2)
            qty = rng.choice([100, 200, 300, 500, 1000]) * 10

            dev = "" if followed else rng.choice([
                "怕回撤提前跑了", "手痒追高，没等回踩", "看到群里说好就买了",
                "该止损没止损，扛了三天",
            ])

            sig_id = None
            if rid is not None:
                sig_id = repo.create_signal(rid, sym, entry.isoformat(), buy)
                if not followed:
                    repo.mark_signal(sig_id, False, dev)

            repo.create_trade(
                sym, "buy", buy, qty, entry.isoformat(), sig_id, rid,
                followed, dev, mood,
                round(rng.uniform(5, 30), 1), round(qty * buy * 0.0003, 2),
            )
            repo.create_trade(
                sym, "sell", sell, qty, exit_.isoformat(), None, rid,
                followed, "", mood, 0.0, round(qty * sell * 0.0003, 2),
            )
            count += 1
            day = day + timedelta(days=1) if rng.random() < 0.35 else day

    print(f"已生成 {count} 笔完整交易（买卖各一）")
    print("提示：这些是模拟数据，用于验证统计链路，不代表任何业绩。")
    print("开始真实记录前请删除 data/journal.db。")


if __name__ == "__main__":
    seed(int(sys.argv[1]) if len(sys.argv) > 1 else 20261023)
