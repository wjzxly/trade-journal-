"""统计与归因：把「我感觉还行」变成「期望值 +0.8R，样本 34 笔」。

三个层次
========
1. 单规则体检   —— 这条规则在我自己的样本里期望值是多少？
2. 执行偏差分析 —— 规则没问题但我不照做的时候，亏多少？
3. 情绪模式     —— 哪类情绪下的操作在系统性亏钱？

所有统计都基于【已配对盈亏】的交易。一笔卖出若能通过备注里的 symbol 配对
上一笔买入，就算一次完整交易。配对逻辑写得很朴素（FIFO），因为中短线
个人账户不需要复杂撮合。

刻意不做：
- 不计算「夏普比率」这类需要日频净值曲线的指标（样本周期里持仓不连续）
- 不做显著性检验（样本量普遍 < 100，p 值没有意义，会给人虚假的确定感）
  用 Bootstrap 置信区间代替，更诚实
"""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Sequence

# 没记录止损、规则也没配默认止损时的兜底风险基准（百分比）。
# 这是一个明确的近似值，统计时会计入 risk_source='default' 的占比并如实展示。
DEFAULT_RISK_PCT = 3.0


@dataclass
class RoundTrip:
    """一次完整的买卖配对。"""

    symbol: str
    open_date: str
    close_date: str
    buy_price: float
    sell_price: float
    qty: int
    fee: float
    rule_id: int | None
    followed: bool
    mood: str
    # 开仓时设定的止损价。为空表示当时没设止损，R 倍数退回默认基准
    stop_price: float | None = None
    # 规则上配置的止损百分比，作为没填止损价时的兜底
    rule_stop_pct: float | None = None

    @property
    def pnl(self) -> float:
        return (self.sell_price - self.buy_price) * self.qty - self.fee

    @property
    def pnl_pct(self) -> float:
        cost = self.buy_price * self.qty
        return round(self.pnl / cost * 100, 2) if cost else 0.0

    @property
    def held_days(self) -> int:
        from datetime import date

        try:
            d1 = date.fromisoformat(self.open_date[:10])
            d2 = date.fromisoformat(self.close_date[:10])
            return (d2 - d1).days
        except (ValueError, TypeError):
            return 0

    @property
    def risk_pct(self) -> float:
        """开仓时承受的单股风险百分比，即 1R 对应的跌幅。

        优先用真实止损价算，其次用规则默认止损，都没有则退回 3%。
        这个值越低，说明止损设得越紧，同样的盈利折算出的 R 越大。
        """
        if self.stop_price and self.buy_price and self.stop_price < self.buy_price:
            pct = (self.buy_price - self.stop_price) / self.buy_price * 100
            # 止损距离异常小（<0.3%）通常是把止损填错了，退回兜底值以免 R 被放大到离谱
            if 0.3 <= pct <= 50:
                return round(pct, 2)
        if self.rule_stop_pct and 0.3 <= self.rule_stop_pct <= 50:
            return round(self.rule_stop_pct, 2)
        return DEFAULT_RISK_PCT

    @property
    def r_multiple(self) -> float:
        """R 倍数 = 收益率 / 单股风险百分比，用于跨股票、跨仓位比较。

        这正是「盈亏比」得以成立的基础：赚 10% 但止损 2%（5R）
        远比赚 10% 但止损 20%（0.5R）优秀。
        """
        return round(self.pnl_pct / self.risk_pct, 2)

    @property
    def risk_source(self) -> str:
        """风险基准的来源，用于在界面上如实标注统计口径。"""
        if self.stop_price and self.buy_price and self.stop_price < self.buy_price:
            pct = (self.buy_price - self.stop_price) / self.buy_price * 100
            if 0.3 <= pct <= 50:
                return "actual"
        if self.rule_stop_pct and 0.3 <= self.rule_stop_pct <= 50:
            return "rule"
        return "default"


@dataclass
class RuleStats:
    rule_id: int | None
    rule_name: str
    version: str
    sample: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_held_days: float = 0.0
    followed_rate: float = 0.0
    kind: str = "entry"
    verdict: str = "样本不足"
    trips: list[RoundTrip] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return round(self.wins / self.sample * 100, 1) if self.sample else 0.0


def build_round_trips(
    trades: Sequence[sqlite3.Row], rules: dict[int, sqlite3.Row] | None = None
) -> list[RoundTrip]:
    """FIFO 配对买卖记录为完整交易。

    规则：按时间顺序，卖出依次消耗该股票最早的买入。加仓、减仓都能处理。
    rules 用于取出规则上的默认止损百分比，供没设具体止损价的交易兜底。
    """
    rules = rules or {}
    lots: dict[str, deque] = defaultdict(deque)
    trips: list[RoundTrip] = []

    def _row_get(row: Any, key: str, default: Any = None) -> Any:
        """兼容 sqlite3.Row 与普通 dict，缺列时返回默认值。"""
        try:
            val = row[key]
        except (IndexError, KeyError):
            return default
        return default if val is None else val

    ordered = sorted(trades, key=lambda t: (t["trade_date"], t["id"]))
    for t in ordered:
        sym = t["symbol"]
        if t["side"] == "buy":
            lots[sym].append(
                {
                    "date": t["trade_date"],
                    "price": t["price"],
                    "qty": t["qty"],
                    "fee": t["fee"] or 0.0,
                    "rule_id": t["rule_id"],
                    "followed": bool(t["followed"]),
                    "mood": t["mood"] or "",
                    "stop_price": _row_get(t, "stop_price"),
                }
            )
            continue

        remaining = t["qty"]
        sell_fee_per_share = (t["fee"] or 0.0) / t["qty"] if t["qty"] else 0.0
        while remaining > 0 and lots[sym]:
            lot = lots[sym][0]
            matched = min(remaining, lot["qty"])
            ratio = matched / lot["qty"] if lot["qty"] else 1.0
            rule = rules.get(lot["rule_id"]) if lot["rule_id"] else None
            rule_stop = _row_get(rule, "stop_loss_pct") if rule is not None else None
            trips.append(
                RoundTrip(
                    symbol=sym,
                    open_date=lot["date"],
                    close_date=t["trade_date"],
                    buy_price=lot["price"],
                    sell_price=t["price"],
                    qty=matched,
                    fee=lot["fee"] * ratio + sell_fee_per_share * matched,
                    rule_id=lot["rule_id"],
                    followed=lot["followed"],
                    mood=lot["mood"],
                    stop_price=lot["stop_price"],
                    rule_stop_pct=rule_stop,
                )
            )
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 0:
                lots[sym].popleft()
    return trips


def bootstrap_ci(values: Sequence[float], iters: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """对均值做 Bootstrap 置信区间。

    样本少时比 t 检验诚实：它会明确告诉你「区间宽得离谱，别下结论」。
    """
    if len(values) < 3:
        return 0.0, 0.0
    rng = random.Random(42)  # 固定种子，保证同一份数据每次刷新结果一致
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(iters * alpha / 2)]
    hi = means[int(iters * (1 - alpha / 2))]
    return round(lo, 2), round(hi, 2)


def _max_drawdown(returns: Sequence[float]) -> float:
    """按交易顺序累计收益的最大回撤（百分比）。"""
    equity, peak, mdd = 0.0, 0.0, 0.0
    for r in returns:
        equity += r
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)
    return round(mdd, 2)


def _exit_verdict(sample: int, pcts: Sequence[float]) -> str:
    """离场规则的判定口径：看单笔最大亏损是否被截断，而不是看胜率。

    止损规则本来就该亏，关键是「亏得够不够小」。
    """
    if sample < 10:
        return f"样本不足（{sample}/10）"
    worst = min(pcts) if pcts else 0.0
    avg_loss = sum(p for p in pcts if p < 0) / max(1, sum(1 for p in pcts if p < 0))
    if worst > -8:
        return f"止损有效，最大亏损 {worst:.1f}%"
    if worst > -12:
        return f"偶有失控，最大亏损 {worst:.1f}%"
    return f"止损失效，最大亏损 {worst:.1f}%"


def _verdict(sample: int, expectancy: float, ci_low: float, ci_high: float) -> str:
    """给出一个不说漂亮话的结论。"""
    if sample < 30:
        return f"样本不足（{sample}/30）"
    if ci_low > 0:
        return "期望为正且下界>0，可继续跑"
    if ci_high < 0:
        return "期望为负，规则需重写"
    return "区间跨零，尚无定论"


def compute_rule_stats(
    trips: Sequence[RoundTrip], rules: dict[int, sqlite3.Row]
) -> list[RuleStats]:
    """按规则分组统计，按样本量降序。"""
    groups: dict[int | None, list[RoundTrip]] = defaultdict(list)
    for t in trips:
        groups[t.rule_id].append(t)

    out: list[RuleStats] = []
    for rule_id, group in groups.items():
        rule = rules.get(rule_id) if rule_id is not None else None
        stats = RuleStats(
            rule_id=rule_id,
            rule_name=rule["name"] if rule else "计划外操作",
            version=rule["version"] if rule else "-",
            sample=len(group),
        )
        pcts = [t.pnl_pct for t in group]
        stats.wins = sum(1 for p in pcts if p > 0)
        stats.losses = sum(1 for p in pcts if p < 0)
        stats.total_pnl = round(sum(t.pnl for t in group), 2)
        stats.avg_pnl_pct = round(sum(pcts) / len(pcts), 2)

        win_pcts = [p for p in pcts if p > 0]
        loss_pcts = [p for p in pcts if p < 0]
        stats.avg_win_pct = round(sum(win_pcts) / len(win_pcts), 2) if win_pcts else 0.0
        stats.avg_loss_pct = round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else 0.0

        gross_win = sum(t.pnl for t in group if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in group if t.pnl < 0))
        stats.profit_factor = round(gross_win / gross_loss, 2) if gross_loss else 0.0

        rs = [t.r_multiple for t in group]
        stats.expectancy_r = round(sum(rs) / len(rs), 2)
        stats.ci_low, stats.ci_high = bootstrap_ci(rs)
        stats.max_drawdown_pct = _max_drawdown(pcts)
        stats.avg_held_days = round(sum(t.held_days for t in group) / len(group), 1)
        stats.followed_rate = round(sum(1 for t in group if t.followed) / len(group) * 100, 1)
        stats.trips = group

        # 离场/风控规则的「胜率」天然接近 0——止损离场几乎必然是亏的，
        # 用入场规则的期望值标准去评判它，会得出「这条规则很烂」的荒谬结论。
        # 离场规则的价值在于「截断亏损」，所以改写判定口径。
        kind = rule["kind"] if rule else "entry"
        stats.kind = kind
        if kind in ("exit", "risk"):
            stats.verdict = _exit_verdict(stats.sample, pcts)
        else:
            stats.verdict = _verdict(stats.sample, stats.expectancy_r, stats.ci_low, stats.ci_high)
        out.append(stats)

    out.sort(key=lambda s: (-s.sample, s.rule_name))
    return out


def execution_gap(trips: Sequence[RoundTrip]) -> dict[str, Any]:
    """执行偏差：照规则做 vs 没照做，差多少。

    这是本系统最有价值的一张表。多数人的问题不是规则差，
    而是「该做的时候没做，不该做的时候手痒」。
    """
    followed = [t for t in trips if t.followed]
    deviated = [t for t in trips if not t.followed]

    def agg(group: list[RoundTrip]) -> dict[str, Any]:
        if not group:
            return {"sample": 0, "avg_pct": 0.0, "win_rate": 0.0, "total_pnl": 0.0}
        pcts = [t.pnl_pct for t in group]
        return {
            "sample": len(group),
            "avg_pct": round(sum(pcts) / len(pcts), 2),
            "win_rate": round(sum(1 for p in pcts if p > 0) / len(pcts) * 100, 1),
            "total_pnl": round(sum(t.pnl for t in group), 2),
        }

    a, b = agg(followed), agg(deviated)
    gap = round(a["avg_pct"] - b["avg_pct"], 2) if a["sample"] and b["sample"] else 0.0
    return {"followed": a, "deviated": b, "gap": gap}


def mood_breakdown(trips: Sequence[RoundTrip]) -> list[dict[str, Any]]:
    """情绪画像：哪种状态下的操作最赔钱。"""
    groups: dict[str, list[RoundTrip]] = defaultdict(list)
    for t in trips:
        groups[t.mood.strip() or "未记录"].append(t)

    rows = []
    for mood, group in groups.items():
        pcts = [t.pnl_pct for t in group]
        rows.append(
            {
                "mood": mood,
                "sample": len(group),
                "avg_pct": round(sum(pcts) / len(pcts), 2),
                "win_rate": round(sum(1 for p in pcts if p > 0) / len(pcts) * 100, 1),
                "total_pnl": round(sum(t.pnl for t in group), 2),
            }
        )
    rows.sort(key=lambda r: r["avg_pct"])
    return rows


def risk_coverage(trips: Sequence[RoundTrip]) -> dict[str, Any]:
    """R 倍数的口径透明化：多少笔用了真实止损，多少笔只能靠兜底。

    如果大部分交易都落在 default 上，说明期望值 R 这个数字不可全信，
    界面上必须让用户看到这一点，而不是藏起来。
    """
    counts = {"actual": 0, "rule": 0, "default": 0}
    for t in trips:
        counts[t.risk_source] = counts.get(t.risk_source, 0) + 1
    total = len(trips)
    return {
        "actual": counts["actual"],
        "rule": counts["rule"],
        "default": counts["default"],
        "total": total,
        "actual_rate": round(counts["actual"] / total * 100, 1) if total else 0.0,
        "default_rate": round(counts["default"] / total * 100, 1) if total else 0.0,
    }


def overview(trips: Sequence[RoundTrip]) -> dict[str, Any]:
    """总览：写在首页最上方的那几个数字。"""
    if not trips:
        return {
            "round_trips": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_pct": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "avg_held_days": 0.0,
            "followed_rate": 0.0,
            "avg_risk_pct": 0.0,
        }
    pcts = [t.pnl_pct for t in trips]
    rs = [t.r_multiple for t in trips]
    gross_win = sum(t.pnl for t in trips if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trips if t.pnl < 0))
    return {
        "round_trips": len(trips),
        "total_pnl": round(sum(t.pnl for t in trips), 2),
        "win_rate": round(sum(1 for p in pcts if p > 0) / len(pcts) * 100, 1),
        "avg_pct": round(sum(pcts) / len(pcts), 2),
        "expectancy_r": round(sum(rs) / len(rs), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else 0.0,
        "avg_held_days": round(sum(t.held_days for t in trips) / len(trips), 1),
        "followed_rate": round(sum(1 for t in trips if t.followed) / len(trips) * 100, 1),
        "avg_risk_pct": round(sum(t.risk_pct for t in trips) / len(trips), 2),
    }
