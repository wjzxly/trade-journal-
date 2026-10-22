"""统计层测试。核心是 FIFO 配对——配错了所有指标都是错的。"""

from __future__ import annotations

from trade_journal.stats import (
    bootstrap_ci,
    build_round_trips,
    compute_rule_stats,
    execution_gap,
    mood_breakdown,
    overview,
)


def trade(**kw):
    """构造一条交易记录，字段对齐 sqlite3.Row 的访问方式。"""
    base = {
        "id": 1,
        "symbol": "600522",
        "side": "buy",
        "price": 10.0,
        "qty": 1000,
        "trade_date": "2026-01-05",
        "rule_id": 1,
        "followed": 1,
        "mood": "平静",
        "fee": 0.0,
    }
    base.update(kw)
    return base


class _Row(dict):
    """模拟 sqlite3.Row：既支持 [] 也支持属性名取值。"""

    def __getitem__(self, k):
        return super().__getitem__(k)


def rule_row(rid=1, name="测试规则", version="v1", kind="entry"):
    """构造规则行。kind 决定统计口径：入场看期望值，离场看止损是否有效。"""
    return _Row(id=rid, name=name, version=version, kind=kind)


class TestFIFO:
    def test_simple_buy_sell(self):
        trades = [_Row(trade(id=1)), _Row(trade(id=2, side="sell", price=11.0, trade_date="2026-01-10"))]
        trips = build_round_trips(trades)
        assert len(trips) == 1
        t = trips[0]
        assert t.buy_price == 10.0
        assert t.sell_price == 11.0
        assert t.pnl == 1000.0
        assert t.pnl_pct == 10.0
        assert t.held_days == 5

    def test_loss_trip(self):
        trades = [_Row(trade(id=1)), _Row(trade(id=2, side="sell", price=9.5, trade_date="2026-01-08"))]
        t = build_round_trips(trades)[0]
        assert t.pnl == -500.0
        assert t.pnl_pct == -5.0

    def test_partial_sell(self):
        trades = [
            _Row(trade(id=1, qty=1000)),
            _Row(trade(id=2, side="sell", qty=400, price=11.0, trade_date="2026-01-08")),
        ]
        trips = build_round_trips(trades)
        assert len(trips) == 1
        assert trips[0].qty == 400
        assert trips[0].pnl == 400.0

    def test_two_buys_one_sell_spans_lots(self):
        trades = [
            _Row(trade(id=1, qty=500, price=10.0)),
            _Row(trade(id=2, qty=500, price=10.5, trade_date="2026-01-06")),
            _Row(trade(id=3, side="sell", qty=800, price=11.0, trade_date="2026-01-10")),
        ]
        trips = build_round_trips(trades)
        # 500 股来自第一笔，300 股来自第二笔 → 两段独立盈亏
        assert len(trips) == 2
        assert trips[0].qty == 500
        assert trips[0].buy_price == 10.0
        assert trips[1].qty == 300
        assert trips[1].buy_price == 10.5

    def test_sell_without_buy_ignored(self):
        trades = [_Row(trade(id=1, side="sell", price=11.0))]
        assert build_round_trips(trades) == []

    def test_buy_never_sold_not_counted(self):
        trades = [_Row(trade(id=1))]
        assert build_round_trips(trades) == []

    def test_multiple_symbols_isolated(self):
        trades = [
            _Row(trade(id=1, symbol="AAA")),
            _Row(trade(id=2, symbol="BBB")),
            _Row(trade(id=3, symbol="AAA", side="sell", price=11.0, trade_date="2026-01-10")),
        ]
        trips = build_round_trips(trades)
        assert len(trips) == 1
        assert trips[0].symbol == "AAA"

    def test_fee_deducted(self):
        trades = [
            _Row(trade(id=1, fee=5.0)),
            _Row(trade(id=2, side="sell", price=11.0, trade_date="2026-01-10", fee=6.0)),
        ]
        t = build_round_trips(trades)[0]
        assert t.pnl == 1000.0 - 11.0


class TestAggregation:
    def _trips(self, pnls, followed=True, rule_id=1, mood="平静"):
        trades = []
        tid = 1
        for i, pct in enumerate(pnls):
            buy, sell = 10.0, 10.0 * (1 + pct / 100)
            trades.append(_Row(trade(id=tid, price=buy, trade_date=f"2026-01-{i + 1:02d}",
                                     followed=int(followed), rule_id=rule_id, mood=mood)))
            trades.append(_Row(trade(id=tid + 1, side="sell", price=sell,
                                     trade_date=f"2026-01-{i + 1:02d}",
                                     followed=int(followed), rule_id=rule_id, mood=mood)))
            tid += 2
        return build_round_trips(trades)

    def test_win_rate(self):
        trips = self._trips([5, -3, 8, -2, 1])
        stats = compute_rule_stats(trips, {1: rule_row(name="测试规则")})[0]
        assert stats.sample == 5
        assert stats.wins == 3
        assert stats.win_rate == 60.0

    def test_expectancy_sign(self):
        good = self._trips([5, -2, 6, -1, 4, 3])
        stats = compute_rule_stats(good, {1: rule_row(name="好规则")})[0]
        assert stats.expectancy_r > 0

        bad = self._trips([-5, -6, -2, -4, 1, -3])
        stats2 = compute_rule_stats(bad, {1: rule_row(name="坏规则")})[0]
        assert stats2.expectancy_r < 0

    def test_profit_factor_no_losses(self):
        trips = self._trips([5, 6, 7])
        stats = compute_rule_stats(trips, {1: rule_row(name="全胜")})[0]
        assert stats.profit_factor == 0.0  # 无亏损时不下定义，返回 0 而非 inf

    def test_verdict_small_sample(self):
        trips = self._trips([5, 6])
        stats = compute_rule_stats(trips, {1: rule_row(name="样本少")})[0]
        assert "样本不足" in stats.verdict

    def test_verdict_enough_sample(self):
        trips = self._trips([5, -2] * 15)  # 30 笔
        stats = compute_rule_stats(trips, {1: rule_row(name="够样本")})[0]
        assert stats.sample == 30
        assert "样本不足" not in stats.verdict

    def test_unplanned_trades_grouped(self):
        trips = self._trips([5], rule_id=None)
        stats = compute_rule_stats(trips, {})[0]
        assert stats.rule_name == "计划外操作"

    def test_exit_rule_uses_stoploss_verdict(self):
        """离场规则的胜率天然接近 0（止损必亏），不能用入场标准评判。"""
        trips = self._trips([-3, -4, -2, -5, -3.5, -4.2, -2.8, -3.1, -4.5, -3.3])
        stats = compute_rule_stats(trips, {1: rule_row(kind="exit")})[0]
        assert stats.kind == "exit"
        assert stats.win_rate == 0.0                     # 全亏
        assert stats.verdict.startswith("止损有效")        # 但结论是正面的
        assert "样本不足" not in stats.verdict             # 离场规则阈值是 10 不是 30

    def test_exit_rule_detects_failed_stoploss(self):
        trips = self._trips([-15, -18, -12, -20, -14, -16, -13, -17, -19, -15])
        stats = compute_rule_stats(trips, {1: rule_row(kind="exit")})[0]
        assert "止损失效" in stats.verdict

    def test_risk_rule_treated_like_exit(self):
        trips = self._trips([-4] * 12)
        stats = compute_rule_stats(trips, {1: rule_row(kind="risk")})[0]
        assert stats.verdict.startswith("止损有效")

    def test_entry_rule_still_uses_expectancy(self):
        trips = self._trips([5, -3, 8, -2, 1])
        stats = compute_rule_stats(trips, {1: rule_row(kind="entry")})[0]
        assert "样本不足" in stats.verdict  # 入场规则阈值仍是 30


class TestExecutionGap:
    def test_gap_positive_when_following_is_better(self):
        good = []
        for i in range(3):
            good.append(_Row(trade(id=i * 2 + 1, price=10.0, trade_date=f"2026-02-{i + 1:02d}", followed=1)))
            good.append(_Row(trade(id=i * 2 + 2, side="sell", price=10.5,
                                   trade_date=f"2026-02-{i + 1:02d}", followed=1)))
        bad = []
        for i in range(3):
            bad.append(_Row(trade(id=100 + i * 2, symbol="000001", price=10.0,
                                  trade_date=f"2026-02-{i + 1:02d}", followed=0)))
            bad.append(_Row(trade(id=101 + i * 2, symbol="000001", side="sell", price=9.5,
                                  trade_date=f"2026-02-{i + 1:02d}", followed=0)))

        trips = build_round_trips(good + bad)
        gap = execution_gap(trips)
        assert gap["followed"]["sample"] == 3
        assert gap["deviated"]["sample"] == 3
        assert gap["gap"] > 0

    def test_empty(self):
        gap = execution_gap([])
        assert gap["gap"] == 0.0
        assert gap["followed"]["sample"] == 0


class TestRiskBasis:
    """R 倍数的分母来源：真实止损 > 规则默认 > 3% 兜底。"""

    def _one(self, buy=10.0, sell=11.0, stop=None, rule_stop=None):
        trades = [
            _Row(trade(id=1, price=buy, stop_price=stop)),
            _Row(trade(id=2, side="sell", price=sell, trade_date="2026-01-10")),
        ]
        rules = {1: _Row(id=1, name="R", version="v1", kind="entry", stop_loss_pct=rule_stop)}
        return build_round_trips(trades, rules)[0]

    def test_actual_stop_wins(self):
        # 买 10，止损 9 → 1R = 10%；赚 1 → 1.0R
        t = self._one(buy=10.0, sell=11.0, stop=9.0, rule_stop=4.0)
        assert t.risk_pct == 10.0
        assert t.risk_source == "actual"
        assert t.r_multiple == 1.0

    def test_tighter_stop_gives_higher_r(self):
        """同样赚 10%，止损越紧 R 越高——这正是不能设太宽的原因。"""
        tight = self._one(buy=10.0, sell=11.0, stop=9.8)   # 2% 风险
        wide = self._one(buy=10.0, sell=11.0, stop=8.0)    # 20% 风险
        assert tight.r_multiple > wide.r_multiple
        assert tight.r_multiple == 5.0
        assert wide.r_multiple == 0.5

    def test_rule_default_used_when_no_stop(self):
        t = self._one(buy=10.0, sell=10.6, stop=None, rule_stop=4.0)
        assert t.risk_pct == 4.0
        assert t.risk_source == "rule"
        assert t.r_multiple == 1.5

    def test_fallback_when_nothing_set(self):
        t = self._one(buy=10.0, sell=10.3, stop=None, rule_stop=None)
        assert t.risk_pct == 3.0
        assert t.risk_source == "default"
        assert t.r_multiple == 1.0

    def test_unplanned_trade_falls_back(self):
        """计划外操作没有规则，只能靠兜底。"""
        trades = [
            _Row(trade(id=1, price=10.0, rule_id=None)),
            _Row(trade(id=2, side="sell", price=10.5, trade_date="2026-01-10", rule_id=None)),
        ]
        t = build_round_trips(trades, {})[0]
        assert t.risk_source == "default"
        assert t.risk_pct == 3.0

    def test_stop_above_buy_ignored(self):
        """止损价填得比买入价还高，说明填反了，不能拿来当风险基准。"""
        t = self._one(buy=10.0, sell=10.5, stop=11.0, rule_stop=4.0)
        assert t.risk_source == "rule"  # 退回规则默认
        assert t.risk_pct == 4.0

    def test_absurdly_tight_stop_rejected(self):
        """止损距离 0.05% 这种多半是手误，会让 R 放大几十倍，必须过滤。"""
        t = self._one(buy=10.0, sell=10.5, stop=9.995, rule_stop=4.0)
        assert t.risk_source == "rule"

    def test_rule_stop_out_of_range_ignored(self):
        t = self._one(buy=10.0, sell=10.5, stop=None, rule_stop=80.0)
        assert t.risk_source == "default"

    def test_build_without_rules_argument(self):
        """rules 参数可选，老调用方式不能崩。"""
        trades = [
            _Row(trade(id=1)),
            _Row(trade(id=2, side="sell", price=11.0, trade_date="2026-01-10")),
        ]
        trips = build_round_trips(trades)
        assert len(trips) == 1
        assert trips[0].risk_source == "default"

    def test_missing_stop_column_tolerated(self):
        """老数据库没有 stop_price 列时，_row_get 应退回 None 而不是抛错。"""
        class NoStopRow(dict):
            def __getitem__(self, k):
                if k == "stop_price":
                    raise IndexError("no such column")
                return super().__getitem__(k)

        trades = [
            NoStopRow(trade(id=1)),
            NoStopRow(trade(id=2, side="sell", price=11.0, trade_date="2026-01-10")),
        ]
        trips = build_round_trips(trades)
        assert trips[0].risk_source == "default"


class TestRiskCoverage:
    def test_counts_and_rates(self):
        from trade_journal.stats import RoundTrip, risk_coverage

        trips = [
            RoundTrip("A", "2026-01-01", "2026-01-05", 10, 11, 100, 0, 1, True, "", 9.0),
            RoundTrip("A", "2026-01-01", "2026-01-05", 10, 11, 100, 0, 1, True, "", 9.0),
            RoundTrip("A", "2026-01-01", "2026-01-05", 10, 11, 100, 0, 1, True, "", None, 4.0),
            RoundTrip("A", "2026-01-01", "2026-01-05", 10, 11, 100, 0, 1, True, "", None, None),
        ]
        rc = risk_coverage(trips)
        assert rc["actual"] == 2
        assert rc["rule"] == 1
        assert rc["default"] == 1
        assert rc["total"] == 4
        assert rc["actual_rate"] == 50.0
        assert rc["default_rate"] == 25.0

    def test_empty(self):
        from trade_journal.stats import risk_coverage

        rc = risk_coverage([])
        assert rc["total"] == 0
        assert rc["actual_rate"] == 0.0


class TestBootstrap:
    def test_ci_contains_mean(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5] * 5
        lo, hi = bootstrap_ci(vals)
        mean = sum(vals) / len(vals)
        assert lo <= mean <= hi

    def test_too_few_samples(self):
        assert bootstrap_ci([1.0, 2.0]) == (0.0, 0.0)

    def test_deterministic(self):
        vals = [1.0, -1.0, 2.0, -2.0, 0.5] * 10
        assert bootstrap_ci(vals) == bootstrap_ci(vals)


class TestMoodAndOverview:
    def test_mood_grouping(self):
        from trade_journal.stats import RoundTrip

        trips = [
            RoundTrip("600522", "2026-01-01", "2026-01-05", 10.0, 11.0, 100, 0, 1, True, "贪婪"),
            RoundTrip("600522", "2026-01-01", "2026-01-05", 10.0, 9.0, 100, 0, 1, True, "恐惧"),
        ]
        moods = mood_breakdown(trips)
        assert len(moods) == 2
        # 最赔钱的排最前
        assert moods[0]["mood"] == "恐惧"

    def test_empty_overview(self):
        o = overview([])
        assert o["round_trips"] == 0
        assert o["win_rate"] == 0.0
