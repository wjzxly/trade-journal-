"""规则引擎测试。重点不是覆盖率，是把边界情况钉死：
字段拼错、缺数据、跨式引用、eval 注入尝试。"""

from __future__ import annotations

import pytest

from trade_journal.rule_engine import RuleSyntaxError, check, parse, validate

ROW = {
    "close": 12.40,
    "ma20": 11.80,
    "ma60": 10.50,
    "vol_ratio": 1.85,
    "macd_hist": 0.06,
    "pct_chg": 3.2,
    "close_prev": 11.72,
    "ma20_prev": 11.75,
    "held_days": 5,
    "pnl_pct": -2.4,
}


class TestBasicComparison:
    def test_simple_gt(self):
        assert check("close > ma20", ROW) is True

    def test_simple_lt(self):
        assert check("close < ma20", ROW) is False

    def test_gte_boundary(self):
        assert check("close >= 12.4", ROW) is True
        assert check("close > 12.4", ROW) is False

    def test_negative_number(self):
        assert check("pnl_pct < -2", ROW) is True
        assert check("pnl_pct > -2", ROW) is False

    def test_integer_parsed_as_int(self):
        # 12.0 会被转成 int(12)，不影响比较但便于调试输出
        assert check("close > 12", ROW) is True


class TestLogicalOperators:
    def test_and_all_true(self):
        assert check("close > ma20 and vol_ratio > 1.5", ROW) is True

    def test_and_one_false(self):
        assert check("close > ma20 and vol_ratio > 3", ROW) is False

    def test_or(self):
        assert check("close < ma20 or vol_ratio > 1.5", ROW) is True

    def test_or_both_false(self):
        assert check("close < ma20 or vol_ratio > 5", ROW) is False

    def test_parentheses_override_precedence(self):
        # and 优先于 or，加括号改变结果
        assert check("close < ma20 or vol_ratio > 1.5 and pct_chg > 5", ROW) is False
        assert check("(close < ma20 or vol_ratio > 1.5) and pct_chg > 5", ROW) is False
        assert check("close < ma20 or (vol_ratio > 1.5 and pct_chg > 1)", ROW) is True

    def test_nested_parentheses(self):
        assert check("(close > ma20 and (vol_ratio > 1.5 or pct_chg > 2))", ROW) is True


class TestPrevReference:
    def test_cross_above(self):
        # 今天站上 20 日线，昨天还在下面 → 金叉
        assert check("close > ma20 and close_prev <= ma20_prev", ROW) is True

    def test_prev_keyword_syntax(self):
        ast = parse("close > ma20 and close@prev <= ma20@prev")
        assert check("close > ma20 and close@prev <= ma20@prev", ROW) is True

    def test_missing_prev_field_returns_false(self):
        row = {"close": 12.4, "ma20": 11.8}
        assert check("close@prev > ma20@prev", row) is False


class TestErrorHandling:
    def test_unknown_field(self):
        ok, err = validate("clsoe > ma20")
        assert not ok
        assert "close" in err  # 给出拼写建议

    def test_empty_condition(self):
        ok, err = validate("   ")
        assert not ok
        assert "空" in err

    def test_unbalanced_paren(self):
        ok, err = validate("(close > ma20")
        assert not ok
        assert "右括号" in err

    def test_missing_value(self):
        ok, err = validate("close >")
        assert not ok

    def test_missing_operator(self):
        ok, err = validate("close ma20")
        assert not ok

    def test_unknown_operator_word(self):
        with pytest.raises(RuleSyntaxError):
            parse("close cross_up ma20")

    def test_eval_injection_rejected(self):
        # 关键安全测试：任何 Python 代码都必须被拒
        for bad in [
            "__import__('os').system('echo x')",
            "close > __import__('os')",
            "close > 1; import os",
            "close > 1 # comment",
        ]:
            ok, _ = validate(bad)
            assert not ok, f"不该通过：{bad}"

    def test_missing_field_value_returns_false(self):
        # 扫全市场时缺数据是常态，当条件不成立而不是崩溃
        row = {"close": 12.4}
        assert check("ma20 > 10", row) is False

    def test_field_vs_field_allowed(self):
        # close > ma20 是最常用的写法，右操作数可以是字段
        assert check("close > ma20", ROW) is True
        assert check("ma20 > close", ROW) is False

    def test_unknown_right_field_rejected(self):
        # 右边拼错的字段名必须报错，避免静默失效
        ok, err = validate("close > ma200")
        assert not ok
        assert "未知字段" in err

    def test_number_required_when_not_field(self):
        # 纯文本不是合法操作数
        ok, err = validate("close > abc")
        assert not ok


class TestCaching:
    def test_same_condition_same_ast(self):
        assert parse("close > ma20") is parse("close > ma20")

    def test_different_conditions_differ(self):
        assert parse("close > ma20") is not parse("close < ma20")
