"""规则引擎：把交易经验写成机器可判定的条件表达式。

设计取舍
========
不用自然语言，也不用 Python 的 eval。用一套**极小的表达式子集**，自己解析。
理由：eval 是注入漏洞，而自然语言不可判定——两者都违背「可验证」这个前提。

语法
----
    expr    := term (('and' | 'or') term)*
    term    := '(' expr ')' | comparison
    comparison := field op operand
    op      := '>' | '>=' | '<' | '<=' | '=' | '!=' | 'cross_above' | 'cross_below'
    operand := number | field        # ← 右边可以是数字，也可以是另一个字段
    field   := <白名单字段>[ '@prev' ]

比较的两个操作数都可以带 @prev 引用前一根 K 线，这是写「上穿/下穿」的方式。

可用字段（对应你每日录入的一行行情快照）
----------------------------------------
    close       收盘价
    pct_chg     当日涨跌幅 %
    ma5/ma10/ma20/ma60   均线值（录入时算好或直接填）
    vol_ratio   量比（当日量 / 5日均量）
    turnover    换手率 %
    rsi14       RSI
    macd_hist   MACD 柱值（>0 视为多头）
    high/low/open/prev_close  参考用
    amplitude   振幅 %
    above_ma20  是否站上20日线（1/0，可直接用 close > ma20 代替）
    held_days   持仓天数（离场规则用）
    pnl_pct     浮动盈亏 %（离场规则用）

cross_above / cross_below 需要「上一根」数据。两种等价写法：
    close > ma20 and close@prev <= ma20@prev      # 推荐
    close > ma20 and close_prev <= ma20_prev      # 兼容行情导出的字段名

示例
----
    "close > ma20 and vol_ratio > 1.5 and macd_hist > 0"
    "pct_chg < -4 and held_days >= 3 and close < ma10"
    "rsi14 < 30 and close > ma60"

刻意保留的局限：不支持加减乘除、不支持跨股票比较、不支持时间函数。
如果一条规则写不出来，说明它还不够明确——这是特性，不是缺陷。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "RuleSyntaxError",
    "parse",
    "evaluate",
    "check",
    "FIELDS",
]

# 允许出现的字段白名单。不在其中一律报错，避免拼写错误静默通过
FIELDS: dict[str, str] = {
    "close": "收盘价",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "prev_close": "昨收",
    "pct_chg": "涨跌幅%",
    "amplitude": "振幅%",
    "ma5": "MA5",
    "ma10": "MA10",
    "ma20": "MA20",
    "ma60": "MA60",
    "vol_ratio": "量比",
    "turnover": "换手率%",
    "rsi14": "RSI14",
    "macd_hist": "MACD柱",
    "held_days": "持仓天数",
    "pnl_pct": "浮动盈亏%",
    "position_pct": "当前仓位%",
}

_MISSING = object()


class RuleSyntaxError(ValueError):
    """规则写错时的异常，消息里会指出位置，便于前端提示。"""


@dataclass(frozen=True)
class Comparison:
    field: str
    op: str
    value: Any
    use_prev: bool = False
    # 右操作数也可以是字段（close > ma20 是最常用的写法），此时该标志为 True
    value_is_field: bool = False
    value_use_prev: bool = False

    def __str__(self) -> str:
        prev = "@prev" if self.use_prev else ""
        if self.value_is_field:
            vprev = "@prev" if self.value_use_prev else ""
            return f"{self.field}{prev} {self.op} {self.value}{vprev}"
        return f"{self.field}{prev} {self.op} {self.value}"


@dataclass(frozen=True)
class Binary:
    op: str          # 'and' | 'or'
    left: Any
    right: Any

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>-?\d+(?:\.\d+)?)
      | (?P<ident>[A-Za-z_][A-Za-z_0-9]*(?:@prev)?)
      | (?P<op>>=|<=|!=|>|<|=|cross_above|cross_below)
      | (?P<lparen>\()
      | (?P<rparen>\))
    )
    """,
    re.VERBOSE,
)

_CMP_OPS = {">", ">=", "<", "<=", "=", "!=", "cross_above", "cross_below"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.strip()
    if not text:
        raise RuleSyntaxError("规则为空")
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise RuleSyntaxError(f"第 {pos} 个字符附近无法识别：{text[pos:pos + 12]!r}")
        kind = m.lastgroup
        # 注意：pattern 开头的 \s* 会被算进整个 match，但不在任何命名分组里，
        # 所以必须取 m.group(kind) 而不是 m.group()，否则 token 值会带上前导空格。
        tokens.append((kind, m.group(kind)))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise RuleSyntaxError("表达式意外结束，可能缺少右括号或数值")
        self.i += 1
        return tok

    def parse(self) -> Any:
        node = self.parse_or()
        if self.peek() is not None:
            kind, val = self.peek()
            raise RuleSyntaxError(f"多余的内容：{val!r}，请检查逻辑连接词")
        return node

    def parse_or(self) -> Any:
        node = self.parse_and()
        while (tok := self.peek()) and tok[1].lower() == "or":
            self.take()
            node = Binary("or", node, self.parse_and())
        return node

    def parse_and(self) -> Any:
        node = self.parse_term()
        while (tok := self.peek()) and tok[1].lower() == "and":
            self.take()
            node = Binary("and", node, self.parse_term())
        return node

    def parse_term(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise RuleSyntaxError("表达式不完整")
        if tok[0] == "lparen":
            self.take()
            node = self.parse_or()
            nxt = self.take()
            if nxt[0] != "rparen":
                raise RuleSyntaxError(f"缺少右括号，实际遇到 {nxt[1]!r}")
            return node
        return self.parse_comparison()

    def parse_comparison(self) -> Comparison:
        kind, raw_field = self.take()
        if kind != "ident":
            raise RuleSyntaxError(f"条件左边应该是字段名，实际是 {raw_field!r}")

        use_prev = raw_field.endswith("@prev")
        field = raw_field[:-5] if use_prev else raw_field

        # 兼容 close_prev 这种写法：与 close@prev 等价，因为多数行情导出的
        # 字段名就是 xxx_prev，让用户改写反而容易出错。
        if not use_prev and field.endswith("_prev") and field[:-5] in FIELDS:
            field, use_prev = field[:-5], True

        if field.lower() in ("and", "or"):
            raise RuleSyntaxError(f"意外的逻辑词 {field!r}")
        if field not in FIELDS:
            hint = _suggest(field)
            raise RuleSyntaxError(f"未知字段 {field!r}{hint}")

        kind, raw_op = self.take()
        if kind != "op" or raw_op not in _CMP_OPS:
            raise RuleSyntaxError(f"{field} 后面应该是比较符，实际是 {raw_op!r}")

        kind, raw_val = self.take()
        value_is_field = False
        value_use_prev = False
        if kind == "number":
            value: Any = float(raw_val)
            if value.is_integer():
                value = int(value)
        elif kind == "ident":
            # 右操作数允许是另一个字段：close > ma20 是最常见的写法，
            # 也允许 close@prev <= ma20@prev 做金叉判断。
            vprev = raw_val.endswith("@prev")
            base = raw_val[:-5] if vprev else raw_val
            if not vprev and base.endswith("_prev") and base[:-5] in FIELDS:
                base, vprev = base[:-5], True
            if base.lower() in ("and", "or"):
                raise RuleSyntaxError(f"比较符 {raw_op} 后面缺少数值或字段")
            if base not in FIELDS:
                raise RuleSyntaxError(
                    f"未知字段 {base!r}{_suggest(base)}；"
                    f"如果这里想填常量，请写数字"
                )
            value, value_is_field, value_use_prev = base, True, vprev
        else:
            raise RuleSyntaxError(f"{raw_op} 后面应该是一个数值或字段，实际是 {raw_val!r}")

        return Comparison(field, raw_op, value, use_prev, value_is_field, value_use_prev)

        return Comparison(field, raw_op, value, use_prev)


def _suggest(field: str) -> str:
    """拼错时给个提示，比干巴巴报错友好。"""
    low = field.lower()
    for name in FIELDS:
        if low[:3] == name[:3] or low in name or name in low:
            return f"，是不是想写 {name!r}？"
    return f"，可用字段：{', '.join(list(FIELDS)[:8])} 等"


_cache: dict[str, Any] = {}


def parse(condition: str) -> Any:
    """解析条件字符串为 AST。带缓存，因为每日批量求值会重复解析同一条规则。"""
    if condition in _cache:
        return _cache[condition]
    tokens = _tokenize(condition)
    ast = _Parser(tokens).parse()
    _cache[condition] = ast
    return ast


def _get_field(field: str, use_prev: bool, row: Mapping[str, Any]) -> Any:
    """从行情快照取值。带 @prev 的先找 `字段_prev`，再退回 `prev` 子字典。"""
    if use_prev:
        for key in (f"{field}_prev", f"prev_{field}"):
            if key in row and row[key] is not None:
                return row[key]
        prev = row.get("prev")
        if isinstance(prev, Mapping) and prev.get(field) is not None:
            return prev[field]
        return _MISSING
    val = row.get(field)
    return _MISSING if val is None else val


def _compare(left: Any, op: str, right: Any) -> bool:
    if op in ("cross_above", "cross_below"):
        raise RuleSyntaxError(
            "cross_above/cross_below 请写成显式比较，"
            "例如 close > ma20 and close@prev <= ma20@prev"
        )
    if op == "=":
        return left == right
    if op == "!=":
        return left != right
    if isinstance(right, str):
        raise RuleSyntaxError(f"字段与文本 {right!r} 不能做大小比较，= 可以")
    if not isinstance(left, (int, float)):
        raise RuleSyntaxError(f"字段值 {left!r} 不是数字，无法比较")
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    raise RuleSyntaxError(f"不支持的比较符 {op!r}")


def evaluate(ast: Any, row: Mapping[str, Any]) -> bool:
    """在单行行情快照上求值。

    缺字段一律视为「不成立」而不是抛错——扫全市场时总会缺数据，
    把缺失当条件不满足，比整批中断更实用。
    """
    if isinstance(ast, Binary):
        if ast.op == "and":
            return evaluate(ast.left, row) and evaluate(ast.right, row)
        return evaluate(ast.left, row) or evaluate(ast.right, row)

    left = _get_field(ast.field, ast.use_prev, row)
    if left is _MISSING:
        return False

    if ast.value_is_field:
        right = _get_field(str(ast.value), ast.value_use_prev, row)
        if right is _MISSING:
            return False
    else:
        right = ast.value

    return _compare(left, ast.op, right)


def check(condition: str, row: Mapping[str, Any]) -> bool:
    """解析并求值，一步到位。"""
    return evaluate(parse(condition), row)


def validate(condition: str) -> tuple[bool, str]:
    """校验规则是否合法，返回 (是否通过, 错误信息)。供 UI 保存前调用。"""
    try:
        parse(condition)
        return True, ""
    except RuleSyntaxError as exc:
        return False, str(exc)
