"""FastAPI 应用：REST 接口 + 静态单页前端。

设计原则：前后端都在一个进程、一个端口里跑。
个人工具不需要构建链，不需要 Node，双击 `python -m trade_journal` 就能用。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, repository as repo
from .db import DEFAULT_DB, init_db
from .rule_engine import FIELDS, RuleSyntaxError, check, validate
from .stats import (
    build_round_trips,
    compute_rule_stats,
    execution_gap,
    mood_breakdown,
    overview,
    risk_coverage,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Trade Journal",
    description="个人交易经验终端：规则 → 信号 → 操作 → 复盘",
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------- models

class RuleIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    condition: str = Field(..., min_length=1)
    kind: str = "entry"
    hypothesis: str = ""
    notes: str = ""
    version: str = "v1"
    # 默认止损幅度（%），仅用于 R 倍数兜底；每笔交易可单独填真实止损价
    stop_loss_pct: float | None = Field(None, gt=0.3, le=50)


class RulePatch(BaseModel):
    name: str | None = None
    condition: str | None = None
    kind: str | None = None
    hypothesis: str | None = None
    notes: str | None = None
    active: int | None = None
    version: str | None = None
    stop_loss_pct: float | None = Field(None, gt=0.3, le=50)


class SignalIn(BaseModel):
    rule_id: int
    symbol: str = Field(..., min_length=1, max_length=16)
    trade_date: str | None = None
    price: float | None = None
    acted: bool | None = None
    skip_reason: str = ""


class TradeIn(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    side: str = Field(..., pattern="^(buy|sell)$")
    price: float = Field(..., gt=0)
    qty: int = Field(..., gt=0)
    trade_date: str | None = None
    signal_id: int | None = None
    rule_id: int | None = None
    followed: bool = True
    deviation: str = ""
    mood: str = ""
    position_pct: float = 0.0
    fee: float = 0.0
    notes: str = ""
    # 开仓时设定的止损价，必须低于买入价才合理
    stop_price: float | None = Field(None, gt=0)


class ReviewIn(BaseModel):
    trade_id: int | None = None
    symbol: str = ""
    outcome: str = Field(..., pattern="^(win|loss|flat)$")
    pnl: float = 0.0
    pnl_pct: float = 0.0
    attribution: str = "rule"
    what_went_right: str = ""
    what_to_fix: str = ""


class ScanIn(BaseModel):
    rows: list[dict[str, Any]]
    rule_ids: list[int] | None = None
    trade_date: str | None = None


# ---------------------------------------------------------------- 规则

@app.get("/api/rules")
def api_rules(active_only: bool = False) -> dict[str, Any]:
    rows = repo.list_rules(active_only)
    return {"items": [dict(r) for r in rows]}


@app.get("/api/rules/fields")
def api_fields() -> dict[str, str]:
    return FIELDS


@app.post("/api/rules/validate")
def api_validate(payload: dict[str, str]) -> dict[str, Any]:
    ok, err = validate(payload.get("condition", ""))
    return {"ok": ok, "error": err}


@app.post("/api/rules", status_code=201)
def api_create_rule(rule: RuleIn) -> dict[str, int]:
    ok, err = validate(rule.condition)
    if not ok:
        raise HTTPException(400, f"规则条件有误：{err}")
    return {"id": repo.create_rule(
        name=rule.name,
        condition=rule.condition,
        kind=rule.kind,
        hypothesis=rule.hypothesis,
        notes=rule.notes,
        version=rule.version,
        stop_loss_pct=rule.stop_loss_pct,
    )}


@app.patch("/api/rules/{rule_id}")
def api_patch_rule(rule_id: int, patch: RulePatch) -> dict[str, bool]:
    if patch.condition is not None:
        ok, err = validate(patch.condition)
        if not ok:
            raise HTTPException(400, f"规则条件有误：{err}")
    repo.update_rule(rule_id, **patch.model_dump(exclude_none=True))
    return {"ok": True}


@app.delete("/api/rules/{rule_id}")
def api_retire_rule(rule_id: int) -> dict[str, bool]:
    repo.retire_rule(rule_id)
    return {"ok": True}


# ---------------------------------------------------------------- 信号

@app.get("/api/signals")
def api_signals(limit: int = 100, acted: str = "all") -> dict[str, Any]:
    return {"items": [dict(r) for r in repo.list_signals(limit, acted)]}


@app.post("/api/signals", status_code=201)
def api_create_signal(sig: SignalIn) -> dict[str, int]:
    from datetime import date

    try:
        sid = repo.create_signal(
            sig.rule_id, sig.symbol, sig.trade_date or date.today().isoformat(),
            sig.price, True, sig.acted, sig.skip_reason,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": sid}


@app.patch("/api/signals/{signal_id}")
def api_mark_signal(signal_id: int, payload: dict[str, Any]) -> dict[str, bool]:
    repo.mark_signal(signal_id, bool(payload.get("acted")), payload.get("skip_reason", ""))
    return {"ok": True}


@app.post("/api/signals/scan")
def api_scan(payload: ScanIn) -> dict[str, Any]:
    """批量扫描：给一批行情快照，返回哪些规则被触发。

    这是「AI 帮你盯盘」的雏形——但只做判定，不做推荐。触发不等于该买。
    """
    rules = {r["id"]: r for r in repo.list_rules(active_only=True)}
    if payload.rule_ids:
        rules = {k: v for k, v in rules.items() if k in set(payload.rule_ids)}

    hits: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in payload.rows:
        symbol = row.get("symbol") or row.get("code")
        if not symbol:
            continue
        for rid, rule in rules.items():
            try:
                if check(rule["condition"], row):
                    hits.append({
                        "rule_id": rid,
                        "rule_name": rule["name"],
                        "rule_version": rule["version"],
                        "symbol": symbol,
                        "price": row.get("close"),
                        "snapshot": row,
                    })
            except (RuleSyntaxError, TypeError) as exc:
                errors.append({"rule_id": str(rid), "error": str(exc)})
    return {"hits": hits, "errors": errors, "scanned_rows": len(payload.rows),
            "rules_evaluated": len(rules)}


# ---------------------------------------------------------------- 交易

@app.get("/api/trades")
def api_trades(limit: int = 200, symbol: str | None = None) -> dict[str, Any]:
    return {"items": [dict(r) for r in repo.list_trades(limit, symbol)]}


@app.post("/api/trades", status_code=201)
def api_create_trade(trade: TradeIn) -> dict[str, int]:
    # 止损价填反是最常见的手误，这里挡掉，否则 R 倍数会被算成负数或离谱值
    if trade.side == "buy" and trade.stop_price is not None:
        if trade.stop_price >= trade.price:
            raise HTTPException(
                400,
                f"买入止损价 {trade.stop_price} 必须低于买入价 {trade.price}，"
                f"否则不是止损而是止盈",
            )
    return {"id": repo.create_trade(**trade.model_dump())}


@app.delete("/api/trades/{trade_id}")
def api_delete_trade(trade_id: int) -> dict[str, bool]:
    repo.delete_trade(trade_id)
    return {"ok": True}


# ---------------------------------------------------------------- 复盘

@app.get("/api/reviews")
def api_reviews(limit: int = 100) -> dict[str, Any]:
    return {"items": [dict(r) for r in repo.list_reviews(limit)]}


@app.post("/api/reviews", status_code=201)
def api_create_review(review: ReviewIn) -> dict[str, int]:
    return {"id": repo.create_review(**review.model_dump())}


# ---------------------------------------------------------------- 统计

@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    rules = repo.rules_map()
    trips = build_round_trips(repo.all_trades(), rules)

    by_rule = []
    for s in compute_rule_stats(trips, rules):
        by_rule.append({
            "rule_id": s.rule_id,
            "rule_name": s.rule_name,
            "version": s.version,
            "sample": s.sample,
            "win_rate": s.win_rate,
            "wins": s.wins,
            "losses": s.losses,
            "total_pnl": s.total_pnl,
            "avg_pnl_pct": s.avg_pnl_pct,
            "avg_win_pct": s.avg_win_pct,
            "avg_loss_pct": s.avg_loss_pct,
            "expectancy_r": s.expectancy_r,
            "profit_factor": s.profit_factor,
            "ci": [s.ci_low, s.ci_high],
            "max_drawdown_pct": s.max_drawdown_pct,
            "avg_held_days": s.avg_held_days,
            "followed_rate": s.followed_rate,
            "kind": s.kind,
            "verdict": s.verdict,
        })

    return {
        "overview": overview(trips),
        "by_rule": by_rule,
        "execution_gap": execution_gap(trips),
        "moods": mood_breakdown(trips),
        "attribution": repo.attribution_breakdown(),
        "streak": repo.streak_days(),
        "risk_coverage": risk_coverage(trips),
        "round_trips": [
            {
                "symbol": t.symbol,
                "open_date": t.open_date,
                "close_date": t.close_date,
                "qty": t.qty,
                "pnl": round(t.pnl, 2),
                "pnl_pct": t.pnl_pct,
                "held_days": t.held_days,
                "followed": t.followed,
                "mood": t.mood,
                "risk_pct": t.risk_pct,
                "r_multiple": t.r_multiple,
                "risk_source": t.risk_source,
            }
            for t in sorted(trips, key=lambda x: x.close_date, reverse=True)[:50]
        ],
    }


# ---------------------------------------------------------------- 页面

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "db": str(DEFAULT_DB)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
