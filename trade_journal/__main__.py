"""python -m trade_journal —— 启动本地服务。"""

from __future__ import annotations

import argparse
import webbrowser
from threading import Timer

from .db import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="个人交易经验终端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    init_db()

    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    import uvicorn

    uvicorn.run(
        "trade_journal.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
