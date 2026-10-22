"""python -m trade_journal.reset —— 清空全部数据，回到干净状态。

为什么需要这个命令：演示数据和真实交易混在一起，统计就失去意义。
手动删 data/journal.db 容易删错位置（尤其是从别的目录运行时），
而且带 WAL 模式时有三个文件要一起处理。这里统一收口。

安全设计：必须显式带 --yes 才会执行，且执行前自动备份一份。
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from .db import DEFAULT_DB


def _db_files(db: Path) -> list[Path]:
    """SQLite 在 WAL 模式下会额外产生 -wal / -shm 两个文件，必须一起清掉。"""
    return [db, Path(str(db) + "-wal"), Path(str(db) + "-shm")]


def reset(db: Path | None = None, backup: bool = True) -> None:
    target = Path(db) if db else DEFAULT_DB
    files = _db_files(target)

    existing = [f for f in files if f.exists()]
    if not existing:
        print(f"没有找到数据库：{target}")
        print("直接启动程序即可，会自动创建空库。")
        return

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = target.with_name(f"{target.stem}.backup_{stamp}{target.suffix}")
        shutil.copy2(target, bak)
        print(f"已备份到：{bak}")

    for f in existing:
        f.unlink()
    print(f"已清空 {len(existing)} 个数据库文件。")
    print("下次启动程序时会自动创建空库，从零开始记录。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="清空交易日志的全部数据（会先自动备份）",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="确认执行。不加这个参数只会显示将要做什么，不会真的删除",
    )
    parser.add_argument("--no-backup", action="store_true", help="跳过备份（不建议）")
    args = parser.parse_args()

    target = DEFAULT_DB
    if not args.yes:
        files = [f for f in _db_files(target) if f.exists()]
        if not files:
            print(f"没有找到数据库：{target}")
            return
        print("将要删除以下文件：")
        for f in files:
            print(f"  {f}")
        print(f"\n数据库大小：{target.stat().st_size / 1024:.1f} KB")
        print("\n这是不可逆操作。确认无误请加 --yes 重新执行：")
        print("  python -m trade_journal.reset --yes")
        return

    reset(backup=not args.no_backup)


if __name__ == "__main__":
    main()
