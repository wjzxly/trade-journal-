"""reset 命令测试。重点是「不能误删」——这是唯一会销毁数据的操作。

reset.py 里 DEFAULT_DB 是模块级常量，测试直接用 reset(db=...) 传临时路径，
不依赖 monkeypatch 全局状态。
"""

from __future__ import annotations

import sqlite3

from trade_journal.reset import reset


def _make_db(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    return path


def test_reset_removes_db(tmp_path):
    db = _make_db(tmp_path / "journal.db")
    reset(db=db, backup=False)
    assert not db.exists()


def test_reset_creates_backup(tmp_path):
    db = _make_db(tmp_path / "journal.db")
    reset(db=db, backup=True)
    assert not db.exists()
    backups = list(tmp_path.glob("journal.backup_*.db"))
    assert len(backups) == 1
    # 备份必须真的含数据，不能只是个空壳
    conn = sqlite3.connect(str(backups[0]))
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    conn.close()


def test_reset_removes_wal_files(tmp_path):
    db = _make_db(tmp_path / "journal.db")
    wal = tmp_path / "journal.db-wal"
    shm = tmp_path / "journal.db-shm"
    wal.write_bytes(b"x")
    shm.write_bytes(b"x")

    reset(db=db, backup=False)
    assert not db.exists()
    assert not wal.exists()
    assert not shm.exists()


def test_reset_missing_db_is_noop(tmp_path):
    """数据库不存在时不应报错，也不应创建备份文件。"""
    missing = tmp_path / "nope.db"
    reset(db=missing, backup=True)
    assert list(tmp_path.glob("*.backup_*")) == []


def test_backup_name_is_unique_per_call(tmp_path):
    """连续两次重置不能互相覆盖备份。"""
    import time

    db = _make_db(tmp_path / "journal.db")
    reset(db=db, backup=True)
    time.sleep(1.05)  # 备份名精确到秒
    _make_db(db)
    reset(db=db, backup=True)
    assert len(list(tmp_path.glob("journal.backup_*.db"))) == 2
