#!/usr/bin/env python3
"""Minimal CLI to refresh/export OHLCV candles from MySQL into TimesFM-ready CSVs."""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pymysql
    from pymysql.cursors import SSDictCursor
except ImportError as exc:  # pragma: no cover - surfaced to the operator
    raise ImportError(
        "PyMySQL is required for MySQL exports. Install it via 'pip install PyMySQL'."
    ) from exc

from common.config import DEFAULT_ENV_PATH, load_env

app = typer.Typer(help="Export or refresh CSV datasets from the 'candles' MySQL table.")

load_env(DEFAULT_ENV_PATH)

def timeframe_to_seconds(timeframe: str) -> int:
    tf = timeframe.strip().upper()
    if tf.startswith("M"):
        return int(tf[1:]) * 60
    if tf.startswith("H"):
        return int(tf[1:]) * 3600
    if tf.startswith("D"):
        return int(tf[1:]) * 86400
    raise typer.BadParameter(f"Unsupported timeframe '{timeframe}'. Use M#, H#, or D#.")


def align_timestamp(ts: datetime, step_seconds: int) -> datetime:
    """Floor timestamps to the timeframe grid to collapse near-duplicates."""

    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (ts - midnight).total_seconds()
    floored = int(elapsed // step_seconds) * step_seconds
    return midnight + timedelta(seconds=floored)


def get_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        cursorclass=SSDictCursor,
    )


def iter_candles(
    conn: pymysql.connections.Connection,
    symbol: str,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Iterable[dict]:
    clauses = ["symbol = %s", "timeframe = %s"]
    params: list[object] = [symbol, timeframe]
    if start:
        clauses.append("open_time >= %s")
        params.append(start)
    if end:
        clauses.append("open_time <= %s")
        params.append(end)
    where = " AND ".join(clauses)
    sql = (
        "SELECT id, open_time, open, high, low, close, volume, updated_at "
        f"FROM candles WHERE {where} "
        "ORDER BY open_time ASC, updated_at DESC, id DESC"
    )

    with conn.cursor(SSDictCursor) as cursor:
        cursor.execute(sql, params)
        for row in cursor:
            yield row


@app.command()
def export(  # noqa: PLR0913 - CLI command with many knobs
    symbol: str = typer.Argument(..., help="Symbol to export, e.g. XAUUSDm."),
    timeframe: str = typer.Argument(..., help="Timeframe to export (M5, M1, H1, etc.)."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the CSV file. Defaults to data/<symbol>_<timeframe>.csv",
    ),
    start: Optional[datetime] = typer.Option(
        None, "--start", help="Optional UTC start timestamp (ISO-8601)."
    ),
    end: Optional[datetime] = typer.Option(
        None, "--end", help="Optional UTC end timestamp (ISO-8601)."
    ),
    db_host: str = typer.Option(..., "--db-host", envvar="DB_HOST", help="MySQL host."),
    db_port: int = typer.Option(3306, "--db-port", envvar="DB_PORT", help="MySQL port."),
    db_user: str = typer.Option(..., "--db-user", envvar="DB_USER", help="MySQL user."),
    db_pass: str = typer.Option(..., "--db-pass", envvar="DB_PASS", help="MySQL password."),
    db_name: str = typer.Option(..., "--db-name", envvar="DB_NAME", help="MySQL database name."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow overwriting the destination CSV."
    ),
    allow_gaps: bool = typer.Option(
        False,
        "--allow-gaps",
        help="Do not abort when weekend/holiday gaps are detected.",
    ),
) -> None:
    """Dump a clean CSV for the requested symbol/timeframe pair."""

    step_seconds = timeframe_to_seconds(timeframe)
    destination = (
        output
        if output is not None
        else Path("data") / f"{symbol.lower()}_{timeframe.lower()}.csv"
    ).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        typer.echo(f"{destination} already exists. Pass --overwrite to refresh it.")
        raise typer.Exit(code=1)

    conn = get_connection(db_host, db_port, db_user, db_pass, db_name)
    tmp_file = destination.with_suffix(".tmp")
    stats = {
        "rows_raw": 0,
        "rows_written": 0,
        "duplicates": 0,
        "misaligned": 0,
        "missing_intervals": 0,
        "missing_examples": [],
        "first_ts": None,
        "last_ts": None,
    }

    def record_missing(delta_seconds: int, previous_bucket: datetime) -> None:
        missing = (delta_seconds // step_seconds) - 1
        stats["missing_intervals"] += missing
        max_examples = 5
        if len(stats["missing_examples"]) >= max_examples:
            return
        for step in range(1, min(missing, max_examples - len(stats["missing_examples"])) + 1):
            stats["missing_examples"].append(
                previous_bucket + timedelta(seconds=step * step_seconds)
            )

    try:
        with tmp_file.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

            current_bucket: Optional[datetime] = None
            best_row: Optional[dict] = None
            best_key: Optional[Tuple[datetime, int]] = None
            prev_bucket: Optional[datetime] = None

            def flush_bucket() -> None:
                nonlocal best_row, best_key, current_bucket, prev_bucket
                if best_row is None or current_bucket is None:
                    return
                if prev_bucket is not None:
                    delta = int((current_bucket - prev_bucket).total_seconds())
                    if delta > step_seconds:
                        record_missing(delta, prev_bucket)
                ts_str = current_bucket.strftime("%Y-%m-%d %H:%M:%S")
                volume = best_row["volume"] if best_row["volume"] is not None else 0
                writer.writerow(
                    [ts_str, best_row["open"], best_row["high"], best_row["low"], best_row["close"], volume]
                )
                stats["rows_written"] += 1
                if stats["first_ts"] is None:
                    stats["first_ts"] = current_bucket
                stats["last_ts"] = current_bucket
                prev_bucket = current_bucket
                best_row = None
                best_key = None

            for row in iter_candles(conn, symbol, timeframe, start, end):
                stats["rows_raw"] += 1
                bucket = align_timestamp(row["open_time"], step_seconds)
                if row["open_time"] != bucket:
                    stats["misaligned"] += 1
                priority = (row.get("updated_at") or datetime.min, row.get("id") or 0)
                if current_bucket is None:
                    current_bucket = bucket
                    best_row = row
                    best_key = priority
                    continue

                if bucket != current_bucket:
                    flush_bucket()
                    current_bucket = bucket
                    best_row = row
                    best_key = priority
                    continue

                stats["duplicates"] += 1
                if best_key is None or priority > best_key:
                    best_row = row
                    best_key = priority

            flush_bucket()

        if stats["rows_written"] == 0:
            tmp_file.unlink(missing_ok=True)
            typer.echo("No rows matched the filters; nothing was exported.")
            raise typer.Exit(code=1)

        if stats["missing_intervals"] and not allow_gaps:
            tmp_file.unlink(missing_ok=True)
            typer.echo(
                "Detected gaps in the exported series. "
                "Run again with --allow-gaps if weekends/holidays are acceptable."
            )
            if stats["missing_examples"]:
                preview = ", ".join(str(ts) for ts in stats["missing_examples"])
                typer.echo(f"First missing timestamps: {preview}")
            raise typer.Exit(code=1)

        tmp_file.replace(destination)
        typer.echo(
            f"Wrote {destination} "
            f"(rows={stats['rows_written']}, raw_rows={stats['rows_raw']}, "
            f"duplicates={stats['duplicates']}, misaligned={stats['misaligned']}, "
            f"missing_intervals={stats['missing_intervals']})"
        )
        if stats["missing_examples"]:
            preview = ", ".join(str(ts) for ts in stats["missing_examples"])
            typer.echo(f"Missing timestamps preview: {preview}")
        typer.echo(
            f"Range: {stats['first_ts']} → {stats['last_ts']}"
        )

    finally:
        conn.close()
        if tmp_file.exists() and destination.exists() and tmp_file.stat().st_ino == destination.stat().st_ino:
            # already moved; nothing to clean
            pass


if __name__ == "__main__":
    app()
