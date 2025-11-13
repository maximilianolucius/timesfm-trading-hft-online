"""Utilities to validate and export OHLCV candles from MySQL into model-ready CSVs."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import typer

try:
    import pymysql
    from pymysql.cursors import SSDictCursor
except ImportError as exc:  # pragma: no cover - surfaced quickly to the operator
    raise ImportError(
        "PyMySQL is required for MySQL validation/export. Install it via 'pip install PyMySQL'."
    ) from exc

app = typer.Typer(help="Validate MySQL OHLCV series and export clean CSV files.")


@dataclass
class SeriesStats:
    """Accumulates integrity stats while streaming through a candle series."""

    symbol: str
    timeframe: str
    frequency: timedelta
    max_gap_report: int = 10
    rows: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    duplicate_count: int = 0
    overlap_count: int = 0
    missing_intervals: int = 0
    missing_examples: List[datetime] = field(default_factory=list)
    misaligned_timestamps: int = 0
    nan_values: int = 0
    _prev_timestamp: Optional[datetime] = None
    tz_infos: set[str] = field(default_factory=set)

    def observe(self, timestamp: datetime, values: Sequence[Optional[float]]) -> None:
        if self.first_timestamp is None:
            self.first_timestamp = timestamp

        if self._prev_timestamp is not None:
            expected_seconds = int(self.frequency.total_seconds())
            delta_seconds = int((timestamp - self._prev_timestamp).total_seconds())

            if delta_seconds == 0:
                self.duplicate_count += 1
            elif delta_seconds < expected_seconds:
                self.overlap_count += 1
            elif delta_seconds > expected_seconds:
                missing = (delta_seconds // expected_seconds) - 1
                self.missing_intervals += missing
                if len(self.missing_examples) < self.max_gap_report:
                    for step in range(1, missing + 1):
                        self.missing_examples.append(self._prev_timestamp + step * self.frequency)
                        if len(self.missing_examples) >= self.max_gap_report:
                            break

        if timestamp.second != 0 or timestamp.microsecond != 0:
            self.misaligned_timestamps += 1
        elif not self._matches_frequency(timestamp):
            self.misaligned_timestamps += 1

        if timestamp.tzinfo is None:
            self.tz_infos.add("naive")
        else:
            tz_name = timestamp.tzinfo.tzname(timestamp) or str(timestamp.tzinfo)
            self.tz_infos.add(tz_name)
        self.nan_values += sum(1 for value in values if self._is_bad(value))
        self.rows += 1
        self.last_timestamp = timestamp
        self._prev_timestamp = timestamp

    def _matches_frequency(self, timestamp: datetime) -> bool:
        expected_seconds = int(self.frequency.total_seconds())
        if expected_seconds == 0:
            return True
        midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = int((timestamp - midnight).total_seconds())
        return (seconds_since_midnight % expected_seconds) == 0

    @staticmethod
    def _is_bad(value: Optional[float]) -> bool:
        if value is None:
            return True
        if isinstance(value, (float, Decimal)):
            return math.isnan(float(value))
        return False


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    timeframe = timeframe.upper()
    if timeframe.startswith("M"):
        return timedelta(minutes=int(timeframe[1:]))
    if timeframe.startswith("H"):
        return timedelta(hours=int(timeframe[1:]))
    if timeframe.startswith("D"):
        return timedelta(days=int(timeframe[1:]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


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


def fetch_available_symbols(
    conn: pymysql.connections.Connection, timeframe: str
) -> List[str]:
    query = (
        "SELECT DISTINCT symbol FROM candles WHERE timeframe = %s ORDER BY symbol"
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (timeframe,))
        rows = cursor.fetchall()
    return [row["symbol"] for row in rows]


def fetch_server_timezones(conn: pymysql.connections.Connection) -> Dict[str, str]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT @@system_time_zone AS system_tz, @@session.time_zone AS session_tz")
        row = cursor.fetchone()
    return {
        "system": row.get("system_tz", "unknown"),
        "session": row.get("session_tz", "unknown"),
    }


def iter_candles(
    conn: pymysql.connections.Connection,
    symbol: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Iterator[Dict[str, Optional[float]]]:
    clauses = ["timeframe = %s", "symbol = %s"]
    params: List[object] = [timeframe, symbol]
    if start is not None:
        clauses.append("open_time >= %s")
        params.append(start)
    if end is not None:
        clauses.append("open_time <= %s")
        params.append(end)

    where = " AND ".join(clauses)
    query = (
        "SELECT open_time, open, high, low, close, volume "
        "FROM candles WHERE " + where + " ORDER BY open_time ASC"
    )
    with conn.cursor(SSDictCursor) as cursor:
        cursor.execute(query, params)
        for row in cursor:
            yield row


def normalize_symbol(symbol: str) -> str:
    return symbol.replace(" ", "").upper()


@app.command()
def validate(
    timeframe: str = typer.Option(..., "--timeframe", "-t", help="Timeframe to validate (e.g. M5)."),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Symbols to validate (defaults to all with the timeframe)."
    ),
    start: Optional[datetime] = typer.Option(
        None, help="Start timestamp (inclusive, UTC). Expected ISO-8601."
    ),
    end: Optional[datetime] = typer.Option(
        None, help="End timestamp (inclusive, UTC). Expected ISO-8601."
    ),
    db_host: str = typer.Option(..., envvar="DB_HOST", help="MySQL host."),
    db_port: int = typer.Option(3306, envvar="DB_PORT", help="MySQL port."),
    db_user: str = typer.Option(..., envvar="DB_USER", help="MySQL user."),
    db_pass: str = typer.Option(..., envvar="DB_PASS", help="MySQL password."),
    db_name: str = typer.Option(..., envvar="DB_NAME", help="MySQL database name."),
) -> None:
    freq = timeframe_to_timedelta(timeframe)
    conn = get_connection(db_host, db_port, db_user, db_pass, db_name)
    try:
        tz_info = fetch_server_timezones(conn)
        typer.echo(f"Server timezone: system={tz_info['system']}, session={tz_info['session']}")

        target_symbols = symbols or fetch_available_symbols(conn, timeframe)
        if not target_symbols:
            typer.echo("No symbols found for timeframe.")
            raise typer.Exit(code=1)

        exit_code = 0
        for symbol in target_symbols:
            stats = SeriesStats(symbol=symbol, timeframe=timeframe, frequency=freq)
            for row in iter_candles(conn, symbol, timeframe, start, end):
                timestamp = row["open_time"]
                if not isinstance(timestamp, datetime):
                    raise typer.BadParameter("open_time column must be DATETIME/TIMESTAMP")
                values = [row["open"], row["high"], row["low"], row["close"], row["volume"]]
                stats.observe(timestamp, values)

            if stats.rows == 0:
                typer.echo(f"{symbol} {timeframe}: no rows returned for the selected window.")
                continue

            typer.echo(
                f"{symbol} {timeframe}: rows={stats.rows}, "
                f"range=({stats.first_timestamp} → {stats.last_timestamp}), "
                f"missing={stats.missing_intervals}, duplicates={stats.duplicate_count}, "
                f"overlap={stats.overlap_count}, misaligned={stats.misaligned_timestamps}, "
                f"nan_values={stats.nan_values}, tz_info={sorted(stats.tz_infos)}"
            )
            if stats.missing_examples:
                preview = ", ".join(str(ts) for ts in stats.missing_examples[:5])
                typer.echo(f"  Missing timestamps (first {len(stats.missing_examples)}): {preview}")
            if stats.nan_values or stats.missing_intervals or stats.duplicate_count or stats.overlap_count:
                exit_code = 1

    finally:
        conn.close()

    raise typer.Exit(code=exit_code)


@app.command()
def export(
    output_dir: Path = typer.Option(
        Path("data"), "--output", "-o", help="Directory where CSV files will be stored."
    ),
    timeframe: str = typer.Option(..., "--timeframe", "-t", help="Timeframe to export (e.g. M5)."),
    symbols: Optional[List[str]] = typer.Option(
        None, "--symbol", "-s", help="Symbols to export (defaults to all with the timeframe)."
    ),
    start: Optional[datetime] = typer.Option(
        None, help="Start timestamp (inclusive, UTC). Expected ISO-8601."
    ),
    end: Optional[datetime] = typer.Option(
        None, help="End timestamp (inclusive, UTC). Expected ISO-8601."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow overwriting existing CSV files."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Export even when gaps or NaNs are detected (a warning is printed).",
    ),
    db_host: str = typer.Option(..., envvar="DB_HOST", help="MySQL host."),
    db_port: int = typer.Option(3306, envvar="DB_PORT", help="MySQL port."),
    db_user: str = typer.Option(..., envvar="DB_USER", help="MySQL user."),
    db_pass: str = typer.Option(..., envvar="DB_PASS", help="MySQL password."),
    db_name: str = typer.Option(..., envvar="DB_NAME", help="MySQL database name."),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    freq = timeframe_to_timedelta(timeframe)
    conn = get_connection(db_host, db_port, db_user, db_pass, db_name)

    try:
        target_symbols = symbols or fetch_available_symbols(conn, timeframe)
        if not target_symbols:
            typer.echo("No symbols found for timeframe.")
            raise typer.Exit(code=1)

        for symbol in target_symbols:
            stats = SeriesStats(symbol=symbol, timeframe=timeframe, frequency=freq)
            safe_symbol = normalize_symbol(symbol)
            target_file = output_dir / f"{safe_symbol.lower()}_{timeframe.lower()}.csv"
            if target_file.exists() and not overwrite:
                typer.echo(
                    f"Skipping {symbol}: {target_file} already exists. Use --overwrite to replace."
                )
                continue

            temp_file = target_file.with_suffix(".tmp")
            with temp_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

                for row in iter_candles(conn, symbol, timeframe, start, end):
                    timestamp = row["open_time"]
                    if not isinstance(timestamp, datetime):
                        temp_file.unlink(missing_ok=True)
                        raise typer.BadParameter("open_time column must be DATETIME/TIMESTAMP")
                    values = [row["open"], row["high"], row["low"], row["close"], row["volume"]]
                    stats.observe(timestamp, values)

                    writer.writerow([
                        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        *values,
                    ])

            if stats.rows == 0:
                temp_file.unlink(missing_ok=True)
                typer.echo(f"Skipped {symbol}: no rows returned for the selected window.")
                continue

            has_issues = bool(
                stats.missing_intervals
                or stats.duplicate_count
                or stats.overlap_count
                or stats.nan_values
            )
            if has_issues and not force:
                temp_file.unlink(missing_ok=True)
                typer.echo(
                    f"Aborted export for {symbol}: integrity issues detected "
                    f"(missing={stats.missing_intervals}, duplicates={stats.duplicate_count}, "
                    f"overlap={stats.overlap_count}, nan_values={stats.nan_values})."
                )
                continue

            temp_file.replace(target_file)
            typer.echo(
                f"Exported {symbol} to {target_file} | rows={stats.rows}, "
                f"missing={stats.missing_intervals}, duplicates={stats.duplicate_count}, "
                f"overlap={stats.overlap_count}, misaligned={stats.misaligned_timestamps}, "
                f"nan_values={stats.nan_values}, tz_info={sorted(stats.tz_infos)}"
            )
            if has_issues and force:
                typer.echo("  WARNING: export forced despite detected issues.")

    finally:
        conn.close()


if __name__ == "__main__":
    app()
