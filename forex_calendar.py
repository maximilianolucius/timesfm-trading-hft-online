import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

try:
    import cloudscraper
except ImportError as exc:  # pragma: no cover - se detecta antes de ejecutar
    raise SystemExit(
        "No se pudo importar 'cloudscraper'. Ejecuta 'python -m pip install cloudscraper beautifulsoup4'"
    ) from exc

from requests import exceptions as requests_exceptions


TABLE_NAME = "forex_events"


def _import_mysql_driver():
    try:
        import mysql.connector as mysql_driver

        return {
            "module": mysql_driver,
            "error": mysql_driver.Error,
            "connect": mysql_driver.connect,
            "name": "mysql.connector",
        }
    except ImportError:
        try:
            import pymysql as mysql_driver

            return {
                "module": mysql_driver,
                "error": mysql_driver.MySQLError,
                "connect": mysql_driver.connect,
                "name": "pymysql",
            }
        except ImportError:
            print(
                "✗ No se encontraron librerías MySQL ('mysql-connector-python' o 'pymysql'). "
                "Instala una de ellas para continuar."
            )
            sys.exit(-1)


class ForexFactoryCalendarScraper:
    TIME_AMPM_RE = re.compile(r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<period>am|pm)$", re.IGNORECASE)
    TIME_24H_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$")
    SPECIAL_TIME_TOKENS = {"all day", "tentative", "tba", "day", "overnight", "n/a", "na", "-"}

    def __init__(
        self,
        request_timeout: int = 30,
        max_retries: int = 4,
        backoff_seconds: int = 5,
        backoff_multiplier: int = 2,
        batch_pause_every: int = 50,
        batch_pause_seconds: int = 5,
    ) -> None:
        self.error_log_path = Path("forex_calendar_errors.log")
        self.load_end()

        self.start_date = self.resolve_start_date()
        self.end_date = self.resolve_end_date()
        if self.end_date < self.start_date:
            self.report_parsing_issue(
                "El rango de fechas configurado es inválido (fin < inicio). Se intercambian valores."
            )
            self.start_date, self.end_date = self.end_date, self.start_date

        self.total_days = (self.end_date - self.start_date).days + 1

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.batch_pause_every = max(batch_pause_every, 1)
        self.batch_pause_seconds = max(batch_pause_seconds, 0)

        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        self.scraper.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.forexfactory.com/",
            }
        )

        self.db_driver_info = _import_mysql_driver()
        self.db_config = self.build_db_config()
        self.mysql_connection = None
        self.last_connection_label: Optional[str] = None
        self.setup_database()

    # -------------------------- utilidades generales --------------------------
    def load_end(self, env_path: str = ".env") -> None:
        self.load_env_file(env_path)

    def load_env_file(self, env_path: str = ".env") -> None:
        path = Path(env_path)
        if not path.exists():
            return

        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError as exc:
            self.report_parsing_issue(f"No se pudo leer el archivo .env: {exc}")

    def resolve_start_date(self) -> date:
        env_value = os.environ.get("FF_START_DATE")
        if env_value:
            try:
                return datetime.strptime(env_value.strip(), "%Y-%m-%d").date()
            except ValueError as exc:
                self.report_parsing_issue(
                    f"FF_START_DATE inválida ('{env_value}'). Se usará 2010-01-01.", exc
                )
        return date(2010, 1, 1)

    def resolve_end_date(self) -> date:
        env_value = os.environ.get("FF_END_DATE")
        if env_value:
            try:
                return datetime.strptime(env_value.strip(), "%Y-%m-%d").date()
            except ValueError as exc:
                self.report_parsing_issue(
                    f"FF_END_DATE inválida ('{env_value}'). Se usará la fecha actual.", exc
                )
        return date.today()

    def build_db_config(self) -> Dict[str, Optional[str]]:
        return {
            "host": os.environ.get("DB_HOST", "127.0.0.1"),
            "port": self.safe_int(os.environ.get("DB_PORT", "3306"), 3306),
            "user": os.environ.get("DB_USER", "root"),
            "password": os.environ.get("DB_PASS", ""),
            "database": os.environ.get("DB_NAME", "forex_calendar"),
            "unix_socket": os.environ.get("DB_SOCKET"),
            "fallback_hosts": [
                host.strip()
                for host in os.environ.get("DB_HOST_FALLBACKS", "").split(",")
                if host.strip()
            ],
        }

    @staticmethod
    def safe_int(value: Optional[str], default: int) -> int:
        if not value:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def report_parsing_issue(
        self,
        message: str,
        error: Optional[Exception] = None,
        row_snapshot: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        details = f"{timestamp}Z - {message}"
        if error:
            details = f"{details} | {error}"
        print(f"⚠ {details}")
        try:
            with self.error_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(details + "\n")
                if row_snapshot:
                    log_file.write(row_snapshot + "\n---\n")
        except OSError as log_exc:
            print(f"✗ No se pudo escribir en el log de errores: {log_exc}")

    # -------------------------- gestión MySQL --------------------------
    def mysql_connect(self, database: Optional[str] = None, autocommit: bool = False):
        if not self.db_driver_info:
            return None

        base_params = {
            "host": self.db_config["host"],
            "port": self.db_config["port"],
            "user": self.db_config["user"],
            "password": self.db_config["password"],
            "charset": "utf8mb4",
            "use_unicode": True,
        }
        if database:
            base_params["database"] = database

        attempts: List[Tuple[str, Dict[str, object]]] = []

        unix_socket = self.db_config.get("unix_socket")
        if unix_socket:
            socket_params = base_params.copy()
            socket_params.pop("host", None)
            socket_params.pop("port", None)
            socket_params["unix_socket"] = unix_socket
            attempts.append((f"socket:{unix_socket}", socket_params))

        host = base_params.get("host")
        if host:
            attempts.append((str(host), base_params))
        else:
            attempts.append(("127.0.0.1", {**base_params, "host": "127.0.0.1"}))

        fallback_hosts = self.db_config.get("fallback_hosts") or []
        default_fallbacks = ["127.0.0.1", "localhost"]

        for fallback in fallback_hosts + default_fallbacks:
            if not fallback:
                continue
            if host and str(host).lower() == fallback.lower():
                continue
            attempts.append((fallback, {**base_params, "host": fallback}))

        last_exc = None
        for label, params in attempts:
            try:
                conn = self.db_driver_info["connect"](**params)
                self.enable_autocommit(conn, autocommit)
                if label != str(host):
                    print(f"✓ Conexión MySQL usando {label}")
                self.last_connection_label = label
                return conn
            except self.db_driver_info["error"] as exc:  # type: ignore[index]
                last_exc = exc
                print(f"✗ Falló conexión MySQL ({label}): {exc}")
                continue

        raise RuntimeError(last_exc) from last_exc

    @staticmethod
    def enable_autocommit(conn, value: bool) -> None:
        if conn is None:
            return
        try:
            conn.autocommit = value  # mysql-connector
        except AttributeError:
            try:
                conn.autocommit(value)  # pymysql
            except Exception:
                pass

    def setup_database(self) -> None:
        if not self.db_driver_info:
            return

        try:
            server_conn = self.mysql_connect(autocommit=True)
        except RuntimeError as exc:
            self.report_parsing_issue(
                "No se pudo conectar al servidor MySQL.",
                exc,
            )
            sys.exit(-1)

        try:
            with server_conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.db_config['database']}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        except self.db_driver_info["error"] as exc:  # type: ignore[index]
            self.report_parsing_issue(
                "No se pudo crear/verificar la base de datos MySQL.",
                exc,
            )
            server_conn.close()
            sys.exit(-1)
        finally:
            server_conn.close()

        try:
            self.mysql_connection = self.mysql_connect(
                database=self.db_config["database"], autocommit=False
            )
        except RuntimeError as exc:
            self.report_parsing_issue(
                "No se pudo conectar a la base de datos MySQL.",
                exc,
            )
            sys.exit(-1)

        table_sql = (
            f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ("
            "id INT AUTO_INCREMENT PRIMARY KEY,"
            "title VARCHAR(255) NOT NULL,"
            "country VARCHAR(10),"
            "event_date DATE NOT NULL,"
            "event_time VARCHAR(32) NOT NULL DEFAULT '',"
            "event_datetime_utc DATETIME NULL,"
            "impact VARCHAR(32),"
            "forecast VARCHAR(100),"
            "previous_value VARCHAR(100),"
            "source_url VARCHAR(255),"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "UNIQUE KEY uniq_event (title, country, event_date, event_time)"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )

        try:
            with self.mysql_connection.cursor() as cursor:
                cursor.execute(table_sql)
            self.mysql_connection.commit()
        except self.db_driver_info["error"] as exc:  # type: ignore[index]
            self.report_parsing_issue(
                "No se pudo crear/verificar la tabla forex_events.",
                exc,
            )
            self.mysql_connection.close()
            sys.exit(-1)

        driver_name = self.db_driver_info.get("name") if self.db_driver_info else "desconocido"
        target = self.last_connection_label or f"{self.db_config['host']}:{self.db_config['port']}"
        print(
            f"✓ Motor MySQL: {driver_name} | Conexión: {target} | Base: {self.db_config['database']} | Tabla: {TABLE_NAME}"
        )

    def persist_to_database(self, events: List[Dict[str, str]], context: str = "") -> None:
        if not events or not self.mysql_connection or not self.db_driver_info:
            return

        insert_sql = (
            f"INSERT IGNORE INTO {TABLE_NAME} "
            "(title, country, event_date, event_time, event_datetime_utc, impact, forecast, previous_value, source_url) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )

        rows_to_insert: List[Tuple[Optional[str], ...]] = []
        for event in events:
            date_value = event.get("Date")
            time_value = event.get("Time") or ""
            iso_utc = event.get("EventDatetimeUTC")
            try:
                date_iso = datetime.strptime(date_value, "%Y-%m-%d").date() if date_value else None
            except (TypeError, ValueError) as exc:
                self.report_parsing_issue(
                    f"Fecha inválida para inserción en DB: '{date_value}'", exc
                )
                continue

            rows_to_insert.append(
                (
                    event.get("Title"),
                    event.get("Country"),
                    date_iso.isoformat() if date_iso else None,
                    time_value,
                    iso_utc,
                    event.get("Impact"),
                    event.get("Forecast"),
                    event.get("Previous"),
                    event.get("URL"),
                )
            )

        if not rows_to_insert:
            return

        existing_count = None
        try:
            with self.mysql_connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                result = cursor.fetchone()
                if result is not None:
                    existing_count = result[0]
        except self.db_driver_info["error"] as exc:  # type: ignore[index]
            self.report_parsing_issue(
                f"No se pudo leer el estado inicial de {TABLE_NAME}.",
                exc,
            )
            sys.exit(-1)

        try:
            with self.mysql_connection.cursor() as cursor:
                cursor.executemany(insert_sql, rows_to_insert)
                affected = cursor.rowcount
            self.mysql_connection.commit()
            label = context or "batch"
            print(f"✓ Registros insertados/ignorados en MySQL ({label}): {affected}")
        except self.db_driver_info["error"] as exc:  # type: ignore[index]
            self.report_parsing_issue("No se pudo insertar datos en MySQL.", exc)
            sys.exit(-1)

        if existing_count == 0 and (affected is None or affected <= 0):
            try:
                with self.mysql_connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                    result = cursor.fetchone()
            except self.db_driver_info["error"] as exc:  # type: ignore[index]
                self.report_parsing_issue(
                    f"No se pudo verificar la cantidad final en {TABLE_NAME}.",
                    exc,
                )
                sys.exit(-1)
            else:
                final_count = result[0] if result else None
                if not final_count:
                    self.report_parsing_issue(
                        "La inserción no afectó filas en una tabla vacía. Verifica permisos o la sentencia SQL."
                    )
                    sys.exit(-1)

    # -------------------------- helpers de parsing --------------------------
    def fetch_html(self, url: str) -> str:
        last_error: Optional[BaseException] = None
        wait_seconds = self.backoff_seconds

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.scraper.get(url, timeout=self.request_timeout)
            except requests_exceptions.RequestException as exc:
                last_error = exc
                print(f"✗ Error de conexión (intento {attempt}/{self.max_retries}): {exc}")
                time.sleep(wait_seconds)
                wait_seconds *= self.backoff_multiplier
                continue

            if response.status_code == 200:
                return response.text

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    sleep_time = int(float(retry_after)) if retry_after else wait_seconds
                except ValueError:
                    sleep_time = wait_seconds
                print(
                    f"⚠ Rate limit alcanzado en {url}. Esperando {sleep_time} segundos antes de reintentar..."
                )
                time.sleep(max(sleep_time, 1))
                wait_seconds *= self.backoff_multiplier
                continue

            if 500 <= response.status_code < 600:
                print(
                    f"⚠ Respuesta {response.status_code} desde {url}. Reintentando en {wait_seconds} segundos..."
                )
                time.sleep(wait_seconds)
                wait_seconds *= self.backoff_multiplier
                continue

            response.raise_for_status()

        raise RuntimeError(f"No se pudo obtener {url}: {last_error or 'se superó el número de reintentos'}")

    def get_year_from_url(self, url: str) -> Optional[int]:
        match = re.search(r"day=[a-z]+\d+\.(\d{4})", url, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def parse_date(self, date_text: str, fallback_year: Optional[int]) -> Optional[str]:
        if not date_text:
            return None

        cleaned = re.sub(r"\s+", " ", date_text.replace("\xa0", " ")).strip()
        cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.split("-")[0].strip()

        weekday_short = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        parts = cleaned.split()
        if parts and parts[0][:3].lower() in weekday_short:
            parts = parts[1:]
        if len(parts) >= 2 and fallback_year:
            month_part, day_part = parts[0], re.sub(r"[^0-9]", "", parts[1])
            if month_part and day_part:
                try:
                    dt = datetime.strptime(
                        f"{month_part[:3]} {day_part} {fallback_year}", "%b %d %Y"
                    )
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

        if fallback_year:
            try:
                dt = datetime.strptime(f"{cleaned} {fallback_year}", "%b %d %Y")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return None

    def parse_time(self, time_text: str) -> str:
        if not time_text:
            return ""

        stripped = time_text.strip()
        lowered = stripped.lower()
        if lowered in self.SPECIAL_TIME_TOKENS:
            return stripped.title()

        lowered = lowered.replace(" ", "")
        match = self.TIME_AMPM_RE.match(lowered)
        if match:
            hour = int(match.group("hour"))
            minute = match.group("minute") or "00"
            period = match.group("period").lower()
            hour = hour % 12
            if period == "pm":
                hour += 12
            return f"{hour:02d}:{minute}"

        match_24 = self.TIME_24H_RE.match(lowered)
        if match_24:
            hour = int(match_24.group("hour"))
            minute = match_24.group("minute")
            return f"{hour:02d}:{minute}"

        self.report_parsing_issue(f"No se pudo interpretar la hora '{time_text}'. Se mantiene el valor original.")
        return stripped

    def clean_cell(self, cell: Optional[BeautifulSoup]) -> str:
        if cell is None:
            return ""
        text = cell.get_text(separator=" ", strip=True).replace("\xa0", " ")
        return re.sub(r"\s+", " ", text)

    def get_impact_level(self, impact_cell: Optional[BeautifulSoup]) -> str:
        if impact_cell is None:
            return "Low"
        span = impact_cell.find("span")
        classes = span.get("class", []) if span else []
        classes = {cls for cls in classes if isinstance(cls, str)}

        if any("impact-red" in cls for cls in classes):
            return "High"
        if any("impact-ora" in cls for cls in classes):
            return "Medium"
        if any("impact-yel" in cls for cls in classes):
            return "Low"
        if any("impact-grey" in cls for cls in classes):
            return "Holiday"
        return "Low"

    def extract_date_from_row(self, row: BeautifulSoup, fallback_year: Optional[int]) -> Optional[str]:
        dateline = row.get("data-day-dateline")
        if dateline:
            try:
                timestamp = int(dateline)
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass

        date_cell = row.find("td", class_="calendar__date")
        if date_cell:
            parsed = self.parse_date(date_cell.get_text(" ", strip=True), fallback_year)
            if parsed:
                return parsed

        row_text = row.get_text(" ", strip=True)
        if row_text:
            parsed = self.parse_date(row_text, fallback_year)
            if parsed:
                return parsed

        return None

    def convert_to_utc(self, date_str: str, time_str: str) -> Tuple[str, str, Optional[str]]:
        if not time_str:
            return date_str, time_str, None

        lowered = time_str.lower()
        if lowered in {token.lower() for token in self.SPECIAL_TIME_TOKENS}:
            return date_str, time_str, None

        match = self.TIME_24H_RE.match(time_str)
        if not match:
            self.report_parsing_issue(
                f"Formato de hora inesperado '{time_str}'. No se convierte a UTC."
            )
            return date_str, time_str, None

        hour = int(match.group("hour"))
        minute = int(match.group("minute"))

        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            self.report_parsing_issue(
                f"Fecha '{date_str}' inválida al convertir a UTC.",
                exc,
            )
            return date_str, time_str, None

        local_dt = base_date + timedelta(hours=hour, minutes=minute)
        utc_dt = local_dt + timedelta(hours=3)
        converted_date = utc_dt.strftime("%Y-%m-%d")
        converted_time = utc_dt.strftime("%H:%M")
        return converted_date, converted_time, utc_dt.strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------- scraping --------------------------
    def scrape_calendar_page(self, url: str) -> List[Dict[str, str]]:
        print(f"\n{'=' * 60}\nProcesando: {url}")
        html = self.fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        rows = soup.select("tr.calendar__row")
        if not rows:
            print("⚠ No se encontraron filas en la tabla del calendario")
            return []

        events: List[Dict[str, str]] = []
        fallback_year = self.get_year_from_url(url)
        current_date: Optional[str] = None

        for row in rows:
            try:
                classes = row.get("class", [])
                if "calendar__row--day-breaker" in classes:
                    extracted = self.extract_date_from_row(row, fallback_year)
                    if extracted:
                        current_date = extracted
                    else:
                        self.report_parsing_issue(
                            f"No se pudo extraer la fecha del separador de día en {url}.",
                            row_snapshot=row.get_text(" ", strip=True),
                        )
                    continue

                extracted_date = self.extract_date_from_row(row, fallback_year)
                if extracted_date:
                    current_date = extracted_date

                if not current_date:
                    continue

                event_cell = row.find("td", class_="calendar__event")
                if event_cell is None:
                    continue

                title = event_cell.get_text(separator=" ", strip=True)
                if not title:
                    continue

                time_cell = row.find("td", class_="calendar__time")
                country_cell = row.find("td", class_="calendar__currency")
                impact_cell = row.find("td", class_="calendar__impact")
                forecast_cell = row.find("td", class_="calendar__forecast")
                previous_cell = row.find("td", class_="calendar__previous")

                raw_time = self.parse_time(time_cell.get_text(" ", strip=True) if time_cell else "")
                country = self.clean_cell(country_cell)
                impact = self.get_impact_level(impact_cell)
                forecast = self.clean_cell(forecast_cell)
                previous = self.clean_cell(previous_cell)

                date_utc, time_utc, iso_utc = self.convert_to_utc(current_date, raw_time)

                event = {
                    "Title": title,
                    "Country": country,
                    "Date": date_utc,
                    "Time": time_utc,
                    "Impact": impact,
                    "Forecast": forecast,
                    "Previous": previous,
                    "URL": url,
                    "OriginalTime": raw_time,
                }
                if iso_utc:
                    event["EventDatetimeUTC"] = iso_utc

                events.append(event)
            except Exception as exc:  # pragma: no cover - protección en scraping
                self.report_parsing_issue(
                    f"Error procesando una fila en {url}",
                    exc,
                    row_snapshot=row.get_text(" ", strip=True),
                )
                continue

        print(f"✓ Eventos extraídos: {len(events)}")
        return events

    def scrape_multiple_urls(self, urls: Iterable[str]) -> List[Dict[str, str]]:
        all_events: List[Dict[str, str]] = []

        for index, url in enumerate(urls, start=1):
            try:
                events = self.scrape_calendar_page(url)
            except Exception as exc:
                self.report_parsing_issue(f"✗ Error procesando {url}", exc)
                continue

            if events:
                context = events[0].get("Date") or url
                self.persist_to_database(events, context=context)

            all_events.extend(events)

            if index % self.batch_pause_every == 0:
                print(
                    f"Pausa de {self.batch_pause_seconds} segundos tras {index} días para evitar bloqueos."
                )
                time.sleep(self.batch_pause_seconds)

        return all_events

    def generate_calendar_urls(self, start: Optional[date] = None, end: Optional[date] = None) -> Iterable[str]:
        start_date = start or self.start_date
        end_date = end or self.end_date
        current = start_date
        while current <= end_date:
            month_token = current.strftime("%b").lower()
            day_token = str(current.day)
            yield f"https://www.forexfactory.com/calendar?day={month_token}{day_token}.{current.year}"
            current += timedelta(days=1)

    # -------------------------- salida --------------------------
    def save_to_csv(self, events: List[Dict[str, str]], filename: str = "forex_calendar_merged.csv") -> None:
        if not events:
            print("\n⚠ No hay eventos para guardar")
            return

        df = pd.DataFrame(events)
        columns = ["Title", "Country", "Date", "Time", "Impact", "Forecast", "Previous", "URL"]
        df = df.reindex(columns=columns)
        df.to_csv(filename, index=False, encoding="utf-8")

        print(f"\n{'=' * 60}\n✓ Datos guardados en: {filename}\n✓ Total de eventos: {len(df)}")
        print(f"\n{'=' * 60}\nPREVIEW DE LOS DATOS:\n{'=' * 60}")
        print(df.head(10).to_string(index=False))

        print(f"\n{'=' * 60}\nESTADÍSTICAS:\n{'=' * 60}")
        print("Eventos por país:")
        print(df["Country"].value_counts().head(10))
        print("\nEventos por nivel de impacto:")
        print(df["Impact"].value_counts())

    def close(self) -> None:
        if self.mysql_connection:
            try:
                self.mysql_connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    scraper: Optional[ForexFactoryCalendarScraper] = None
    try:
        print("=" * 60)
        print("FOREX FACTORY CALENDAR SCRAPER")
        print("=" * 60)

        scraper = ForexFactoryCalendarScraper()
        print(
            f"Rango de fechas: {scraper.start_date.isoformat()} -> {scraper.end_date.isoformat()} "
            f"({scraper.total_days} días)"
        )

        url_iterator = scraper.generate_calendar_urls()
        events = scraper.scrape_multiple_urls(url_iterator)
        scraper.save_to_csv(events, "forex_calendar_merged.csv")

        print("\n✓ Proceso completado exitosamente!")
    except Exception as exc:  # pragma: no cover - main script
        print(f"\n✗ Error general: {exc}")
    finally:
        if scraper:
            scraper.close()
