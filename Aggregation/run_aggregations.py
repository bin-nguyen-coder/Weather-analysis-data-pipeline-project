"""Runner that executes the SQL aggregation scripts against PostgreSQL.

Each ``*.sql`` file in this folder is self-contained and idempotent (it uses
``ON CONFLICT ... DO UPDATE``), so it can be executed sequentially every run.
psycopg2 allows multiple statements in one ``cursor.execute`` call, meaning
the plain SQL files can also be run manually with ``psql``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Ingestion.db_upsert import get_connection, init_schema  # noqa: E402
from Workflow.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_AGGREGATION_FILES = [
    "daily_aggregates.sql",
    "city_stats.sql",
]


def execute_sql_file(conn: Any, sql_path: Path) -> int:
    """Execute the statements in ``sql_path``; return number of rows affected.

    ``rowcount`` reflects the last statement only, so this is best-effort for
    informational purposes.
    """
    with open(sql_path, "r", encoding="utf-8") as handle:
        script = handle.read()
    with conn.cursor() as cursor:
        cursor.execute(script)
        affected = cursor.rowcount
    conn.commit()
    logger.info("Executed %s (affected %d rows)", sql_path.name, affected)
    return affected


def run_aggregations(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every aggregation SQL file over the analytics layer.

    Returns a report mapping SQL file name -> affected rows / tables.
    """
    config = config if config is not None else load_config()
    aggregation_dir = Path(__file__).resolve().parent

    conn = get_connection(config)
    try:
        init_schema(conn)
        results: List[Dict[str, Any]] = []
        for name in _AGGREGATION_FILES:
            sql_path = aggregation_dir / name
            if not sql_path.exists():
                logger.error("Missing aggregation script: %s", sql_path)
                continue
            affected = execute_sql_file(conn, sql_path)
            results.append({"script": name, "affected_rows": affected})
        return {"aggregations": results}
    finally:
        conn.close()


def run(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline entrypoint alias for ``run_aggregations``."""
    return run_aggregations(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run()
    print(report)