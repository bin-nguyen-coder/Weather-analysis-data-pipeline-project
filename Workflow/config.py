"""Shared configuration loader for the weather data pipeline.

Every pipeline module reads its settings from ``Workflow/workflow_config.json``
through these helpers.  Relative artifact paths are resolved against the
repository root so modules behave identically whether they are launched
from the CLI, from the Airflow DAG, or from the local runner.

Environment variables can override any database setting (useful for Docker
Compose / GitHub Actions, e.g. ``PGHOST``, ``PGPORT``, ``PGDATABASE``,
``PGUSER``, ``PGPASSWORD``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "workflow_config.json"

# Environment variable -> database config key mapping.
_DB_ENV_OVERRIDES = {
    "host": "PGHOST",
    "port": "PGPORT",
    "dbname": "PGDATABASE",
    "user": "PGUSER",
    "password": "PGPASSWORD",
}


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load ``workflow_config.json``, resolve artifact paths and apply env
    overrides for the database connection.

    Parameters
    ----------
    path : optional
        Custom path to the JSON configuration. Defaults to the repository's
        ``Workflow/workflow_config.json``.

    Returns
    -------
    dict
        Fully resolved configuration.
    """
    config_path = Path(path) if path else CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    # Resolve relative artifact paths against the repository root.
    paths = config.get("paths", {})
    for key, value in paths.items():
        if isinstance(value, str):
            paths[key] = str(_resolve_path(value))

    # Environment overrides for database credentials (12-factor style).
    db = config.setdefault("database", {})
    for key, env_name in _DB_ENV_OVERRIDES.items():
        env_value = os.getenv(env_name)
        if env_value:
            db[key] = int(env_value) if key == "port" else env_value

    return config


def _resolve_path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else Path(PROJECT_ROOT, path)


if __name__ == "__main__":
    import json as _json

    config = load_config()
    print(_json.dumps(config, indent=2, default=str))