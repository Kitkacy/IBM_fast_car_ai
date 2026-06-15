"""Shared utility functions used across the torcs_rl module."""

import csv
import json
import math
from pathlib import Path

import numpy as np


def safe_float(value):
    """Convert *value* to float or return ``None`` on any failure."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def write_json(path, payload):
    """Write *payload* as pretty-printed JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv_row(path, fieldnames, row):
    """Append a single CSV *row* to *path*, creating headers on first write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)