from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_quarterly_fundamental_momentum_diagnostic_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "quarterly_fundamental_runner_under_test", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_stable_hash_sample_sorts_hex_keys_deterministically():
    frame = pd.DataFrame({"accession": ["z", "a", "m"]})

    observed = RUNNER._stable_hash_sample(
        frame, column="accession", count=2
    )["accession"].tolist()
    expected = sorted(
        frame["accession"],
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )[:2]

    assert observed == expected


def test_write_json_converts_nonfinite_values_to_null(tmp_path):
    path = tmp_path / "evidence.json"

    RUNNER._write_json(
        path,
        {"ordinary": 1.5, "missing": float("nan"), "nested": [float("inf")]},
    )

    assert json.loads(path.read_text()) == {
        "ordinary": 1.5,
        "missing": None,
        "nested": [None],
    }
