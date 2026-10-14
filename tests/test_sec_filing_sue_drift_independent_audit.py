import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_sec_filing_sue_drift_v1 import (
    independent_bootstraps,
    independent_hac,
    verify_manifest,
)


def test_independent_hac_matches_hand_calculation_at_zero_lags():
    values = np.array([-0.02, 0.01, 0.03, -0.01, 0.02])
    result = independent_hac(values, lags=0)
    expected_mean = values.mean()
    residual = values - expected_mean
    expected_se = math.sqrt(np.dot(residual, residual) / len(values) / len(values))
    assert result["mean"] == pytest.approx(expected_mean)
    assert result["standard_error"] == pytest.approx(expected_se)
    assert result["t"] == pytest.approx(expected_mean / expected_se)


def test_independent_bootstrap_repair_is_deterministic_and_design_centered():
    values = np.sin(np.linspace(0, 11, 53)) / 100
    first = independent_bootstraps(values, draws=2_000, block_length=8, seed=929)
    second = independent_bootstraps(values, draws=2_000, block_length=8, seed=929)
    assert np.array_equal(first["raw"], second["raw"])
    assert np.array_equal(first["corrected"], second["corrected"])
    assert np.array_equal(first["circular"], second["circular"])
    assert np.array_equal(
        first["corrected"], first["raw"] - first["design_expectation"]
    )
    mcse = first["corrected_summary"]["null_std"] / np.sqrt(first["draws"])
    assert abs(first["corrected_summary"]["null_mean"]) <= 4 * mcse


def test_manifest_verification_detects_hash_mutation(tmp_path: Path):
    output = tmp_path / "result.csv"
    output.write_text("x\n1\n", encoding="utf-8")
    import hashlib
    import json

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "record_count": 1,
                "all_records_exist": True,
                "records": [
                    {
                        "path": "result.csv",
                        "role": "output",
                        "bytes": output.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    clean = verify_manifest(
        root=tmp_path, manifest_path=manifest, required_names={"result.csv", "artifact_manifest.json"}
    )
    assert clean["all_hashes_match"]
    output.write_text("x\n2\n", encoding="utf-8")
    mutated = verify_manifest(
        root=tmp_path, manifest_path=manifest, required_names={"result.csv", "artifact_manifest.json"}
    )
    assert not mutated["all_hashes_match"]
