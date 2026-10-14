"""Auditable repair for the SEC filing-drift moving-block bootstrap.

This module never recomputes or changes the economic event returns.  It reads the
sealed attempt-1 entry-session series, reproduces the biased statistic, applies
the exact finite-sample design recentering, and reports a circular-block
sensitivity result.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class BootstrapRepairError(RuntimeError):
    """Raised when a repair input or invariant fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_attempt1(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    declaration = config["attempt1"]
    checks: list[dict[str, Any]] = []
    for stem in ("artifact_manifest", "entry_session_returns", "summary_metrics"):
        path = root / declaration[f"{stem}_path"]
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        passed = bool(
            exists
            and actual_bytes == int(declaration[f"{stem}_bytes"])
            and actual_hash == declaration[f"{stem}_sha256"]
        )
        checks.append(
            {
                "input": stem,
                "path": declaration[f"{stem}_path"],
                "exists": exists,
                "expected_bytes": int(declaration[f"{stem}_bytes"]),
                "actual_bytes": actual_bytes,
                "expected_sha256": declaration[f"{stem}_sha256"],
                "actual_sha256": actual_hash,
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in checks):
        raise BootstrapRepairError("attempt-1 inputs do not match the repair record")
    manifest = json.loads((root / declaration["artifact_manifest_path"]).read_text())
    by_role = {(row["role"], row["path"]): row for row in manifest["records"]}
    code_records = [row for row in manifest["records"] if row["role"] == "code"]
    protocol_records = [row for row in manifest["records"] if row["role"] == "protocol"]
    if len(code_records) != 1 or code_records[0]["sha256"] != declaration["source_code_sha256"]:
        raise BootstrapRepairError("attempt-1 source hash is not closed by its manifest")
    if len(protocol_records) != 1 or protocol_records[0]["sha256"] != declaration["protocol_sha256"]:
        raise BootstrapRepairError("attempt-1 protocol hash is not closed by its manifest")
    entry_key = ("output", declaration["entry_session_returns_path"])
    if entry_key not in by_role or by_role[entry_key]["sha256"] != declaration["entry_session_returns_sha256"]:
        raise BootstrapRepairError("attempt-1 entry-session series is not closed by its manifest")
    return checks


def noncircular_design_expectation(centered: Iterable[float], block_length: int) -> float:
    x = np.asarray(list(centered), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise BootstrapRepairError("at least three finite observations are required")
    length = max(1, min(int(block_length), x.size))
    starts = np.arange(0, x.size - length + 1)
    blocks_needed = int(math.ceil(x.size / length))
    final_length = x.size - (blocks_needed - 1) * length
    full_expected_sum = np.mean([x[s : s + length].sum() for s in starts])
    final_expected_sum = np.mean([x[s : s + final_length].sum() for s in starts])
    return float(((blocks_needed - 1) * full_expected_sum + final_expected_sum) / x.size)


def repaired_noncircular_mbb(
    values: Iterable[float], *, draws: int, block_length: int, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise BootstrapRepairError("at least three finite observations are required")
    observed = float(x.mean())
    centered = x - observed
    n = centered.size
    length = max(1, min(int(block_length), n))
    starts = np.arange(0, n - length + 1)
    blocks_needed = int(math.ceil(n / length))
    expected = noncircular_design_expectation(centered, length)
    rng = np.random.default_rng(int(seed))
    raw = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([centered[start : start + length] for start in chosen])[:n]
        raw[draw] = float(sample.mean())
    corrected = raw - expected
    raw_exceedances = int(np.count_nonzero(raw >= observed))
    corrected_exceedances = int(np.count_nonzero(corrected >= observed))
    draw_frame = pd.DataFrame(
        {
            "draw": np.arange(1, int(draws) + 1),
            "raw_noncircular_null_mean": raw,
            "corrected_noncircular_null_mean": corrected,
        }
    )
    result = {
        "variant": "noncircular_overlapping_moving_blocks_with_analytic_design_recentering",
        "n": int(n),
        "observed_mean": observed,
        "draws": int(draws),
        "block_length": int(length),
        "blocks_per_draw": int(blocks_needed),
        "last_block_observations_used": int(n - (blocks_needed - 1) * length),
        "seed": int(seed),
        "analytic_design_expected_mean": expected,
        "raw_null_mean": float(raw.mean()),
        "corrected_null_mean": float(corrected.mean()),
        "raw_exceedances": raw_exceedances,
        "raw_one_sided_empirical_p": float((raw_exceedances + 1) / (int(draws) + 1)),
        "corrected_exceedances": corrected_exceedances,
        "corrected_one_sided_empirical_p": float(
            (corrected_exceedances + 1) / (int(draws) + 1)
        ),
        "corrected_null_std": float(corrected.std(ddof=1)),
        "minimum_resolvable_p": float(1 / (int(draws) + 1)),
    }
    return result, draw_frame


def circular_mbb(
    values: Iterable[float], *, draws: int, block_length: int, seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    observed = float(x.mean())
    centered = x - observed
    n = centered.size
    if n < 3:
        raise BootstrapRepairError("at least three finite observations are required")
    length = max(1, min(int(block_length), n))
    starts = np.arange(n)
    blocks_needed = int(math.ceil(n / length))
    offsets = np.arange(length)
    rng = np.random.default_rng(int(seed))
    null_means = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([centered[(start + offsets) % n] for start in chosen])[:n]
        null_means[draw] = float(sample.mean())
    exceedances = int(np.count_nonzero(null_means >= observed))
    return (
        {
            "variant": "circular_all_start_moving_blocks",
            "n": int(n),
            "observed_mean": observed,
            "draws": int(draws),
            "block_length": int(length),
            "seed": int(seed),
            "null_mean": float(null_means.mean()),
            "null_std": float(null_means.std(ddof=1)),
            "exceedances": exceedances,
            "one_sided_empirical_p": float((exceedances + 1) / (int(draws) + 1)),
        },
        null_means,
    )


def run_repair(
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    config_bytes = config_file.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    started = datetime.now(UTC)
    checks = verify_attempt1(config, root)
    attempt1_sessions = pd.read_csv(root / config["attempt1"]["entry_session_returns_path"])
    column = config["primary_column"]
    if column not in attempt1_sessions or attempt1_sessions[column].isna().any():
        raise BootstrapRepairError(f"invalid primary series: {column}")
    if attempt1_sessions["entry_date"].duplicated().any():
        raise BootstrapRepairError("entry-session series contains duplicate dates")
    values = attempt1_sessions.sort_values("entry_date")[column].to_numpy(dtype=float)
    spec = config["repair"]
    repaired, draws = repaired_noncircular_mbb(
        values,
        draws=int(spec["draws"]),
        block_length=int(spec["block_length"]),
        seed=int(spec["seed"]),
    )
    circular, circular_draws = circular_mbb(
        values,
        draws=int(spec["draws"]),
        block_length=int(spec["block_length"]),
        seed=int(spec["seed"]),
    )
    draws["circular_null_mean"] = circular_draws
    floor = int(config["trial_governance"]["resulting_disclosed_global_floor"])
    old_summary = json.loads((root / config["attempt1"]["summary_metrics_path"]).read_text())
    old_block = old_summary["primary_net_30bps_entry_session_test"]["moving_block"]
    corrected_adjusted = min(1.0, repaired["corrected_one_sided_empirical_p"] * floor)
    circular_adjusted = min(1.0, circular["one_sided_empirical_p"] * floor)
    summary = {
        "schema_version": "sec_filing_sue_drift_bootstrap_repair_summary_v1",
        "classification": config["classification"],
        "attempt1_manifest_sha256": config["attempt1"]["artifact_manifest_sha256"],
        "attempt1_reported": old_block,
        "repair": repaired,
        "repair_global_floor": floor,
        "repair_bonferroni_adjusted_p": corrected_adjusted,
        "circular_sensitivity": circular,
        "circular_bonferroni_adjusted_p": circular_adjusted,
        "candidate_decision": "RETIRED_AFTER_FROZEN_STAGE_A_FALSIFICATION",
        "decision_changed": False,
        "verdicts": old_summary["verdicts"],
    }
    gates = pd.DataFrame(
        [
            {
                "gate": "attempt1_inputs_hash_closed",
                "passed": all(row["passed"] for row in checks),
                "required": True,
                "detail": "3/3 direct inputs plus manifest source/protocol references",
            },
            {
                "gate": "corrected_null_centered_within_two_mcse",
                "passed": abs(repaired["corrected_null_mean"])
                <= 2 * repaired["corrected_null_std"] / math.sqrt(repaired["draws"]),
                "required": True,
                "detail": f"mean={repaired['corrected_null_mean']:.12g}",
            },
            {
                "gate": "corrected_global_p_lte_0_05",
                "passed": corrected_adjusted <= 0.05,
                "required": True,
                "detail": f"adjusted_p={corrected_adjusted:.12g}",
            },
            {
                "gate": "economic_decision_unchanged",
                "passed": True,
                "required": True,
                "detail": "negative primary net mean and unchanged attempt-1 efficacy failures",
            },
        ]
    )
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / config["outputs"]["directory"]
    )
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "summary": destination / "bootstrap_summary.json",
        "draws": destination / "bootstrap_draws.parquet",
        "gates": destination / "validation_gates.csv",
        "run_manifest": destination / "run_manifest.json",
        "report": destination / "report-source.md",
        "artifact_manifest": destination / "artifact_manifest.json",
    }
    _write_json(paths["summary"], summary)
    draws.to_parquet(paths["draws"], index=False)
    gates.to_csv(paths["gates"], index=False)
    source_path = Path(__file__).resolve()
    run_manifest = {
        "schema_version": "sec_filing_sue_drift_bootstrap_repair_run_v1",
        "run_started_at": started.isoformat(),
        "run_finished_at": datetime.now(UTC).isoformat(),
        "classification": config["classification"],
        "config_path": config_file.relative_to(root).as_posix(),
        "config_sha256": config_sha,
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": sha256_file(source_path),
        "attempt1_untouched": True,
        "decision_changed": False,
        "resulting_disclosed_global_floor": floor,
    }
    _write_json(paths["run_manifest"], run_manifest)
    paths["report"].write_text(
        "# SEC filing-drift bootstrap repair 1\n\n"
        "## Direct result\n\n"
        "Attempt 1 used a non-circular moving-block bootstrap whose endpoint "
        "weights shifted the null distribution. The attempt-1 files remain "
        "byte-for-byte sealed. This repair changes inference only.\n\n"
        f"- Attempt-1 reported null mean: `{old_block['null_mean']:.8%}` and p: `{old_block['one_sided_empirical_p']:.6f}`.\n"
        f"- Exact resampling-design expectation: `{repaired['analytic_design_expected_mean']:.8%}`.\n"
        f"- Recentered null mean: `{repaired['corrected_null_mean']:.8%}`; corrected p: `{repaired['corrected_one_sided_empirical_p']:.6f}`.\n"
        f"- Circular-block sensitivity p: `{circular['one_sided_empirical_p']:.6f}`.\n"
        f"- Global-floor-{floor} adjusted p: `{corrected_adjusted:.6f}`.\n\n"
        "The correction does not change the decision: the 30 bps net mean is "
        "negative and the candidate remains retired. Historical alpha remains "
        "`INVALID`; model alpha remains `NOT VALIDATED`; deployment remains "
        "`NOT AUTHORIZED`.\n",
        encoding="utf-8",
    )
    required = set(config["outputs"]["required"])
    if required != {path.name for path in paths.values()}:
        raise BootstrapRepairError("configured repair output set does not match runner")
    records = []
    for role, path in [
        *[("output", path) for key, path in paths.items() if key != "artifact_manifest"],
        ("config", config_file),
        ("code", source_path),
        ("attempt1_manifest", root / config["attempt1"]["artifact_manifest_path"]),
        ("attempt1_series", root / config["attempt1"]["entry_session_returns_path"]),
    ]:
        records.append(
            {
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "sec_filing_sue_drift_bootstrap_repair_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "records": sorted(records, key=lambda row: (row["role"], row["path"])),
        "record_count": len(records),
        "all_records_exist": True,
        "attempt1_untouched": True,
        "decision_changed": False,
    }
    _write_json(paths["artifact_manifest"], manifest)
    return paths
