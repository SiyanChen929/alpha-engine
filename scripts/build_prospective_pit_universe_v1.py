#!/usr/bin/env python3
"""Freeze the first prospective public-data equity cohort."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import yaml

from alpha_factor_engine.contracts import sha256_file
from alpha_factor_engine.data.prospective_universe import (
    ProspectiveUniverseRules,
    build_artifact_manifest,
    build_prospective_universe,
    verify_artifact_manifest,
    verify_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact already exists with different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    snapshot_config = config["snapshot"]
    inputs = config["inputs"]
    rules = ProspectiveUniverseRules(
        **{
            **config["screens"],
            "allowed_exchange_codes": tuple(config["screens"]["allowed_exchange_codes"]),
        }
    )
    snapshot, candidates, sources, qa = build_prospective_universe(
        root=ROOT,
        sec_cache=inputs["sec_cache"],
        yahoo_cache_dir=inputs["yahoo_cache_dir"],
        yahoo_start=inputs["yahoo_start"],
        yahoo_end=inputs["yahoo_end"],
        cutoff_date=snapshot_config["cutoff_date"],
        freeze_at=snapshot_config["freeze_at"],
        rules=rules,
    )
    cohort_id = str(snapshot["cohort_id"].iloc[0])
    suffix = cohort_id.split(":", 1)[1][:12]
    frozen_key = pd.Timestamp(snapshot_config["freeze_at"]).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts" / snapshot_config["name"] / f"{frozen_key}_{suffix}"
    output.mkdir(parents=True, exist_ok=True)

    qa_checks = pd.DataFrame(qa["checks"])
    source_verification = verify_source_manifest(sources, root=ROOT)
    qa["source_manifest_verification"] = source_verification
    qa["config_sha256"] = sha256_file(config_path)
    qa["runtime_access"] = {
        "SEC_USER_AGENT_set": bool(os.getenv("SEC_USER_AGENT")),
        "MARKETDATA_TOKEN_set": bool(
            os.getenv("MARKETDATA_TOKEN") or os.getenv("MARKETDATA_API_KEY")
        ),
        "live_refresh_performed": False,
        "reason": "local immutable raw caches used; live refresh requires authorized provider credentials",
    }

    report = f"""# Prospective PIT universe snapshot v1

## Outcome

- Cohort: `{cohort_id}`
- Frozen at: `{snapshot_config['freeze_at']}`
- Price cutoff: `{snapshot_config['cutoff_date']}`
- Members: **{len(snapshot):,}** from {len(candidates):,} ordered SEC candidates
- Raw source records rehashed: **{source_verification['verified_records']:,}/{source_verification['records']:,}**
- Missing candidate raw caches: **{qa['raw_candidate_caches_missing']:,}**; these rows are retained in the rejection audit
- Snapshot integrity: **{qa['snapshot_integrity_status']}**
- Prospective monitoring after the freeze: **{qa['prospective_monitoring_status']}**
- Historical alpha/backtest claim: **{qa['historical_backtest_status']}**
- Deployment claim: **{qa['deployment_status']}**

This is an immutable, as-known-from-local-cache cohort for observations made after
the freeze.  Membership uses SEC issuer/ticker mappings plus Yahoo metadata and
price/ADV screens through the completed 2026-08-31 session.  No return label,
future survival field, or post-cutoff bar enters selection.

## Information clock

`price_source_effective_at` is the last completed session close.  Yahoo prices
become usable 30 minutes later under the engine's conservative daily-data rule.
The raw SEC mapping has no publication timestamp in its payload, so the combined
`source_available_at`, `membership_available_at`, and `ingested_at` are fixed to
the later snapshot freeze.  `identity_source_effective_at` remains empty with an
explicit unavailable status; CIK/ticker/exchange are usable only prospectively
from `identity_source_available_at`.  The cohort may only be consumed at decisions
at or after that time.

## Fail-closed limits

- CIK identifies an issuer, not a listed security.
- `snapshot_security_key` binds CIK + ticker + exchange only inside this cohort;
  it is not a permanent ID.
- Share class is `UNKNOWN`; public inputs do not establish CRSP-style share code.
- The snapshot contains no earlier inactive names or delisting returns.
- Therefore every row has `historical_backtest_eligible=false` and
  `deployable_eligible=false`.
- Current exchange metadata is from the retained Yahoo payload.  An authoritative
  exchange/security-master reconciliation remains open.
- The local SEC mapping was already cached.  Live SEC refresh was skipped because
  no authorized `SEC_USER_AGENT` is configured; MarketData stock data is absent
  and no token is configured in this runtime.

This artifact does **not** claim a survivorship-free historical universe.  Its
purpose is to stop the bias from growing: future monthly cohorts can now be
appended, reconciled, and evaluated prospectively without rebuilding membership
from future winners.

## Files

- `universe_snapshot.csv`: selected prospective members and clocks
- `candidate_audit.csv`: all candidates, including every explicit rejection
- `source_manifest.csv`: compressed and decompressed SHA-256 for every raw input
- `qa_checks.csv` / `qa_summary.json`: integrity and blocked-claim gates
- `runtime_environment.json`: interpreter and data-library versions
- `artifact_manifest.json`: byte-level code/config/input/output evidence
"""

    runtime = {
        "captured_at": snapshot_config["freeze_at"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "pyyaml": yaml.__version__,
    }

    paths = {
        "snapshot": output / "universe_snapshot.csv",
        "candidates": output / "candidate_audit.csv",
        "sources": output / "source_manifest.csv",
        "checks": output / "qa_checks.csv",
        "qa": output / "qa_summary.json",
        "report": output / "report-source.md",
        "runtime": output / "runtime_environment.json",
    }
    _write_immutable(paths["snapshot"], _csv_bytes(snapshot))
    _write_immutable(paths["candidates"], _csv_bytes(candidates))
    _write_immutable(paths["sources"], _csv_bytes(sources))
    _write_immutable(paths["checks"], _csv_bytes(qa_checks))
    _write_immutable(paths["qa"], _json_bytes(qa))
    _write_immutable(paths["report"], report.encode("utf-8"))
    _write_immutable(paths["runtime"], _json_bytes(runtime))

    manifest = build_artifact_manifest(
        root=ROOT,
        cohort_id=cohort_id,
        freeze_at=snapshot_config["freeze_at"],
        source_root_sha256=str(snapshot["source_root_sha256"].iloc[0]),
        inputs=[
            config_path,
            ROOT / inputs["sec_cache"],
            ROOT / "src/alpha_factor_engine/data/prospective_universe.py",
            ROOT / "src/alpha_factor_engine/data/yahoo.py",
            ROOT / "src/alpha_factor_engine/contracts.py",
            ROOT / "pyproject.toml",
            Path(__file__).resolve(),
        ],
        outputs=paths.values(),
    )
    manifest_path = output / "artifact_manifest.json"
    _write_immutable(manifest_path, _json_bytes(manifest))
    manifest_verification = verify_artifact_manifest(manifest, root=ROOT)
    if not source_verification["passed"] or not manifest_verification["passed"]:
        raise RuntimeError(
            f"evidence verification failed: source={source_verification}; artifact={manifest_verification}"
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "cohort_id": cohort_id,
                "members": len(snapshot),
                "candidate_rows": len(candidates),
                "source_records": len(sources),
                "artifact_records_verified": manifest_verification["checked_records"],
                "historical_backtest_status": qa["historical_backtest_status"],
            },
            indent=2,
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/prospective_pit_universe_v1.yml"),
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
