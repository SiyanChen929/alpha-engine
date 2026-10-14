from dataclasses import FrozenInstanceError
import json

import pytest

from alpha_factor_engine.research.governance import (
    ExperimentTrialLedger,
    ForecastRecord,
    GovernanceViolation,
    ImmutableExperimentSpec,
    PromotionThresholds,
    SealedForecastLedger,
    TrialRecord,
    append_forecast_records,
    append_trial_record,
    create_or_verify_preregistration,
    create_or_verify_trial_record,
    evaluate_production_promotion,
    initialize_forecast_ledger,
    immutable_spec_hash,
    load_forecast_ledger,
    load_preregistration,
    load_trial_ledger,
    load_trial_record,
    write_trial_ledger_atomic,
)


def _experiment(*, opened_period: bool = False, suffix: str = "base") -> ImmutableExperimentSpec:
    return ImmutableExperimentSpec.create(
        experiment_id=f"momentum-primary-{suffix}",
        specification={
            "strategy": {"score": "momentum_12_1", "top_fraction": 0.20},
            "cost_bps": 15,
            "features": ["momentum_12_1"],
        },
        registered_at="2026-09-01T12:00:00Z",
        first_decision_at="2026-09-30T20:00:00Z",
        opened_period=opened_period,
    )


def _trial(*, opened_period: bool = False, suffix: str = "base") -> TrialRecord:
    return TrialRecord.create(
        trial_id=f"trial-{suffix}",
        experiment=_experiment(opened_period=opened_period, suffix=suffix),
    )


def _forecast(
    trial: TrialRecord,
    *,
    forecast_id: str,
    security_id: str,
    decision_at: str,
    created_at: str,
    score: float = 0.5,
) -> ForecastRecord:
    return ForecastRecord.create(
        forecast_id=forecast_id,
        trial_id=trial.trial_id,
        spec_sha256=trial.spec_sha256,
        security_id=security_id,
        decision_at=decision_at,
        created_at=created_at,
        score=score,
        payload={"model_artifact_sha256": "a" * 64},
    )


def test_spec_hash_is_stable_and_original_mutation_cannot_change_frozen_spec():
    mutable = {
        "features": ["momentum_12_1", "roe"],
        "model": {"trees": 300, "learning_rate": 0.04},
    }
    reordered = {
        "model": {"learning_rate": 0.04, "trees": 300},
        "features": ["momentum_12_1", "roe"],
    }
    spec = ImmutableExperimentSpec.create(
        experiment_id="stable-hash",
        specification=mutable,
        registered_at="2026-09-01T12:00:00Z",
        first_decision_at="2026-09-30T20:00:00Z",
    )

    assert spec.spec_sha256 == immutable_spec_hash(reordered)
    mutable["features"].append("future_feature")
    assert spec.to_specification()["features"] == ["momentum_12_1", "roe"]
    with pytest.raises(GovernanceViolation, match="differs from preregistration"):
        spec.verify(mutable)


def test_unopened_experiment_must_be_registered_before_first_decision():
    with pytest.raises(GovernanceViolation, match="marked opened_period"):
        ImmutableExperimentSpec.create(
            experiment_id="late-registration",
            specification={"score": "momentum_12_1"},
            registered_at="2026-10-01T00:00:00Z",
            first_decision_at="2026-09-30T20:00:00Z",
            opened_period=False,
        )

    opened = ImmutableExperimentSpec.create(
        experiment_id="late-registration",
        specification={"score": "momentum_12_1"},
        registered_at="2026-10-01T00:00:00Z",
        first_decision_at="2026-09-30T20:00:00Z",
        opened_period=True,
    )
    assert opened.opened_period is True


def test_trial_ledger_is_immutable_append_only_and_records_opened_trials():
    first = _trial(suffix="one")
    second = _trial(opened_period=True, suffix="two")
    empty = ExperimentTrialLedger()
    one = empty.append(first)
    two = one.append(second)

    assert empty.trials == ()
    assert len(one.trials) == 1
    assert len(two.trials) == 2
    assert two.opened_trial_ids == ("trial-two",)
    assert one.ledger_sha256 != two.ledger_sha256
    with pytest.raises(FrozenInstanceError):
        first.purpose = "mutated"
    with pytest.raises(GovernanceViolation, match="duplicate trial_id"):
        two.append(second)


def test_forecast_ledger_enforces_creation_time_and_monotonic_decisions():
    trial = _trial()
    with pytest.raises(GovernanceViolation, match="created before"):
        _forecast(
            trial,
            forecast_id="invalid-time",
            security_id="AAPL",
            created_at="2026-09-30T20:00:00Z",
            decision_at="2026-09-30T20:00:00Z",
        )

    ledger = SealedForecastLedger.for_trial(trial).append(
        [
            _forecast(
                trial,
                forecast_id="sep-aapl",
                security_id="AAPL",
                created_at="2026-09-30T19:59:00Z",
                decision_at="2026-09-30T20:00:00Z",
            ),
            _forecast(
                trial,
                forecast_id="sep-msft",
                security_id="MSFT",
                created_at="2026-09-30T19:59:00Z",
                decision_at="2026-09-30T20:00:00Z",
            ),
        ]
    )
    assert ledger.decision_count == 1

    later = ledger.append(
        [
            _forecast(
                trial,
                forecast_id="oct-aapl",
                security_id="AAPL",
                created_at="2026-10-30T19:59:00Z",
                decision_at="2026-10-30T20:00:00Z",
            )
        ]
    )
    assert later.decision_count == 2
    with pytest.raises(GovernanceViolation, match="monotonic"):
        later.append(
            [
                _forecast(
                    trial,
                    forecast_id="backfill",
                    security_id="NVDA",
                    created_at="2026-10-01T00:00:00Z",
                    decision_at="2026-10-15T20:00:00Z",
                )
            ]
        )


def test_forecast_append_rejects_mutation_deletion_and_duplicate_keys():
    trial = _trial()
    first = _forecast(
        trial,
        forecast_id="sep-aapl",
        security_id="AAPL",
        created_at="2026-09-30T19:59:00Z",
        decision_at="2026-09-30T20:00:00Z",
    )
    ledger = SealedForecastLedger.for_trial(trial).append([first])
    mutated = _forecast(
        trial,
        forecast_id="sep-aapl",
        security_id="AAPL",
        created_at="2026-09-30T19:59:00Z",
        decision_at="2026-09-30T20:00:00Z",
        score=0.9,
    )
    mutation_candidate = SealedForecastLedger(
        trial_id=trial.trial_id,
        spec_sha256=trial.spec_sha256,
        registered_at=trial.registered_at,
        first_decision_at=trial.first_decision_at,
        records=(mutated,),
    )
    with pytest.raises(GovernanceViolation, match="mutate or reorder"):
        ledger.validate_append(mutation_candidate)
    with pytest.raises(GovernanceViolation, match="delete"):
        ledger.validate_append(SealedForecastLedger.for_trial(trial))
    with pytest.raises(GovernanceViolation, match="duplicate forecast decision/security"):
        ledger.append(
            [
                _forecast(
                    trial,
                    forecast_id="sep-aapl-duplicate",
                    security_id="AAPL",
                    created_at="2026-09-30T19:59:30Z",
                    decision_at="2026-09-30T20:00:00Z",
                )
            ]
        )


def _promotion_ledger(trial: TrialRecord) -> SealedForecastLedger:
    return SealedForecastLedger.for_trial(trial).append(
        [
            _forecast(
                trial,
                forecast_id="sep-aapl",
                security_id="AAPL",
                created_at="2026-09-30T19:59:00Z",
                decision_at="2026-09-30T20:00:00Z",
            ),
            _forecast(
                trial,
                forecast_id="oct-aapl",
                security_id="AAPL",
                created_at="2026-10-30T19:59:00Z",
                decision_at="2026-10-30T20:00:00Z",
            ),
        ]
    )


def _passing_metrics() -> dict[str, float]:
    return {
        "sharpe": 0.8,
        "newey_west_tstat": 2.5,
        "bootstrap_sharpe_p05": 0.1,
        "max_drawdown": -0.1,
        "incremental_annualized_return": 0.02,
        "incremental_newey_west_tstat": 2.2,
    }


def _thresholds() -> PromotionThresholds:
    return PromotionThresholds(
        minimum_sealed_decisions=2,
        minimum_incremental_annualized_return=0.0,
        minimum_incremental_newey_west_tstat=2.0,
        required_production_checks=("survivorship_free", "execution_grade_quotes"),
    )


def test_production_promotion_passes_only_complete_unopened_evidence():
    trial = _trial()
    decision = evaluate_production_promotion(
        trial=trial,
        forecasts=_promotion_ledger(trial),
        metrics=_passing_metrics(),
        production_checks={"survivorship_free": True, "execution_grade_quotes": True},
        thresholds=_thresholds(),
    )

    assert decision.passed is True
    assert decision.action == "PROMOTE_TO_PRODUCTION"
    assert all(check.passed for check in decision.checks)


def test_production_promotion_fails_opened_period_missing_data_and_metrics():
    trial = _trial(opened_period=True)
    metrics = _passing_metrics()
    metrics.pop("bootstrap_sharpe_p05")
    decision = evaluate_production_promotion(
        trial=trial,
        forecasts=_promotion_ledger(trial),
        metrics=metrics,
        production_checks={"survivorship_free": True},
        thresholds=_thresholds(),
    )
    checks = {check.name: check for check in decision.checks}

    assert decision.passed is False
    assert decision.action == "BLOCK_PRODUCTION"
    assert checks["period_was_sealed"].passed is False
    assert checks["bootstrap_sharpe_p05"].passed is False
    assert checks["execution_grade_quotes"].passed is False


def test_preregistration_create_if_absent_is_idempotent_and_hash_verified(tmp_path):
    path = tmp_path / "preregistration.json"
    experiment = _experiment()
    created = create_or_verify_preregistration(path, experiment)
    original_bytes = path.read_bytes()

    verified = create_or_verify_preregistration(path, experiment)
    assert created == verified == load_preregistration(path)
    assert path.read_bytes() == original_bytes

    different = ImmutableExperimentSpec.create(
        experiment_id=experiment.experiment_id,
        specification={"strategy": {"score": "momentum_6_1"}},
        registered_at=experiment.registered_at,
        first_decision_at=experiment.first_decision_at,
    )
    with pytest.raises(GovernanceViolation, match="differs from requested"):
        create_or_verify_preregistration(path, different)
    assert path.read_bytes() == original_bytes

    tampered = json.loads(path.read_text())
    tampered["record"]["experiment_id"] = "tampered"
    path.write_text(json.dumps(tampered))
    with pytest.raises(GovernanceViolation, match="envelope hash mismatch"):
        load_preregistration(path)

    deep_tampered = json.loads(original_bytes)
    deep_tampered["record"]["spec_sha256"] = "0" * 64
    envelope_core = {
        "schema": deep_tampered["schema"],
        "version": deep_tampered["version"],
        "record": deep_tampered["record"],
    }
    deep_tampered["envelope_sha256"] = immutable_spec_hash(envelope_core)
    path.write_text(json.dumps(deep_tampered))
    with pytest.raises(GovernanceViolation, match="specification hash mismatch"):
        load_preregistration(path)


def test_trial_record_and_atomic_trial_ledger_round_trip_and_concurrency_guard(tmp_path):
    single_path = tmp_path / "trial.json"
    first = _trial(suffix="one")
    assert create_or_verify_trial_record(single_path, first) == first
    assert load_trial_record(single_path) == first

    ledger_path = tmp_path / "trials.json"
    one = append_trial_record(ledger_path, first)
    second = _trial(suffix="two")
    two = append_trial_record(
        ledger_path,
        second,
        expected_ledger_sha256=one.ledger_sha256,
    )
    assert load_trial_ledger(ledger_path).to_dict() == two.to_dict()
    before_stale_append = ledger_path.read_bytes()

    with pytest.raises(GovernanceViolation, match="cannot delete"):
        write_trial_ledger_atomic(
            ledger_path,
            one,
            expected_ledger_sha256=two.ledger_sha256,
        )
    assert ledger_path.read_bytes() == before_stale_append

    third = _trial(suffix="three")
    with pytest.raises(GovernanceViolation, match="changed since"):
        append_trial_record(
            ledger_path,
            third,
            expected_ledger_sha256=ExperimentTrialLedger().ledger_sha256,
        )
    assert ledger_path.read_bytes() == before_stale_append

    tampered = json.loads(ledger_path.read_text())
    tampered["record"]["trials"][0]["purpose"] = "hidden-retune"
    ledger_path.write_text(json.dumps(tampered))
    with pytest.raises(GovernanceViolation, match="envelope hash mismatch"):
        load_trial_ledger(ledger_path)


def test_forecast_log_appends_new_batches_without_rewriting_sealed_prefix(tmp_path):
    trial = _trial()
    path = tmp_path / "sealed_forecasts.jsonl"
    empty = SealedForecastLedger.for_trial(trial)
    assert initialize_forecast_ledger(path, empty) == empty
    header_bytes = path.read_bytes()

    first = _forecast(
        trial,
        forecast_id="sep-aapl",
        security_id="AAPL",
        created_at="2026-09-30T19:59:00Z",
        decision_at="2026-09-30T20:00:00Z",
    )
    one = append_forecast_records(
        path,
        [first],
        expected_ledger_sha256=empty.ledger_sha256,
    )
    first_append_bytes = path.read_bytes()
    assert first_append_bytes.startswith(header_bytes)

    second = _forecast(
        trial,
        forecast_id="oct-aapl",
        security_id="AAPL",
        created_at="2026-10-30T19:59:00Z",
        decision_at="2026-10-30T20:00:00Z",
    )
    two = append_forecast_records(
        path,
        [second],
        expected_ledger_sha256=one.ledger_sha256,
    )
    second_append_bytes = path.read_bytes()
    assert second_append_bytes.startswith(first_append_bytes)
    assert load_forecast_ledger(path).to_dict() == two.to_dict()
    assert SealedForecastLedger.from_dict(two.to_dict()) == two

    with pytest.raises(GovernanceViolation, match="changed since"):
        append_forecast_records(
            path,
            [
                _forecast(
                    trial,
                    forecast_id="nov-aapl",
                    security_id="AAPL",
                    created_at="2026-11-30T19:59:00Z",
                    decision_at="2026-11-30T20:00:00Z",
                )
            ],
            expected_ledger_sha256=empty.ledger_sha256,
        )
    assert path.read_bytes() == second_append_bytes


def test_forecast_log_load_rejects_tampered_and_torn_append_records(tmp_path):
    trial = _trial()
    original = tmp_path / "original.jsonl"
    initialize_forecast_ledger(original, SealedForecastLedger.for_trial(trial))
    append_forecast_records(
        original,
        [
            _forecast(
                trial,
                forecast_id="sep-aapl",
                security_id="AAPL",
                created_at="2026-09-30T19:59:00Z",
                decision_at="2026-09-30T20:00:00Z",
            )
        ],
    )

    lines = original.read_text().splitlines()
    batch = json.loads(lines[1])
    batch["records"][0]["score"] = 99.0
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(lines[0] + "\n" + json.dumps(batch) + "\n")
    with pytest.raises(GovernanceViolation, match="batch hash mismatch"):
        load_forecast_ledger(tampered)

    batch_core = {key: value for key, value in batch.items() if key != "batch_sha256"}
    batch["batch_sha256"] = immutable_spec_hash(batch_core)
    deep_tampered = tmp_path / "deep-tampered.jsonl"
    deep_tampered.write_text(lines[0] + "\n" + json.dumps(batch) + "\n")
    with pytest.raises(GovernanceViolation, match="forecast record hash mismatch"):
        load_forecast_ledger(deep_tampered)

    torn = tmp_path / "torn.jsonl"
    torn.write_bytes(original.read_bytes() + b'{"kind":"append"')
    with pytest.raises(GovernanceViolation, match="torn final record"):
        load_forecast_ledger(torn)
