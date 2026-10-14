"""Fail-closed experiment governance for prospective alpha research.

This module is intentionally independent of the active research pipeline.  It
provides immutable preregistration records, append-only trial and forecast
ledgers, and a production-promotion gate that can be integrated later without
changing historical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import fcntl


class GovernanceViolation(ValueError):
    """Raised when an experiment would lose preregistration or seal integrity."""


def _utc_datetime(value: str | datetime, *, field_name: str) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise GovernanceViolation(f"{field_name} is not a valid timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise GovernanceViolation(f"{field_name} must be an ISO timestamp or datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GovernanceViolation(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _utc_iso(value: str | datetime, *, field_name: str) -> str:
    return _utc_datetime(value, field_name=field_name).isoformat()


def _canonical_value(value: Any) -> Any:
    """Convert supported values to an unambiguous JSON representation."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise GovernanceViolation("canonical specifications cannot contain NaN or infinity")
        return 0.0 if number == 0 else number
    if isinstance(value, datetime):
        return _utc_iso(value, field_name="datetime value")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GovernanceViolation("canonical specification keys must be strings")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise GovernanceViolation(f"unsupported canonical specification type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for immutable research fingerprints."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def immutable_spec_hash(value: Any) -> str:
    """Hash a specification after deterministic, type-safe canonicalization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return immutable_spec_hash(payload)


def _strict_json_loads(content: str, *, source: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise GovernanceViolation(f"{source} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise GovernanceViolation(f"{source} contains non-finite JSON value {value}")

    try:
        decoded = json.loads(
            content,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except GovernanceViolation:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GovernanceViolation(f"{source} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise GovernanceViolation(f"{source} must contain a JSON object")
    return decoded


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GovernanceViolation(f"{name} keys differ; missing={missing}, extra={extra}")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("short write while persisting governance record")
        offset += written


def _exclusive_create(path: Path, content: bytes) -> bool:
    """Publish complete bytes only if the destination does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".create", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            # Same-directory hard linking gives create-if-absent semantics
            # without exposing a partially written destination.
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _append_one_record(path: Path, content: bytes) -> int:
    """Append one transaction record and roll back a detected short write."""

    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    original_size = os.lseek(descriptor, 0, os.SEEK_END)
    try:
        written = os.write(descriptor, content)
        if written != len(content):
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
            raise OSError("short append while persisting sealed forecasts")
        os.fsync(descriptor)
    except BaseException:
        current_size = os.lseek(descriptor, 0, os.SEEK_END)
        if current_size != original_size:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
        raise
    finally:
        os.close(descriptor)
    return original_size


@contextmanager
def _persistence_lock(path: Path, *, exclusive: bool = True):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GovernanceViolation(f"cannot read governance JSON {path}") from exc
    return _strict_json_loads(content, source=str(path))


@dataclass(frozen=True)
class ImmutableExperimentSpec:
    """A preregistered specification stored as canonical JSON, not a mutable dict."""

    experiment_id: str
    registered_at: str
    first_decision_at: str
    opened_period: bool
    canonical_spec: str
    spec_sha256: str

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        specification: Mapping[str, Any],
        registered_at: str | datetime,
        first_decision_at: str | datetime,
        opened_period: bool = False,
    ) -> "ImmutableExperimentSpec":
        if not experiment_id.strip():
            raise GovernanceViolation("experiment_id cannot be empty")
        registered = _utc_iso(registered_at, field_name="registered_at")
        first_decision = _utc_iso(first_decision_at, field_name="first_decision_at")
        if (
            _utc_datetime(registered, field_name="registered_at")
            >= _utc_datetime(first_decision, field_name="first_decision_at")
            and not opened_period
        ):
            raise GovernanceViolation(
                "an experiment registered at or after its first decision must be marked opened_period"
            )
        frozen_json = canonical_json(specification)
        digest = hashlib.sha256(frozen_json.encode("utf-8")).hexdigest()
        return cls(
            experiment_id=experiment_id,
            registered_at=registered,
            first_decision_at=first_decision,
            opened_period=bool(opened_period),
            canonical_spec=frozen_json,
            spec_sha256=digest,
        )

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise GovernanceViolation("experiment_id cannot be empty")
        registered = _utc_iso(self.registered_at, field_name="registered_at")
        first_decision = _utc_iso(self.first_decision_at, field_name="first_decision_at")
        if registered != self.registered_at or first_decision != self.first_decision_at:
            raise GovernanceViolation("experiment timestamps must use normalized UTC ISO format")
        try:
            decoded = json.loads(self.canonical_spec)
        except json.JSONDecodeError as exc:
            raise GovernanceViolation("canonical_spec is not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise GovernanceViolation("canonical_spec must contain a JSON object")
        if canonical_json(decoded) != self.canonical_spec:
            raise GovernanceViolation("canonical_spec is not canonical JSON")
        actual_hash = hashlib.sha256(self.canonical_spec.encode("utf-8")).hexdigest()
        if actual_hash != self.spec_sha256:
            raise GovernanceViolation("immutable experiment specification hash mismatch")
        if (
            _utc_datetime(self.registered_at, field_name="registered_at")
            >= _utc_datetime(self.first_decision_at, field_name="first_decision_at")
            and not self.opened_period
        ):
            raise GovernanceViolation("unopened experiment was not registered before its first decision")

    def to_specification(self) -> dict[str, Any]:
        """Return a new object so callers cannot mutate the frozen specification."""

        decoded = json.loads(self.canonical_spec)
        if not isinstance(decoded, dict):
            raise GovernanceViolation("experiment specification must decode to an object")
        return decoded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImmutableExperimentSpec":
        expected = {
            "experiment_id",
            "registered_at",
            "first_decision_at",
            "opened_period",
            "canonical_spec",
            "spec_sha256",
        }
        _exact_keys(payload, expected, name="immutable experiment specification")
        if not isinstance(payload["opened_period"], bool):
            raise GovernanceViolation("opened_period must be a boolean")
        try:
            return cls(**{key: payload[key] for key in expected})
        except GovernanceViolation:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise GovernanceViolation("invalid immutable experiment specification") from exc

    def verify(self, candidate: Mapping[str, Any] | None = None) -> bool:
        self.__post_init__()
        if candidate is not None and immutable_spec_hash(candidate) != self.spec_sha256:
            raise GovernanceViolation("candidate specification differs from preregistration")
        return True


@dataclass(frozen=True)
class TrialRecord:
    """One disclosed experiment trial, including opened-period status."""

    trial_id: str
    experiment_id: str
    spec_sha256: str
    registered_at: str
    first_decision_at: str
    opened_period: bool
    purpose: str
    record_sha256: str

    @classmethod
    def create(
        cls,
        *,
        trial_id: str,
        experiment: ImmutableExperimentSpec,
        purpose: str = "candidate",
    ) -> "TrialRecord":
        if not trial_id.strip():
            raise GovernanceViolation("trial_id cannot be empty")
        if not purpose.strip():
            raise GovernanceViolation("trial purpose cannot be empty")
        payload = {
            "trial_id": trial_id,
            "experiment_id": experiment.experiment_id,
            "spec_sha256": experiment.spec_sha256,
            "registered_at": experiment.registered_at,
            "first_decision_at": experiment.first_decision_at,
            "opened_period": experiment.opened_period,
            "purpose": purpose,
        }
        return cls(**payload, record_sha256=_payload_hash(payload))

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.experiment_id.strip() or not self.purpose.strip():
            raise GovernanceViolation("trial identifiers and purpose cannot be empty")
        _utc_datetime(self.registered_at, field_name="registered_at")
        _utc_datetime(self.first_decision_at, field_name="first_decision_at")
        payload = {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "spec_sha256": self.spec_sha256,
            "registered_at": self.registered_at,
            "first_decision_at": self.first_decision_at,
            "opened_period": self.opened_period,
            "purpose": self.purpose,
        }
        if _payload_hash(payload) != self.record_sha256:
            raise GovernanceViolation("trial record hash mismatch")

    def verify(self) -> bool:
        self.__post_init__()
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialRecord":
        expected = {
            "trial_id",
            "experiment_id",
            "spec_sha256",
            "registered_at",
            "first_decision_at",
            "opened_period",
            "purpose",
            "record_sha256",
        }
        _exact_keys(payload, expected, name="trial record")
        if not isinstance(payload["opened_period"], bool):
            raise GovernanceViolation("opened_period must be a boolean")
        try:
            return cls(**{key: payload[key] for key in expected})
        except GovernanceViolation:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise GovernanceViolation("invalid trial record") from exc


@dataclass(frozen=True)
class ExperimentTrialLedger:
    """An immutable append-only record of every tried experiment specification."""

    trials: tuple[TrialRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trials, tuple):
            raise GovernanceViolation("trial ledger storage must be an immutable tuple")
        trial_ids: set[str] = set()
        prior_registered_at: datetime | None = None
        for trial in self.trials:
            trial.verify()
            if trial.trial_id in trial_ids:
                raise GovernanceViolation(f"duplicate trial_id: {trial.trial_id}")
            trial_ids.add(trial.trial_id)
            registered = _utc_datetime(trial.registered_at, field_name="registered_at")
            if prior_registered_at is not None and registered < prior_registered_at:
                raise GovernanceViolation("trial ledger registration times must be monotonic")
            prior_registered_at = registered

    def append(self, trial: TrialRecord) -> "ExperimentTrialLedger":
        return ExperimentTrialLedger(self.trials + (trial,))

    def validate_append(self, candidate: "ExperimentTrialLedger") -> bool:
        candidate.__post_init__()
        if len(candidate.trials) < len(self.trials):
            raise GovernanceViolation("trial ledger append cannot delete existing trials")
        old_hashes = tuple(trial.record_sha256 for trial in self.trials)
        candidate_prefix = tuple(
            trial.record_sha256 for trial in candidate.trials[: len(self.trials)]
        )
        if candidate_prefix != old_hashes:
            raise GovernanceViolation("trial ledger append attempted to mutate or reorder trials")
        return True

    @property
    def ledger_sha256(self) -> str:
        return immutable_spec_hash([trial.record_sha256 for trial in self.trials])

    @property
    def opened_trial_ids(self) -> tuple[str, ...]:
        return tuple(trial.trial_id for trial in self.trials if trial.opened_period)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials": [trial.to_dict() for trial in self.trials],
            "ledger_sha256": self.ledger_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentTrialLedger":
        _exact_keys(payload, {"trials", "ledger_sha256"}, name="experiment trial ledger")
        raw_trials = payload["trials"]
        if not isinstance(raw_trials, list):
            raise GovernanceViolation("experiment trial ledger trials must be a list")
        trials: list[TrialRecord] = []
        for index, raw_trial in enumerate(raw_trials):
            if not isinstance(raw_trial, Mapping):
                raise GovernanceViolation(f"trial ledger item {index} must be an object")
            trials.append(TrialRecord.from_dict(raw_trial))
        ledger = cls(tuple(trials))
        if payload["ledger_sha256"] != ledger.ledger_sha256:
            raise GovernanceViolation("experiment trial ledger hash mismatch")
        return ledger


@dataclass(frozen=True)
class ForecastRecord:
    """One forecast sealed before its associated decision timestamp."""

    forecast_id: str
    trial_id: str
    spec_sha256: str
    security_id: str
    decision_at: str
    created_at: str
    score: float
    target_weight: float | None
    payload_json: str
    record_sha256: str

    @classmethod
    def create(
        cls,
        *,
        forecast_id: str,
        trial_id: str,
        spec_sha256: str,
        security_id: str,
        decision_at: str | datetime,
        created_at: str | datetime,
        score: float,
        target_weight: float | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> "ForecastRecord":
        decision = _utc_iso(decision_at, field_name="decision_at")
        created = _utc_iso(created_at, field_name="created_at")
        numeric_score = float(score)
        numeric_weight = None if target_weight is None else float(target_weight)
        payload_json = canonical_json(payload or {})
        unsigned = {
            "forecast_id": forecast_id,
            "trial_id": trial_id,
            "spec_sha256": spec_sha256,
            "security_id": security_id,
            "decision_at": decision,
            "created_at": created,
            "score": numeric_score,
            "target_weight": numeric_weight,
            "payload_json": payload_json,
        }
        return cls(**unsigned, record_sha256=_payload_hash(unsigned))

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in [self.forecast_id, self.trial_id, self.spec_sha256, self.security_id]
        ):
            raise GovernanceViolation("forecast identifiers cannot be empty")
        created = _utc_datetime(self.created_at, field_name="created_at")
        decision = _utc_datetime(self.decision_at, field_name="decision_at")
        if created >= decision:
            raise GovernanceViolation("forecast must be created before its decision timestamp")
        if isinstance(self.score, bool) or not isinstance(self.score, Real):
            raise GovernanceViolation("forecast score must be numeric")
        if not math.isfinite(self.score):
            raise GovernanceViolation("forecast score must be finite")
        if self.target_weight is not None:
            if isinstance(self.target_weight, bool) or not isinstance(self.target_weight, Real):
                raise GovernanceViolation("forecast target_weight must be numeric")
            if not math.isfinite(self.target_weight):
                raise GovernanceViolation("forecast target_weight must be finite")
        try:
            decoded_payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise GovernanceViolation("forecast payload_json is invalid") from exc
        if not isinstance(decoded_payload, Mapping):
            raise GovernanceViolation("forecast payload_json must contain an object")
        if canonical_json(decoded_payload) != self.payload_json:
            raise GovernanceViolation("forecast payload_json is not canonical")
        unsigned = {
            "forecast_id": self.forecast_id,
            "trial_id": self.trial_id,
            "spec_sha256": self.spec_sha256,
            "security_id": self.security_id,
            "decision_at": self.decision_at,
            "created_at": self.created_at,
            "score": self.score,
            "target_weight": self.target_weight,
            "payload_json": self.payload_json,
        }
        if _payload_hash(unsigned) != self.record_sha256:
            raise GovernanceViolation("forecast record hash mismatch")

    def verify(self) -> bool:
        self.__post_init__()
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ForecastRecord":
        expected = {
            "forecast_id",
            "trial_id",
            "spec_sha256",
            "security_id",
            "decision_at",
            "created_at",
            "score",
            "target_weight",
            "payload_json",
            "record_sha256",
        }
        _exact_keys(payload, expected, name="forecast record")
        try:
            return cls(**{key: payload[key] for key in expected})
        except GovernanceViolation:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise GovernanceViolation("invalid forecast record") from exc


@dataclass(frozen=True)
class SealedForecastLedger:
    """Immutable forecast ledger whose only supported update is validated append."""

    trial_id: str
    spec_sha256: str
    registered_at: str
    first_decision_at: str
    records: tuple[ForecastRecord, ...] = ()

    @classmethod
    def for_trial(cls, trial: TrialRecord) -> "SealedForecastLedger":
        trial.verify()
        return cls(
            trial_id=trial.trial_id,
            spec_sha256=trial.spec_sha256,
            registered_at=trial.registered_at,
            first_decision_at=trial.first_decision_at,
        )

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in [self.trial_id, self.spec_sha256, self.registered_at, self.first_decision_at]
        ):
            raise GovernanceViolation("forecast ledger identity and timing fields cannot be empty")
        if not isinstance(self.records, tuple):
            raise GovernanceViolation("forecast ledger storage must be an immutable tuple")
        registered = _utc_datetime(self.registered_at, field_name="registered_at")
        first_decision = _utc_datetime(self.first_decision_at, field_name="first_decision_at")
        forecast_ids: set[str] = set()
        forecast_keys: set[tuple[str, str]] = set()
        prior_decision: datetime | None = None
        for record in self.records:
            record.verify()
            if record.trial_id != self.trial_id or record.spec_sha256 != self.spec_sha256:
                raise GovernanceViolation("forecast does not match the ledger trial/specification")
            created = _utc_datetime(record.created_at, field_name="created_at")
            decision = _utc_datetime(record.decision_at, field_name="decision_at")
            if created < registered:
                raise GovernanceViolation("forecast predates trial registration")
            if decision < first_decision:
                raise GovernanceViolation("forecast decision predates the preregistered evaluation period")
            if prior_decision is not None and decision < prior_decision:
                raise GovernanceViolation("forecast decision timestamps must be monotonic")
            prior_decision = decision
            if record.forecast_id in forecast_ids:
                raise GovernanceViolation(f"duplicate forecast_id: {record.forecast_id}")
            forecast_ids.add(record.forecast_id)
            key = (record.decision_at, record.security_id)
            if key in forecast_keys:
                raise GovernanceViolation(f"duplicate forecast decision/security key: {key}")
            forecast_keys.add(key)

    def append(self, records: Iterable[ForecastRecord]) -> "SealedForecastLedger":
        new_records = tuple(records)
        if not new_records:
            return self
        candidate = SealedForecastLedger(
            trial_id=self.trial_id,
            spec_sha256=self.spec_sha256,
            registered_at=self.registered_at,
            first_decision_at=self.first_decision_at,
            records=self.records + new_records,
        )
        self.validate_append(candidate)
        return candidate

    def validate_append(self, candidate: "SealedForecastLedger") -> bool:
        candidate.verify_integrity()
        identity = (self.trial_id, self.spec_sha256, self.registered_at, self.first_decision_at)
        candidate_identity = (
            candidate.trial_id,
            candidate.spec_sha256,
            candidate.registered_at,
            candidate.first_decision_at,
        )
        if candidate_identity != identity:
            raise GovernanceViolation("forecast ledger identity cannot change during append")
        if len(candidate.records) < len(self.records):
            raise GovernanceViolation("forecast append cannot delete sealed records")
        old_hashes = tuple(record.record_sha256 for record in self.records)
        candidate_prefix = tuple(
            record.record_sha256 for record in candidate.records[: len(self.records)]
        )
        if candidate_prefix != old_hashes:
            raise GovernanceViolation("forecast append attempted to mutate or reorder sealed records")
        return True

    def verify_integrity(self) -> bool:
        self.__post_init__()
        return True

    @property
    def decision_count(self) -> int:
        return len({record.decision_at for record in self.records})

    @property
    def ledger_sha256(self) -> str:
        return immutable_spec_hash(
            {
                "trial_id": self.trial_id,
                "spec_sha256": self.spec_sha256,
                "registered_at": self.registered_at,
                "first_decision_at": self.first_decision_at,
                "record_hashes": [record.record_sha256 for record in self.records],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "spec_sha256": self.spec_sha256,
            "registered_at": self.registered_at,
            "first_decision_at": self.first_decision_at,
            "records": [record.to_dict() for record in self.records],
            "ledger_sha256": self.ledger_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SealedForecastLedger":
        expected = {
            "trial_id",
            "spec_sha256",
            "registered_at",
            "first_decision_at",
            "records",
            "ledger_sha256",
        }
        _exact_keys(payload, expected, name="sealed forecast ledger")
        raw_records = payload["records"]
        if not isinstance(raw_records, list):
            raise GovernanceViolation("sealed forecast records must be a list")
        records: list[ForecastRecord] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise GovernanceViolation(f"forecast ledger item {index} must be an object")
            records.append(ForecastRecord.from_dict(raw_record))
        try:
            ledger = cls(
                trial_id=payload["trial_id"],
                spec_sha256=payload["spec_sha256"],
                registered_at=payload["registered_at"],
                first_decision_at=payload["first_decision_at"],
                records=tuple(records),
            )
        except GovernanceViolation:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise GovernanceViolation("invalid sealed forecast ledger") from exc
        if payload["ledger_sha256"] != ledger.ledger_sha256:
            raise GovernanceViolation("sealed forecast ledger hash mismatch")
        return ledger


_PERSISTENCE_VERSION = 1
_SPEC_SCHEMA = "alpha_factor_engine.immutable_experiment_spec"
_TRIAL_SCHEMA = "alpha_factor_engine.trial_record"
_TRIAL_LEDGER_SCHEMA = "alpha_factor_engine.experiment_trial_ledger"
_FORECAST_LOG_SCHEMA = "alpha_factor_engine.sealed_forecast_log"


def _persistence_envelope(schema: str, record: Mapping[str, Any]) -> dict[str, Any]:
    core = {"schema": schema, "version": _PERSISTENCE_VERSION, "record": record}
    return {**core, "envelope_sha256": immutable_spec_hash(core)}


def _verify_persistence_envelope(
    payload: Mapping[str, Any], *, schema: str, source: str
) -> Mapping[str, Any]:
    expected = {"schema", "version", "record", "envelope_sha256"}
    _exact_keys(payload, expected, name=f"{source} envelope")
    if (
        payload["schema"] != schema
        or type(payload["version"]) is not int
        or payload["version"] != _PERSISTENCE_VERSION
    ):
        raise GovernanceViolation(f"{source} has unsupported schema or version")
    if not isinstance(payload["record"], Mapping):
        raise GovernanceViolation(f"{source} record must be an object")
    core = {
        "schema": payload["schema"],
        "version": payload["version"],
        "record": payload["record"],
    }
    if payload["envelope_sha256"] != immutable_spec_hash(core):
        raise GovernanceViolation(f"{source} envelope hash mismatch")
    return payload["record"]


def _write_envelope_atomic(path: Path, *, schema: str, record: Mapping[str, Any]) -> None:
    _atomic_write(path, _json_bytes(_persistence_envelope(schema, record)))


def _create_or_verify_envelope(
    path: Path,
    *,
    schema: str,
    record: Mapping[str, Any],
    loader: Any,
) -> Any:
    expected = _persistence_envelope(schema, record)
    if _exclusive_create(path, _json_bytes(expected)):
        return loader(path)
    loaded = loader(path)
    if loaded.to_dict() != dict(record):
        raise GovernanceViolation(f"existing {schema} differs from requested immutable record")
    return loaded


def create_or_verify_preregistration(
    path: str | Path, experiment: ImmutableExperimentSpec
) -> ImmutableExperimentSpec:
    """Create a preregistration exactly once, or verify an identical existing file."""

    experiment.verify()
    return _create_or_verify_envelope(
        Path(path),
        schema=_SPEC_SCHEMA,
        record=experiment.to_dict(),
        loader=load_preregistration,
    )


def load_preregistration(path: str | Path) -> ImmutableExperimentSpec:
    source = Path(path)
    record = _verify_persistence_envelope(
        _read_json_object(source), schema=_SPEC_SCHEMA, source=str(source)
    )
    return ImmutableExperimentSpec.from_dict(record)


def create_or_verify_trial_record(path: str | Path, trial: TrialRecord) -> TrialRecord:
    """Persist an immutable single-trial record without overwriting an existing one."""

    trial.verify()
    return _create_or_verify_envelope(
        Path(path),
        schema=_TRIAL_SCHEMA,
        record=trial.to_dict(),
        loader=load_trial_record,
    )


def load_trial_record(path: str | Path) -> TrialRecord:
    source = Path(path)
    record = _verify_persistence_envelope(
        _read_json_object(source), schema=_TRIAL_SCHEMA, source=str(source)
    )
    return TrialRecord.from_dict(record)


def _write_trial_ledger_atomic_unlocked(
    path: Path,
    ledger: ExperimentTrialLedger,
    *,
    expected_ledger_sha256: str | None = None,
) -> None:
    ledger.__post_init__()
    if path.exists():
        current = load_trial_ledger(path)
        if (
            expected_ledger_sha256 is not None
            and current.ledger_sha256 != expected_ledger_sha256
        ):
            raise GovernanceViolation("trial ledger changed since the caller's last verified snapshot")
        current.validate_append(ledger)
    elif expected_ledger_sha256 not in (None, ExperimentTrialLedger().ledger_sha256):
        raise GovernanceViolation("trial ledger is absent but caller expected an existing snapshot")
    _write_envelope_atomic(path, schema=_TRIAL_LEDGER_SCHEMA, record=ledger.to_dict())


def write_trial_ledger_atomic(
    path: str | Path,
    ledger: ExperimentTrialLedger,
    *,
    expected_ledger_sha256: str | None = None,
) -> None:
    """Atomically persist only an append extension of the current trial ledger."""

    destination = Path(path)
    with _persistence_lock(destination):
        _write_trial_ledger_atomic_unlocked(
            destination,
            ledger,
            expected_ledger_sha256=expected_ledger_sha256,
        )


def load_trial_ledger(path: str | Path) -> ExperimentTrialLedger:
    source = Path(path)
    record = _verify_persistence_envelope(
        _read_json_object(source), schema=_TRIAL_LEDGER_SCHEMA, source=str(source)
    )
    return ExperimentTrialLedger.from_dict(record)


def append_trial_record(
    path: str | Path,
    trial: TrialRecord,
    *,
    expected_ledger_sha256: str | None = None,
) -> ExperimentTrialLedger:
    """Serialize concurrent trial additions and atomically publish the new ledger."""

    destination = Path(path)
    trial.verify()
    with _persistence_lock(destination):
        current = load_trial_ledger(destination) if destination.exists() else ExperimentTrialLedger()
        if (
            expected_ledger_sha256 is not None
            and current.ledger_sha256 != expected_ledger_sha256
        ):
            raise GovernanceViolation("trial ledger changed since the caller's last verified snapshot")
        updated = current.append(trial)
        _write_trial_ledger_atomic_unlocked(
            destination,
            updated,
            expected_ledger_sha256=current.ledger_sha256,
        )
        verified = load_trial_ledger(destination)
        if verified.ledger_sha256 != updated.ledger_sha256:
            raise GovernanceViolation("atomically written trial ledger failed verification")
        return verified


def _forecast_log_header(ledger: SealedForecastLedger) -> dict[str, Any]:
    if ledger.records:
        raise GovernanceViolation("forecast log initialization requires an empty ledger")
    core = {
        "schema": _FORECAST_LOG_SCHEMA,
        "version": _PERSISTENCE_VERSION,
        "kind": "header",
        "ledger": ledger.to_dict(),
    }
    return {**core, "header_sha256": immutable_spec_hash(core)}


def _forecast_batch(
    *,
    sequence: int,
    previous: SealedForecastLedger,
    records: tuple[ForecastRecord, ...],
    resulting: SealedForecastLedger,
) -> dict[str, Any]:
    core = {
        "schema": _FORECAST_LOG_SCHEMA,
        "version": _PERSISTENCE_VERSION,
        "kind": "append",
        "sequence": sequence,
        "previous_ledger_sha256": previous.ledger_sha256,
        "records": [record.to_dict() for record in records],
        "resulting_ledger_sha256": resulting.ledger_sha256,
    }
    return {**core, "batch_sha256": immutable_spec_hash(core)}


def _load_forecast_log_unlocked(path: Path) -> tuple[SealedForecastLedger, int]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GovernanceViolation(f"cannot read sealed forecast log {path}") from exc
    if not content or not content.endswith("\n"):
        raise GovernanceViolation("sealed forecast log is empty or has a torn final record")
    lines = content.splitlines()
    header = _strict_json_loads(lines[0], source=f"{path}:1")
    header_keys = {"schema", "version", "kind", "ledger", "header_sha256"}
    _exact_keys(header, header_keys, name="sealed forecast log header")
    if (
        header["schema"] != _FORECAST_LOG_SCHEMA
        or type(header["version"]) is not int
        or header["version"] != _PERSISTENCE_VERSION
        or header["kind"] != "header"
    ):
        raise GovernanceViolation("sealed forecast log has unsupported header")
    if not isinstance(header["ledger"], Mapping):
        raise GovernanceViolation("sealed forecast log header ledger must be an object")
    header_core = {
        "schema": header["schema"],
        "version": header["version"],
        "kind": header["kind"],
        "ledger": header["ledger"],
    }
    if header["header_sha256"] != immutable_spec_hash(header_core):
        raise GovernanceViolation("sealed forecast log header hash mismatch")
    current = SealedForecastLedger.from_dict(header["ledger"])
    if current.records:
        raise GovernanceViolation("sealed forecast log header cannot contain forecast records")

    for expected_sequence, line in enumerate(lines[1:], start=1):
        batch = _strict_json_loads(line, source=f"{path}:{expected_sequence + 1}")
        batch_keys = {
            "schema",
            "version",
            "kind",
            "sequence",
            "previous_ledger_sha256",
            "records",
            "resulting_ledger_sha256",
            "batch_sha256",
        }
        _exact_keys(batch, batch_keys, name="sealed forecast append batch")
        if (
            batch["schema"] != _FORECAST_LOG_SCHEMA
            or type(batch["version"]) is not int
            or batch["version"] != _PERSISTENCE_VERSION
            or batch["kind"] != "append"
            or type(batch["sequence"]) is not int
            or batch["sequence"] != expected_sequence
        ):
            raise GovernanceViolation("sealed forecast append sequence/schema mismatch")
        batch_core = {key: batch[key] for key in batch_keys - {"batch_sha256"}}
        if batch["batch_sha256"] != immutable_spec_hash(batch_core):
            raise GovernanceViolation("sealed forecast append batch hash mismatch")
        if batch["previous_ledger_sha256"] != current.ledger_sha256:
            raise GovernanceViolation("sealed forecast append hash chain is broken")
        raw_records = batch["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise GovernanceViolation("sealed forecast append batch must contain records")
        records: list[ForecastRecord] = []
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise GovernanceViolation(
                    f"sealed forecast append record {index} must be an object"
                )
            records.append(ForecastRecord.from_dict(raw_record))
        candidate = current.append(records)
        if batch["resulting_ledger_sha256"] != candidate.ledger_sha256:
            raise GovernanceViolation("sealed forecast append resulting hash mismatch")
        current = candidate
    return current, len(lines) - 1


def initialize_forecast_ledger(
    path: str | Path, ledger: SealedForecastLedger
) -> SealedForecastLedger:
    """Create an append-only forecast log, or verify the existing log identity."""

    destination = Path(path)
    ledger.verify_integrity()
    if ledger.records:
        raise GovernanceViolation("initialize_forecast_ledger requires an empty ledger")
    with _persistence_lock(destination):
        if _exclusive_create(destination, _json_bytes(_forecast_log_header(ledger))):
            loaded, _ = _load_forecast_log_unlocked(destination)
            return loaded
        existing, _ = _load_forecast_log_unlocked(destination)
        ledger.validate_append(existing)
        return existing


def load_forecast_ledger(path: str | Path) -> SealedForecastLedger:
    source = Path(path)
    with _persistence_lock(source, exclusive=False):
        ledger, _ = _load_forecast_log_unlocked(source)
        return ledger


def append_forecast_records(
    path: str | Path,
    records: Iterable[ForecastRecord],
    *,
    expected_ledger_sha256: str | None = None,
) -> SealedForecastLedger:
    """Append one hash-chained batch without rewriting any prior forecast bytes."""

    destination = Path(path)
    additions = tuple(records)
    if not additions:
        return load_forecast_ledger(destination)
    with _persistence_lock(destination):
        current, batch_count = _load_forecast_log_unlocked(destination)
        if (
            expected_ledger_sha256 is not None
            and current.ledger_sha256 != expected_ledger_sha256
        ):
            raise GovernanceViolation("forecast ledger changed since the caller's verified snapshot")
        updated = current.append(additions)
        batch = _forecast_batch(
            sequence=batch_count + 1,
            previous=current,
            records=additions,
            resulting=updated,
        )
        original_size = _append_one_record(destination, _json_bytes(batch))
        try:
            verified, _ = _load_forecast_log_unlocked(destination)
            if verified.ledger_sha256 != updated.ledger_sha256:
                raise GovernanceViolation("persisted forecast append failed hash verification")
        except BaseException:
            descriptor = os.open(destination, os.O_WRONLY)
            try:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            raise
        return verified


DEFAULT_PRODUCTION_CHECKS = (
    "survivorship_free",
    "delisting_returns",
    "historical_borrow",
    "historical_industry_classification",
    "alfred_vintages_used",
    "execution_grade_quotes",
)


@dataclass(frozen=True)
class PromotionThresholds:
    minimum_sealed_decisions: int = 24
    sharpe_minimum: float = 0.50
    newey_west_tstat_minimum: float = 2.00
    bootstrap_sharpe_p05_minimum: float = 0.00
    maximum_drawdown_floor: float = -0.20
    minimum_incremental_annualized_return: float | None = None
    minimum_incremental_newey_west_tstat: float | None = None
    required_production_checks: tuple[str, ...] = DEFAULT_PRODUCTION_CHECKS

    def __post_init__(self) -> None:
        if self.minimum_sealed_decisions <= 0:
            raise GovernanceViolation("minimum_sealed_decisions must be positive")
        if len(set(self.required_production_checks)) != len(self.required_production_checks):
            raise GovernanceViolation("required production checks must be unique")


@dataclass(frozen=True)
class PromotionCheck:
    name: str
    passed: bool
    actual: Any
    requirement: Any


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    action: str
    checks: tuple[PromotionCheck, ...]
    trial_id: str
    spec_sha256: str
    forecast_ledger_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action,
            "trial_id": self.trial_id,
            "spec_sha256": self.spec_sha256,
            "forecast_ledger_sha256": self.forecast_ledger_sha256,
            "checks": [asdict(check) for check in self.checks],
        }


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def evaluate_production_promotion(
    *,
    trial: TrialRecord,
    forecasts: SealedForecastLedger,
    metrics: Mapping[str, Any],
    production_checks: Mapping[str, bool],
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    """Evaluate hard production gates using sealed, preregistered evidence only."""

    trial.verify()
    forecasts.verify_integrity()
    if forecasts.trial_id != trial.trial_id or forecasts.spec_sha256 != trial.spec_sha256:
        raise GovernanceViolation("promotion evidence does not match the registered trial")
    if (
        forecasts.registered_at != trial.registered_at
        or forecasts.first_decision_at != trial.first_decision_at
    ):
        raise GovernanceViolation("promotion forecast ledger timing differs from the registered trial")

    resolved = thresholds or PromotionThresholds()
    checks: list[PromotionCheck] = [
        PromotionCheck("period_was_sealed", not trial.opened_period, trial.opened_period, False),
        PromotionCheck(
            "minimum_sealed_decisions",
            forecasts.decision_count >= resolved.minimum_sealed_decisions,
            forecasts.decision_count,
            resolved.minimum_sealed_decisions,
        ),
    ]

    metric_requirements = [
        ("sharpe", resolved.sharpe_minimum),
        ("newey_west_tstat", resolved.newey_west_tstat_minimum),
        ("bootstrap_sharpe_p05", resolved.bootstrap_sharpe_p05_minimum),
        ("max_drawdown", resolved.maximum_drawdown_floor),
    ]
    if resolved.minimum_incremental_annualized_return is not None:
        metric_requirements.append(
            ("incremental_annualized_return", resolved.minimum_incremental_annualized_return)
        )
    if resolved.minimum_incremental_newey_west_tstat is not None:
        metric_requirements.append(
            (
                "incremental_newey_west_tstat",
                resolved.minimum_incremental_newey_west_tstat,
            )
        )
    for name, minimum in metric_requirements:
        value = _finite_metric(metrics, name)
        checks.append(PromotionCheck(name, value is not None and value >= minimum, value, minimum))

    for name in resolved.required_production_checks:
        value = production_checks.get(name)
        checks.append(PromotionCheck(name, value is True, value, True))

    passed = all(check.passed for check in checks)
    return PromotionDecision(
        passed=passed,
        action="PROMOTE_TO_PRODUCTION" if passed else "BLOCK_PRODUCTION",
        checks=tuple(checks),
        trial_id=trial.trial_id,
        spec_sha256=trial.spec_sha256,
        forecast_ledger_sha256=forecasts.ledger_sha256,
    )
