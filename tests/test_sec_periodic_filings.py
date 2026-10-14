import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

import alpha_factor_engine.data.sec_periodic_filings as periodic
from alpha_factor_engine.data.sec_periodic_filings import (
    PeriodicFilingDataError,
    SecPeriodicFilingsClient,
    build_acceptance_header_url,
    build_primary_document_url,
    load_periodic_filing_inventory,
    parse_raw_acceptance_header,
)


CIK = "0000123456"
RECENT_ACCESSION = "0000123456-24-000001"
HISTORY_ACCESSION = "0000123456-23-000002"
ACCEPTANCE = "2024-02-20T21:15:30.000Z"


def _row(
    *,
    accession=RECENT_ACCESSION,
    filing_date="2024-02-20",
    report_date="2023-12-31",
    acceptance=ACCEPTANCE,
    form="10-K",
    primary_document="issuer-20231231.htm",
):
    return {
        "accessionNumber": accession,
        "filingDate": filing_date,
        "reportDate": report_date,
        "acceptanceDateTime": acceptance,
        "form": form,
        "primaryDocument": primary_document,
    }


def _columnar(rows):
    return {
        column: [row[column] for row in rows]
        for column in periodic.REQUIRED_HEADER_COLUMNS
    }


def _history_entry(
    *,
    name=f"CIK{CIK}-submissions-001.json",
    filing_from="2023-01-01",
    filing_to="2023-12-31",
    filing_count=2,
):
    return {
        "name": name,
        "filingCount": filing_count,
        "filingFrom": filing_from,
        "filingTo": filing_to,
    }


def _write_gzip_json(path: Path, payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return raw, compressed


def _write_cache(
    root: Path,
    *,
    recent_rows=None,
    history_rows=None,
    history_entries=None,
    cik=CIK,
):
    recent_rows = recent_rows if recent_rows is not None else [_row()]
    history_rows = history_rows if history_rows is not None else []
    if history_entries is None:
        history_entries = [_history_entry()] if history_rows else []
    main = {
        "cik": cik,
        "filings": {
            "recent": _columnar(recent_rows),
            "files": history_entries,
        },
    }
    main_path = root / "submissions" / f"CIK{cik}.json.gz"
    main_raw, main_compressed = _write_gzip_json(main_path, main)
    archives = []
    if history_rows:
        for entry in history_entries:
            archive_path = (
                root
                / "submissions_archive"
                / f"CIK{cik}"
                / f"{entry['name']}.gz"
            )
            raw, compressed = _write_gzip_json(
                archive_path, _columnar(history_rows)
            )
            archives.append((archive_path, raw, compressed))
    return main_path, main_raw, main_compressed, archives


def test_inventory_loads_current_and_history_excludes_amendments_and_hashes_sources(
    tmp_path,
):
    history = [
        _row(
            accession=HISTORY_ACCESSION,
            filing_date="2023-05-04",
            report_date="2023-03-31",
            acceptance="2023-05-04T20:05:00.000Z",
            form="10-Q",
            primary_document="issuer-20230331.htm",
        ),
        _row(
            accession="0000123456-23-000003",
            filing_date="2023-05-08",
            report_date="2023-03-31",
            acceptance="2023-05-08T13:00:00.000Z",
            form="10-Q/A",
            primary_document="issuer-20230331a.htm",
        ),
    ]
    recent = [
        _row(),
        _row(
            accession="0000123456-24-000004",
            filing_date="2024-02-21",
            acceptance="2024-02-21T18:00:00.000Z",
            form="10-K/A",
            primary_document="issuer-20231231a.htm",
        ),
        _row(
            accession="0000123456-24-000005",
            filing_date="2024-03-01",
            acceptance="2024-03-01T18:00:00.000Z",
            form="8-K",
            primary_document="issuer-8k.htm",
        ),
    ]
    main_path, main_raw, main_compressed, archives = _write_cache(
        tmp_path, recent_rows=recent, history_rows=history
    )

    inventory = load_periodic_filing_inventory(
        tmp_path,
        issuer_ciks=[123456],
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

    assert inventory["accession"].tolist() == [HISTORY_ACCESSION, RECENT_ACCESSION]
    assert inventory["form"].tolist() == ["10-Q", "10-K"]
    assert inventory["issuer_cik"].tolist() == [CIK, CIK]
    assert str(inventory["acceptance_at"].dtype) == "datetime64[ns, UTC]"
    assert set(inventory["source_kind"]) == {"recent", "history"}
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/123456/"
        "000012345624000001/issuer-20231231.htm"
    )
    assert inventory.loc[1, "primary_document_url"] == expected_url

    recent_row = inventory.loc[inventory["source_kind"].eq("recent")].iloc[0]
    assert recent_row["source_path"] == main_path.resolve().as_posix()
    assert recent_row["source_file_sha256"] == hashlib.sha256(
        main_compressed
    ).hexdigest()
    assert recent_row["source_content_sha256"] == hashlib.sha256(main_raw).hexdigest()
    archive_path, archive_raw, archive_compressed = archives[0]
    history_row = inventory.loc[inventory["source_kind"].eq("history")].iloc[0]
    assert history_row["source_path"] == archive_path.resolve().as_posix()
    assert history_row["source_file_sha256"] == hashlib.sha256(
        archive_compressed
    ).hexdigest()
    assert history_row["source_content_sha256"] == hashlib.sha256(
        archive_raw
    ).hexdigest()


def test_old_declared_history_outside_requested_range_need_not_be_present(tmp_path):
    old = _history_entry(
        name=f"CIK{CIK}-submissions-001.json",
        filing_from="1994-01-01",
        filing_to="2000-12-31",
        filing_count=100,
    )
    _write_cache(tmp_path, history_entries=[old])

    inventory = load_periodic_filing_inventory(
        tmp_path, start_date="2018-01-01"
    )

    assert inventory["accession"].tolist() == [RECENT_ACCESSION]


def test_selected_declared_history_is_mandatory(tmp_path):
    _write_cache(tmp_path, history_entries=[_history_entry()])

    with pytest.raises(FileNotFoundError, match="history is absent"):
        load_periodic_filing_inventory(tmp_path, start_date="2023-01-01")


def test_inventory_never_opens_sources_outside_frozen_manifest(tmp_path):
    history = [
        _row(
            accession=HISTORY_ACCESSION,
            filing_date="2023-05-04",
            report_date="2023-03-31",
            acceptance="2023-05-04T20:05:00.000Z",
            form="10-Q",
            primary_document="issuer-20230331.htm",
        )
    ]
    main, _, _, archives = _write_cache(tmp_path, history_rows=history)
    with pytest.raises(PeriodicFilingDataError, match="outside the frozen source manifest"):
        load_periodic_filing_inventory(
            tmp_path,
            start_date="2023-01-01",
            allowed_source_paths=[main],
        )
    inventory = load_periodic_filing_inventory(
        tmp_path,
        start_date="2023-01-01",
        allowed_source_paths=[main, archives[0][0]],
    )
    assert set(inventory["accession"]) == {RECENT_ACCESSION, HISTORY_ACCESSION}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accessionNumber", "123", "invalid accession"),
        ("primaryDocument", "../issuer.htm", "invalid primaryDocument"),
        (
            "acceptanceDateTime",
            "2024-02-20T21:15:30",
            "invalid acceptanceDateTime",
        ),
        ("reportDate", "2024-02-30", "invalid reportDate"),
    ],
)
def test_inventory_rejects_invalid_periodic_header_fields(
    tmp_path, field, value, message
):
    row = _row()
    row[field] = value
    _write_cache(tmp_path, recent_rows=[row])

    with pytest.raises(PeriodicFilingDataError, match=message):
        load_periodic_filing_inventory(tmp_path)


def test_inventory_rejects_mismatched_column_lengths(tmp_path):
    main = {
        "cik": CIK,
        "filings": {
            "recent": _columnar([_row()]),
            "files": [],
        },
    }
    main["filings"]["recent"]["primaryDocument"] = []
    path = tmp_path / "submissions" / f"CIK{CIK}.json.gz"
    _write_gzip_json(path, main)

    with pytest.raises(PeriodicFilingDataError, match="column lengths differ"):
        load_periodic_filing_inventory(tmp_path)


def test_inventory_rejects_conflicting_duplicate_accession(tmp_path):
    history_row = _row(primary_document="different.htm")
    entry = _history_entry(
        filing_from="2024-01-01", filing_to="2024-12-31", filing_count=1
    )
    _write_cache(
        tmp_path,
        recent_rows=[_row()],
        history_rows=[history_row],
        history_entries=[entry],
    )

    with pytest.raises(PeriodicFilingDataError, match="conflicting periodic"):
        load_periodic_filing_inventory(tmp_path)


def test_identical_duplicate_accession_is_deduplicated_deterministically(tmp_path):
    entry = _history_entry(
        filing_from="2024-01-01", filing_to="2024-12-31", filing_count=1
    )
    _write_cache(
        tmp_path,
        recent_rows=[_row()],
        history_rows=[_row()],
        history_entries=[entry],
    )

    inventory = load_periodic_filing_inventory(tmp_path)

    assert len(inventory) == 1
    assert inventory.loc[0, "accession"] == RECENT_ACCESSION


def test_cik_filename_and_payload_conflict_fails_closed(tmp_path):
    _write_cache(tmp_path, cik=CIK)
    path = tmp_path / "submissions" / f"CIK{CIK}.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["cik"] = "0000654321"
    _write_gzip_json(path, payload)

    with pytest.raises(PeriodicFilingDataError, match="CIK conflicts"):
        load_periodic_filing_inventory(tmp_path)


def test_primary_document_url_validates_all_path_components():
    url = build_primary_document_url(
        issuer_cik=123456,
        accession=RECENT_ACCESSION,
        primary_document="issuer-20231231.HTM",
    )

    assert url.endswith("/000012345624000001/issuer-20231231.HTM")
    with pytest.raises(PeriodicFilingDataError, match="invalid accession"):
        build_primary_document_url(
            issuer_cik=123456,
            accession="../../etc/passwd",
            primary_document="issuer.htm",
        )
    with pytest.raises(PeriodicFilingDataError, match="invalid primaryDocument"):
        build_primary_document_url(
            issuer_cik=123456,
            accession=RECENT_ACCESSION,
            primary_document="https://evil.invalid/x.htm",
        )


class _FakeResponse:
    def __init__(
        self,
        *,
        url,
        content=b"",
        status_code=200,
        content_type="text/html; charset=utf-8",
        headers=None,
    ):
        self.url = url
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type, **(headers or {})}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def _html(label="one"):
    return (
        "<!doctype html><html><head><title>10-K</title></head>"
        f"<body><p>{label} filing content with enough bytes.</p></body></html>"
    ).encode()


def _client(tmp_path, user_agent="AlphaResearch data-team@realfirm.com"):
    return SecPeriodicFilingsClient(
        cache_dir=tmp_path,
        user_agent=user_agent,
        min_interval_seconds=0.1,
    )


def _download(client, **overrides):
    arguments = {
        "issuer_cik": CIK,
        "accession": RECENT_ACCESSION,
        "primary_document": "issuer-20231231.htm",
        "acceptance_at": ACCEPTANCE,
    }
    arguments.update(overrides)
    return client.download_primary_html(**arguments)


@pytest.mark.parametrize(
    "user_agent",
    [
        "bot@example.invalid",
        "AlphaResearch bot@example.com",
        "only-email@realfirm.com",
        "no-contact-email",
        "AlphaResearch x@realfirm.com\r\nInjected: value",
    ],
)
def test_client_requires_identified_non_placeholder_sec_user_agent(
    tmp_path, user_agent
):
    with pytest.raises(PeriodicFilingDataError, match="real contact email"):
        _client(tmp_path, user_agent=user_agent)


def test_download_is_content_addressed_gzipped_and_resumes_from_verified_cache(
    tmp_path,
):
    client = _client(tmp_path)
    content = _html()
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(url=url, content=content)

    client.session.get = get
    downloaded, record, path = _download(client)

    digest = hashlib.sha256(content).hexdigest()
    assert downloaded == content
    assert path.name == f"{digest}.html.gz"
    assert gzip.decompress(path.read_bytes()) == content
    assert record.sha256 == digest
    assert record.notes["cache_status"] == "downloaded"
    assert record.notes["compressed_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["timeout"] == 60

    def unexpected_request(*args, **kwargs):
        pytest.fail("verified cache should avoid a second HTTP request")

    client.session.get = unexpected_request
    cached, cached_record, cached_path = _download(
        client, expected_sha256=digest
    )
    assert cached == content
    assert cached_path == path
    assert cached_record.notes["cache_status"] == "cached_verified"


def test_existing_cache_hash_mismatch_fails_before_network(tmp_path):
    client = _client(tmp_path)
    directory = (
        tmp_path / "primary_documents" / f"CIK{CIK}" / RECENT_ACCESSION
    )
    directory.mkdir(parents=True)
    wrong_name = "a" * 64
    (directory / f"{wrong_name}.html.gz").write_bytes(
        gzip.compress(_html(), mtime=0)
    )

    def unexpected_request(*args, **kwargs):
        pytest.fail("invalid cache should fail before network access")

    client.session.get = unexpected_request
    with pytest.raises(PeriodicFilingDataError, match="hash conflicts with filename"):
        _download(client)


def test_refresh_rejects_changed_content_for_existing_accession(tmp_path):
    client = _client(tmp_path)
    first = _html("first")
    second = _html("second")
    client.session.get = lambda url, **kwargs: _FakeResponse(
        url=url, content=first
    )
    _download(client)
    client.session.get = lambda url, **kwargs: _FakeResponse(
        url=url, content=second
    )

    with pytest.raises(PeriodicFilingDataError, match="content changed"):
        _download(client, refresh=True)

    directory = (
        tmp_path / "primary_documents" / f"CIK{CIK}" / RECENT_ACCESSION
    )
    assert len(list(directory.glob("*.html.gz"))) == 1


def test_cached_primary_requires_source_bound_provenance_sidecar(tmp_path):
    client = _client(tmp_path)
    content = _html()
    digest = hashlib.sha256(content).hexdigest()
    directory = tmp_path / "primary_documents" / f"CIK{CIK}" / RECENT_ACCESSION
    directory.mkdir(parents=True)
    (directory / f"{digest}.html.gz").write_bytes(gzip.compress(content, mtime=0))
    client.session.get = lambda *args, **kwargs: pytest.fail(
        "unprovenanced cache must fail before network"
    )
    with pytest.raises(PeriodicFilingDataError, match="provenance sidecar"):
        _download(client)


def test_download_retries_transient_http_failure(monkeypatch, tmp_path):
    client = _client(tmp_path)
    responses = [
        _FakeResponse(url="unused", status_code=503, headers={"Retry-After": "0"}),
        None,
    ]
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        response = responses.pop(0)
        if response is None:
            return _FakeResponse(url=url, content=_html())
        response.url = url
        return response

    monkeypatch.setattr(periodic.time, "sleep", lambda seconds: None)
    client.session.get = get

    content, record, _ = _download(client)

    assert content == _html()
    assert len(calls) == 2
    assert record.notes["cache_status"] == "downloaded"


@pytest.mark.parametrize(
    ("response_factory", "message"),
    [
        (
            lambda url: _FakeResponse(
                url=url,
                content=b'{"not":"html"}',
                content_type="application/json",
            ),
            "media type",
        ),
        (
            lambda url: _FakeResponse(
                url="https://www.sec.gov/Archives/redirected.htm",
                content=_html(),
            ),
            "response URL conflicts",
        ),
        (
            lambda url: _FakeResponse(
                url=url,
                content=(
                    b"<html><body>Your Request Originates from an Undeclared "
                    b"Automated Tool and is blocked.</body></html>"
                ),
            ),
            "access-block page",
        ),
    ],
)
def test_download_rejects_wrong_response_identity_or_content(
    tmp_path, response_factory, message
):
    client = _client(tmp_path)
    client.session.get = lambda url, **kwargs: response_factory(url)

    with pytest.raises(PeriodicFilingDataError, match=message):
        _download(client)


def test_download_validates_acceptance_and_expected_hash_before_network(tmp_path):
    client = _client(tmp_path)

    def unexpected_request(*args, **kwargs):
        pytest.fail("invalid metadata should fail before network access")

    client.session.get = unexpected_request
    with pytest.raises(PeriodicFilingDataError, match="acceptanceDateTime"):
        _download(client, acceptance_at="2024-02-20 21:15:30")
    with pytest.raises(PeriodicFilingDataError, match="expected.*sha256"):
        _download(client, expected_sha256="not-a-hash")


def _raw_header(clock="20240220161530"):
    return (
        b"<SEC-HEADER>\n<ACCESSION-NUMBER>" + RECENT_ACCESSION.encode()
        + b"\n<ACCEPTANCE-DATETIME>" + clock.encode() + b"\n</SEC-HEADER>\n"
    )


def test_raw_acceptance_header_url_and_new_york_clock():
    url = build_acceptance_header_url(
        issuer_cik=CIK, accession=RECENT_ACCESSION
    )
    assert url.endswith(
        "/123456/000012345624000001/0000123456-24-000001.hdr.sgml"
    )
    assert parse_raw_acceptance_header(_raw_header()) == pd.Timestamp(
        "2024-02-20T21:15:30Z"
    )
    with pytest.raises(PeriodicFilingDataError, match="expected one"):
        parse_raw_acceptance_header(b"<SEC-HEADER></SEC-HEADER>")


def test_acceptance_header_download_reconciles_and_resumes(tmp_path):
    client = _client(tmp_path)
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(
            url=url, content=_raw_header(), content_type="text/plain"
        )

    client.session.get = get
    content, raw_clock, record, path = client.download_acceptance_header(
        issuer_cik=CIK,
        accession=RECENT_ACCESSION,
        api_acceptance_at=ACCEPTANCE,
    )
    assert content == _raw_header()
    assert raw_clock == pd.Timestamp("2024-02-20T21:15:30Z")
    assert record.notes["api_raw_delta_seconds"] == 0
    assert record.notes["cache_status"] == "downloaded"
    assert gzip.decompress(path.read_bytes()) == content
    assert len(calls) == 1

    client.session.get = lambda *args, **kwargs: pytest.fail(
        "verified raw-header cache should avoid network"
    )
    _, _, cached_record, cached_path = client.download_acceptance_header(
        issuer_cik=CIK,
        accession=RECENT_ACCESSION,
        api_acceptance_at=ACCEPTANCE,
    )
    assert cached_path == path
    assert cached_record.notes["cache_status"] == "cached_verified"


def test_acceptance_header_clock_conflict_fails_closed(tmp_path):
    client = _client(tmp_path)
    client.session.get = lambda url, **kwargs: _FakeResponse(
        url=url, content=_raw_header("20240220161540"), content_type="text/plain"
    )
    with pytest.raises(PeriodicFilingDataError, match="clock conflict"):
        client.download_acceptance_header(
            issuer_cik=CIK,
            accession=RECENT_ACCESSION,
            api_acceptance_at=ACCEPTANCE,
            maximum_delta_seconds=1,
        )
