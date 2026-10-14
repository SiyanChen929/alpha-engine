import pandas as pd
import pytest

from alpha_factor_engine.research.sec_filing_change_plan import (
    FilingChangePlanError,
    acquisition_documents,
    assign_pair_download_phases,
    attach_current_proxy_mapping,
    build_rough_prior_year_pairs,
)


def _inventory():
    rows = []
    for accession, form, filing, report in (
        ("0000000001-22-000001", "10-K", "2022-02-20", "2021-12-31"),
        ("0000000001-23-000001", "10-K", "2023-02-20", "2022-12-31"),
        ("0000000001-24-000001", "10-K", "2024-02-20", "2023-12-31"),
        ("0000000001-23-000002", "10-Q", "2023-05-01", "2023-03-31"),
        ("0000000001-24-000002", "10-Q", "2024-05-01", "2024-03-31"),
        ("0000000002-24-000001", "10-K", "2024-03-01", "2023-12-31"),
    ):
        rows.append(
            {
                "issuer_cik": accession[:10],
                "accession": accession,
                "form": form,
                "filing_date": pd.Timestamp(filing),
                "report_date": pd.Timestamp(report),
                "acceptance_at": pd.Timestamp(f"{filing}T20:00:00Z"),
                "acceptance_at_raw": f"{filing}T20:00:00.000Z",
                "primary_document": accession + ".htm",
                "primary_document_url": "https://www.sec.gov/" + accession,
            }
        )
    return pd.DataFrame(rows)


def _universe():
    return pd.DataFrame(
        {
            "cik": [1],
            "ticker": ["AAA"],
            "proxy_universe_rank": [1],
            "selection_as_of": ["2026-08-31"],
            "evidence_status": ["current_survivor_proxy"],
        }
    )


def test_mapping_exposes_inventory_ciks_absent_from_proxy():
    mapped, misses = attach_current_proxy_mapping(_inventory(), _universe())
    assert mapped["issuer_cik"].unique().tolist() == ["0000000001"]
    assert misses["issuer_cik"].unique().tolist() == ["0000000002"]
    assert mapped["ticker"].unique().tolist() == ["AAA"]


def test_mapping_rejects_duplicate_cik_or_unlabelled_universe():
    universe = pd.concat([_universe(), _universe()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        attach_current_proxy_mapping(_inventory(), universe)
    universe = _universe()
    universe.loc[0, "evidence_status"] = "unknown"
    with pytest.raises(FilingChangePlanError, match="current_survivor"):
        attach_current_proxy_mapping(_inventory(), universe)


def test_rough_pair_is_header_only_and_closest_to_one_year():
    mapped, _ = attach_current_proxy_mapping(_inventory(), _universe())
    pairs, no_pair = build_rough_prior_year_pairs(
        mapped, signal_start="2023-01-01", signal_end="2024-12-31"
    )
    by_current = pairs.set_index("current_accession")
    assert by_current.loc["0000000001-24-000001", "prior_accession"] == "0000000001-23-000001"
    assert by_current.loc["0000000001-24-000002", "prior_accession"] == "0000000001-23-000002"
    assert pairs["pairing_status"].eq("HEADER_ONLY_ACQUISITION_SUPERSET_NOT_SIGNAL").all()
    assert "0000000001-23-000002" in set(no_pair["current_accession"])


def test_acquisition_documents_are_unique_union_of_pair_sides():
    mapped, _ = attach_current_proxy_mapping(_inventory(), _universe())
    pairs, _ = build_rough_prior_year_pairs(
        mapped, signal_start="2023-01-01", signal_end="2024-12-31"
    )
    documents = acquisition_documents(mapped, pairs)
    expected = set(pairs["current_accession"]) | set(pairs["prior_accession"])
    assert set(documents["accession"]) == expected
    assert documents["accession"].is_unique


def test_rough_plan_keeps_every_plausible_prior_for_dei_resolution():
    inventory = _inventory().iloc[0:0].copy()
    rows = []
    for accession, filing, report in (
        ("0000000001-23-000010", "2023-02-20", "2022-12-31"),
        ("0000000001-23-000011", "2023-04-01", "2023-02-01"),
        ("0000000001-24-000010", "2024-02-20", "2023-12-31"),
    ):
        rows.append(
            {
                "issuer_cik": "0000000001",
                "accession": accession,
                "form": "10-K",
                "filing_date": pd.Timestamp(filing),
                "report_date": pd.Timestamp(report),
                "acceptance_at": pd.Timestamp(f"{filing}T20:00:00Z"),
                "acceptance_at_raw": f"{filing}T20:00:00.000Z",
                "primary_document": accession + ".htm",
                "primary_document_url": "https://www.sec.gov/" + accession,
            }
        )
    mapped, _ = attach_current_proxy_mapping(pd.DataFrame(rows), _universe())
    pairs, _ = build_rough_prior_year_pairs(
        mapped, signal_start="2024-01-01", signal_end="2024-12-31"
    )
    current = pairs.loc[pairs["current_accession"].eq("0000000001-24-000010")]
    assert set(current["prior_accession"]) == {
        "0000000001-23-000010",
        "0000000001-23-000011",
    }
    assert set(current["rough_candidate_rank"]) == {1, 2}


def test_download_phase_uses_new_york_release_month_not_utc_year():
    pairs = pd.DataFrame(
        {
            "current_accession": ["late-et", "new-year"],
            "current_acceptance_at": [
                "2023-01-01T00:30:00Z",  # 2022-12-31 19:30 New York
                "2023-01-01T05:30:00Z",  # 2023-01-01 00:30 New York
            ],
            "proxy_universe_rank": [10, 10],
        }
    )
    phased = assign_pair_download_phases(
        pairs,
        development_release_month_end="2022-12",
        confirmation_release_month_end="2024-12",
    ).set_index("current_accession")
    assert str(phased.loc["late-et", "current_release_month"]) == "2022-12"
    assert phased.loc["late-et", "download_phase"] == "00_top25_engineering_smoke"
    assert str(phased.loc["new-year", "current_release_month"]) == "2023-01"
    assert phased.loc["new-year", "download_phase"] == "10_full_development_and_confirmation"
