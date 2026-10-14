import numpy as np
import pandas as pd

from alpha_factor_engine.research.quarterly_fundamental_momentum_v2 import (
    align_quarterly_features,
    extract_quarterly_release_features,
)


def _duration(
    start: str,
    end: str,
    value: float,
    accession: str,
    form: str,
    filed: str,
    fp: str,
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "val": value,
        "accn": accession,
        "fy": int(end[:4]),
        "fp": fp,
        "form": form,
        "filed": filed,
    }


def _instant(
    end: str,
    value: float,
    accession: str,
    form: str,
    filed: str,
    fp: str,
) -> dict[str, object]:
    return {
        "end": end,
        "val": value,
        "accn": accession,
        "fy": int(end[:4]),
        "fp": fp,
        "form": form,
        "filed": filed,
    }


def _payload(concepts: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": rows}} for tag, rows in concepts.items()
            }
        }
    }


def _q1_payload(*, cross_tag: bool = False, conflict: bool = False) -> dict[str, object]:
    ni_now = _duration(
        "2025-01-01", "2025-03-31", 15, "q1", "10-Q", "2025-05-01", "Q1"
    )
    ni_prior = _duration(
        "2024-01-01", "2024-03-31", 10, "q1", "10-Q", "2025-05-01", "Q1"
    )
    concepts: dict[str, list[dict[str, object]]] = {
        "NetIncomeLoss": [ni_now] + ([] if cross_tag else [ni_prior]),
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _duration(
                "2025-01-01", "2025-03-31", 120, "q1", "10-Q", "2025-05-01", "Q1"
            ),
            _duration(
                "2024-01-01", "2024-03-31", 100, "q1", "10-Q", "2025-05-01", "Q1"
            ),
        ],
        "GrossProfit": [
            _duration(
                "2025-01-01", "2025-03-31", 48, "q1", "10-Q", "2025-05-01", "Q1"
            ),
            _duration(
                "2024-01-01", "2024-03-31", 35, "q1", "10-Q", "2025-05-01", "Q1"
            ),
        ],
        "Assets": [_instant("2025-03-31", 500, "q1", "10-Q", "2025-05-01", "Q1")],
    }
    if cross_tag:
        concepts["ProfitLoss"] = [ni_prior]
    if conflict:
        concepts["NetIncomeLoss"].append({**ni_now, "val": 16})
    return _payload(concepts)


def _q4_payload(
    *,
    omit_revenue_prior_ytd: bool = False,
    cross_accession_ytd: bool = False,
    include_direct: bool = False,
) -> dict[str, object]:
    k = "k25"
    q3 = "q3-25"
    filed_k = "2026-02-15"
    filed_q3 = "2025-11-01"
    ytd_accession = q3 if cross_accession_ytd else k
    ytd_form = "10-Q" if cross_accession_ytd else "10-K"
    ytd_filed = filed_q3 if cross_accession_ytd else filed_k
    ytd_fp = "Q3" if cross_accession_ytd else "FY"
    concepts: dict[str, list[dict[str, object]]] = {
        "NetIncomeLoss": [
            _duration("2025-01-01", "2025-12-31", 100, k, "10-K", filed_k, "FY"),
            _duration("2024-01-01", "2024-12-31", 80, k, "10-K", filed_k, "FY"),
            _duration("2025-01-01", "2025-09-30", 70, ytd_accession, ytd_form, ytd_filed, ytd_fp),
            _duration("2024-01-01", "2024-09-30", 60, ytd_accession, ytd_form, ytd_filed, ytd_fp),
        ],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _duration("2025-01-01", "2025-12-31", 1200, k, "10-K", filed_k, "FY"),
            _duration("2024-01-01", "2024-12-31", 1000, k, "10-K", filed_k, "FY"),
            _duration("2025-01-01", "2025-09-30", 900, ytd_accession, ytd_form, ytd_filed, ytd_fp),
            *(
                []
                if omit_revenue_prior_ytd
                else [
                    _duration(
                        "2024-01-01",
                        "2024-09-30",
                        800,
                        ytd_accession,
                        ytd_form,
                        ytd_filed,
                        ytd_fp,
                    )
                ]
            ),
        ],
        "GrossProfit": [
            _duration("2025-01-01", "2025-12-31", 480, k, "10-K", filed_k, "FY"),
            _duration("2024-01-01", "2024-12-31", 350, k, "10-K", filed_k, "FY"),
            _duration("2025-01-01", "2025-09-30", 360, ytd_accession, ytd_form, ytd_filed, ytd_fp),
            _duration("2024-01-01", "2024-09-30", 280, ytd_accession, ytd_form, ytd_filed, ytd_fp),
        ],
        "Assets": [_instant("2025-12-31", 500, k, "10-K", filed_k, "FY")],
    }
    if include_direct:
        concepts["NetIncomeLoss"].extend(
            [
                _duration("2025-10-01", "2025-12-31", 32, k, "10-K", filed_k, "FY"),
                _duration("2024-10-01", "2024-12-31", 18, k, "10-K", filed_k, "FY"),
            ]
        )
        concepts["RevenueFromContractWithCustomerExcludingAssessedTax"].extend(
            [
                _duration("2025-10-01", "2025-12-31", 310, k, "10-K", filed_k, "FY"),
                _duration("2024-10-01", "2024-12-31", 190, k, "10-K", filed_k, "FY"),
            ]
        )
    return _payload(concepts)


def _inventory(*rows: tuple[str, str, str, str, str]) -> dict[str, object]:
    return {
        accession: {
            "form": form,
            "filing_date": filed,
            "report_date": report,
            "acceptance_at": accepted,
        }
        for accession, form, filed, report, accepted in rows
    }


def test_valid_10q_requires_same_tag_and_exact_asset_end():
    release = extract_quarterly_release_features("ABC", 1, _q1_payload())
    row = release.loc[release["accession"].eq("q1")].iloc[0]
    assert row["net_income_same_tag_comparator"]
    assert np.isclose(row["earnings_change_on_assets"], 0.01)
    assert np.isclose(row["sales_growth_yoy"], 0.20)


def test_cross_tag_or_conflicting_context_fails_closed():
    cross_tag = extract_quarterly_release_features(
        "ABC", 1, _q1_payload(cross_tag=True)
    ).iloc[0]
    conflict = extract_quarterly_release_features(
        "ABC", 1, _q1_payload(conflict=True)
    ).iloc[0]
    assert pd.isna(cross_tag["earnings_change_on_assets"])
    assert pd.isna(conflict["earnings_change_on_assets"])


def test_q4_four_leg_arithmetic_and_provenance():
    inventory = _inventory(
        ("q3-25", "10-Q", "2025-11-01", "2025-09-30", "2025-11-01T20:00:00Z"),
        ("k25", "10-K", "2026-02-15", "2025-12-31", "2026-02-15T20:00:00Z"),
    )
    release = extract_quarterly_release_features("ABC", 1, _q4_payload(), inventory)
    row = release.loc[release["accession"].eq("k25")].iloc[0]
    assert row["release_kind"] == "q4_10k_strict"
    assert row["net_income_current"] == 30
    assert row["net_income_prior_year"] == 20
    assert np.isclose(row["earnings_change_on_assets"], 0.02)
    assert row["revenue_current"] == 300
    assert row["revenue_prior_year"] == 200
    assert np.isclose(row["sales_growth_yoy"], 0.50)
    assert row["revenue_q3_ytd_accession"] == "k25"
    assert row["revenue_q4_method"] == "same_10k_derived"
    assert row["header_inventory_verified"]


def test_cross_accession_q4_derivation_is_excluded_without_scope_guard():
    inventory = _inventory(
        ("q3-25", "10-Q", "2025-11-01", "2025-09-30", "2025-11-01T20:00:00Z"),
        ("k25", "10-K", "2026-02-15", "2025-12-31", "2026-02-15T20:00:00Z"),
    )
    release = extract_quarterly_release_features(
        "ABC", 1, _q4_payload(cross_accession_ytd=True), inventory
    )
    row = release.loc[release["accession"].eq("k25")].iloc[0]
    assert row["release_kind"] == "q4_10k_barrier"
    assert pd.isna(row["earnings_change_on_assets"])
    assert pd.isna(row["sales_growth_yoy"])


def test_direct_q4_has_priority_over_same_filing_derived_value():
    inventory = _inventory(
        ("k25", "10-K", "2026-02-15", "2025-12-31", "2026-02-15T20:00:00Z"),
    )
    release = extract_quarterly_release_features(
        "ABC", 1, _q4_payload(include_direct=True), inventory
    )
    row = release.loc[release["accession"].eq("k25")].iloc[0]
    assert row["net_income_current"] == 32
    assert row["net_income_prior_year"] == 18
    assert row["net_income_q4_method"] == "direct_10k"
    assert row["revenue_current"] == 310
    assert row["revenue_prior_year"] == 190


def test_incomplete_q4_and_header_only_10k_are_null_barriers():
    inventory = _inventory(
        ("q3-25", "10-Q", "2025-11-01", "2025-09-30", "2025-11-01T20:00:00Z"),
        ("k25", "10-K", "2026-02-15", "2025-12-31", "2026-02-15T20:00:00Z"),
    )
    incomplete = extract_quarterly_release_features(
        "ABC", 1, _q4_payload(omit_revenue_prior_ytd=True), inventory
    )
    k_row = incomplete.loc[incomplete["accession"].eq("k25")].iloc[0]
    assert pd.isna(k_row["sales_growth_yoy"])
    assert k_row["supersession_only"]

    header_only = extract_quarterly_release_features(
        "ABC",
        1,
        _q1_payload(),
        _inventory(
            ("q1", "10-Q", "2025-05-01", "2025-03-31", "2025-05-01T20:00:00Z"),
            ("k25", "10-K", "2026-02-15", "2025-12-31", "2026-02-15T20:00:00Z"),
        ),
    )
    barrier = header_only.loc[header_only["accession"].eq("k25")].iloc[0]
    assert barrier["release_kind"] == "inventory_10k_barrier"
    assert pd.isna(barrier["earnings_change_on_assets"])


def test_unknown_period_header_also_terminates_state():
    release = extract_quarterly_release_features(
        "ABC",
        1,
        _q1_payload(),
        _inventory(
            ("q1", "10-Q", "2025-05-01", "2025-03-31", "2025-05-01T20:00:00Z"),
            ("bad-k", "10-K", "2025-08-01", "", "2025-08-01T20:00:00Z"),
        ),
    )
    barrier = release.loc[release["accession"].eq("bad-k")].iloc[0]
    assert barrier["release_kind"] == "unknown_period_null_barrier"
    assert pd.isna(barrier["fiscal_end"])


def test_late_old_amendment_cannot_regress_fiscal_state():
    q1 = _q1_payload()["facts"]["us-gaap"]
    concepts = {tag: value["units"]["USD"][:] for tag, value in q1.items()}
    for tag, values in list(concepts.items()):
        if tag == "Assets":
            values.append(_instant("2025-06-30", 510, "q2", "10-Q", "2025-08-01", "Q2"))
        else:
            values.extend(
                [
                    _duration("2025-04-01", "2025-06-30", 20, "q2", "10-Q", "2025-08-01", "Q2"),
                    _duration("2024-04-01", "2024-06-30", 18, "q2", "10-Q", "2025-08-01", "Q2"),
                ]
            )
    releases = extract_quarterly_release_features(
        "ABC",
        1,
        _payload(concepts),
        _inventory(
            ("q1", "10-Q", "2025-05-01", "2025-03-31", "2025-05-01T20:00:00Z"),
            ("q2", "10-Q", "2025-08-01", "2025-06-30", "2025-08-01T20:00:00Z"),
            ("q1a", "10-Q/A", "2025-08-15", "2025-03-31", "2025-08-15T20:00:00Z"),
        ),
    )
    assert "q1a" not in set(releases["accession"])
    assert releases.sort_values("available_at").iloc[-1]["fiscal_end"] == pd.Timestamp(
        "2025-06-30"
    )


def _release_row(ticker: str, cik: int, earnings: float, sales: float | None) -> dict[str, object]:
    return {
        "ticker": ticker,
        "cik": cik,
        "accession": f"r-{ticker}",
        "available_at": pd.Timestamp("2025-01-01T12:00:00Z"),
        "fiscal_end": pd.Timestamp("2024-09-30"),
        "earnings_change_on_assets": earnings,
        "sales_growth_yoy": sales,
        "gross_margin_change_yoy": 0.0,
    }


def _panel(tickers: list[str], *, entry: str = "2025-02-03T14:30:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": tickers,
            "signal_date": pd.to_datetime(["2025-02-01T01:00:00Z"] * len(tickers)),
            "entry_at": pd.to_datetime([entry] * len(tickers)),
            "adv_usd_3m": np.arange(len(tickers), dtype=float) + 100,
            "target_return": np.arange(len(tickers), dtype=float),
        }
    )


def test_common_intersection_rank_is_label_and_row_order_invariant():
    releases = pd.DataFrame(
        [
            _release_row("A", 1, 1.0, 1.0),
            _release_row("B", 2, 2.0, 2.0),
            _release_row("C", 3, 100.0, None),
        ]
    )
    panel = _panel(["A", "B", "C"])
    first = align_quarterly_features(panel, releases)
    changed = panel.sample(frac=1, random_state=7).copy()
    changed["target_return"] = [-100, 50, 900]
    second = align_quarterly_features(changed, releases).set_index("ticker")
    observed = first.set_index("ticker")
    assert observed.loc["A", "rank_earnings_change_on_assets"] == 0.5
    assert observed.loc["B", "rank_earnings_change_on_assets"] == 1.0
    assert pd.isna(observed.loc["C", "quarterly_fundamental_momentum_score"])
    pd.testing.assert_series_equal(
        observed["quarterly_fundamental_momentum_score"].sort_index(),
        second["quarterly_fundamental_momentum_score"].sort_index(),
    )


def test_clock_is_strict_and_one_cik_counts_once():
    release_time = pd.Timestamp("2025-02-03T14:00:00Z")
    releases = pd.DataFrame(
        [
            {**_release_row("A", 1, 1.0, 1.0), "available_at": release_time},
            {**_release_row("A2", 1, 1.0, 1.0), "available_at": release_time},
        ]
    )
    # decision_at equals the release clock: strict inequality makes it unusable.
    equal = align_quarterly_features(
        _panel(["A"], entry="2025-02-03T14:30:00Z"), releases
    )
    assert not equal.iloc[0]["quarterly_feature_eligible"]

    later_panel = _panel(["A", "A2"], entry="2025-02-03T15:00:00Z")
    later_panel.loc[later_panel["ticker"].eq("A"), "adv_usd_3m"] = 1_000
    later = align_quarterly_features(later_panel, releases)
    assert later["issuer_representative"].sum() == 1
    assert later.loc[later["ticker"].eq("A"), "quarterly_feature_eligible"].iloc[0]
    assert not later.loc[later["ticker"].eq("A2"), "quarterly_feature_eligible"].iloc[0]


def test_stale_age_boundary_is_explicit():
    releases = pd.DataFrame([_release_row("A", 1, 1.0, 1.0)])
    available = pd.Timestamp(releases.iloc[0]["available_at"])
    at_boundary = available + pd.Timedelta(days=180)
    panel = _panel(["A"], entry=(at_boundary + pd.Timedelta(minutes=30)).isoformat())
    assert align_quarterly_features(panel, releases).iloc[0][
        "quarterly_feature_eligible"
    ]
    after = panel.copy()
    after["entry_at"] += pd.Timedelta(nanoseconds=1)
    assert not align_quarterly_features(after, releases).iloc[0][
        "quarterly_feature_eligible"
    ]
