from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_factor_engine.features.filing_change import (
    FilingChangeError,
    assign_release_month_quintiles,
    binary_jaccard,
    binary_jaccard_counts,
    build_strict_fiscal_pairs,
    clean_primary_filing,
    extract_fiscal_identity,
    raw_tf_cosine,
    raw_tf_cosine_counts,
    similarity_record,
)


def test_parser_removes_only_numeric_heavy_tables_and_hidden_content():
    html = b"""
    <html><style>secret style</style><script>secret script</script><body>
      <p>Management expects worldwide demand to increase.</p>
      <table><tr><td>Revenue 2024</td><td>123456789</td></tr></table>
      <table><tr><td>Risk Factors</td><td>Supply chain disruption may continue</td></tr></table>
      <ix:hidden><ix:nonNumeric name="dei:EntityRegistrantName">Hidden Corp</ix:nonNumeric></ix:hidden>
    </body></html>
    """
    cleaned = clean_primary_filing(html)
    assert "management expects worldwide demand" in cleaned.text
    assert "supply chain disruption" in cleaned.text
    assert "secret" not in cleaned.text
    assert "hidden corp" not in cleaned.text
    assert "revenue" not in cleaned.text
    assert cleaned.tables_seen == 2
    assert cleaned.numeric_tables_removed == 1
    assert cleaned.nonnumeric_tables_retained == 1


def test_self_closing_hidden_tag_does_not_suppress_remainder():
    cleaned = clean_primary_filing("<html><ix:hidden/><p>Visible remainder</p></html>")
    assert cleaned.tokens == ("visible", "remainder")


def test_hidden_subtree_with_void_tag_does_not_leak_or_swallow_remainder():
    cleaned = clean_primary_filing(
        "<ix:hidden>secret<br>more secret</ix:hidden><p>Visible after</p>"
    )
    assert cleaned.tokens == ("visible", "after")


def test_hidden_attributes_are_removed():
    cleaned = clean_primary_filing(
        '<div style="display: none">secret</div>'
        '<div aria-hidden="true">also secret</div><p>Visible</p>'
    )
    assert cleaned.tokens == ("visible",)


def test_unclosed_table_fails_closed():
    with pytest.raises(FilingChangeError, match="unclosed HTML table"):
        clean_primary_filing("<table><tr><td>Risk text")


def test_fiscal_identity_supports_inline_and_direct_xbrl():
    inline = """
    <ix:nonNumeric name="dei:DocumentFiscalYearFocus">2024</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">Q2</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentPeriodEndDate">2024-06-30</ix:nonNumeric>
    """
    result = extract_fiscal_identity(inline)
    assert result == {
        "fiscal_year": 2024,
        "fiscal_period": "Q2",
        "period_end": pd.Timestamp("2024-06-30"),
        "fiscal_identity_complete": True,
    }
    direct = "<dei:DocumentFiscalYearFocus>2023</dei:DocumentFiscalYearFocus>"
    assert extract_fiscal_identity(direct)["fiscal_year"] == 2023


def test_fiscal_identity_conflicts_fail_closed():
    html = """
    <ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">Q1</ix:nonNumeric>
    <ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">Q2</ix:nonNumeric>
    """
    with pytest.raises(FilingChangeError, match="conflicting"):
        extract_fiscal_identity(html)


def test_similarity_matches_paper_examples_and_has_no_global_vocabulary():
    a = "we expect demand to increase".split()
    b = "we expect worldwide demand to increase".split()
    c = "we expect weakness in sales".split()
    assert raw_tf_cosine(a, b) == pytest.approx(5 / np.sqrt(30))
    assert binary_jaccard(a, b) == pytest.approx(5 / 6)
    assert raw_tf_cosine(a, c) == pytest.approx(2 / 5)
    assert binary_jaccard(a, c) == pytest.approx(2 / 8)
    assert raw_tf_cosine(a + ["demand"], b) != raw_tf_cosine(a, b)
    left_counts = {term: a.count(term) for term in set(a)}
    right_counts = {term: b.count(term) for term in set(b)}
    assert raw_tf_cosine_counts(left_counts, right_counts) == raw_tf_cosine(a, b)
    assert binary_jaccard_counts(left_counts, right_counts) == binary_jaccard(a, b)


def test_similarity_minimum_tokens_is_a_pre_outcome_gate():
    current = clean_primary_filing("alpha beta gamma")
    prior = clean_primary_filing("alpha beta delta")
    rejected = similarity_record(current, prior, minimum_tokens=4)
    assert not rejected["text_eligible"]
    assert np.isnan(rejected["cosine_similarity"])
    accepted = similarity_record(current, prior, minimum_tokens=3)
    assert accepted["text_eligible"]
    assert accepted["jaccard_similarity"] == pytest.approx(0.5)


def _filings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "prior-k", "10-K", "2023-02-01T20:00:00Z", "2022-12-31", 2022, "FY"),
            (1, "current-k", "10-K", "2024-02-01T20:00:00Z", "2023-12-31", 2023, "FY"),
            (1, "prior-q", "10-Q", "2023-05-01T20:00:00Z", "2023-03-31", 2023, "Q1"),
            (1, "current-q", "10-Q", "2024-05-01T20:00:00Z", "2024-03-31", 2024, "Q1"),
            (2, "amend", "10-Q/A", "2024-06-01T20:00:00Z", "2024-03-31", 2024, "Q1"),
            (3, "unknown", "10-Q", "2024-06-01T20:00:00Z", "2024-03-31", 2024, None),
        ],
        columns=[
            "cik",
            "accession",
            "form",
            "accepted_at",
            "report_date",
            "fiscal_year",
            "fiscal_period",
        ],
    )


def test_pairs_only_same_cik_form_period_and_prior_fiscal_year():
    pairs, exclusions = build_strict_fiscal_pairs(_filings())
    assert set(pairs["current_accession"]) == {"current-k", "current-q"}
    mapping = pairs.set_index("current_accession")["prior_accession"].to_dict()
    assert mapping == {"current-k": "prior-k", "current-q": "prior-q"}
    reasons = exclusions.set_index("accession")["reason"].to_dict()
    assert reasons["amend"] == "unsupported_or_amended_form"
    assert reasons["unknown"] == "missing_fiscal_identity"


def test_duplicate_original_fiscal_identity_excludes_whole_key():
    filings = _filings()
    duplicate = filings.loc[filings["accession"].eq("prior-k")].copy()
    duplicate["accession"] = "prior-k-duplicate"
    pairs, exclusions = build_strict_fiscal_pairs(pd.concat([filings, duplicate]))
    assert "current-k" not in set(pairs["current_accession"])
    assert set(
        exclusions.loc[
            exclusions["reason"].eq("duplicate_original_fiscal_filing"), "accession"
        ]
    ) == {"prior-k", "prior-k-duplicate"}


def test_monthly_quintiles_use_only_completed_release_month():
    rows = []
    for month, shift in (("2024-01", 0.0), ("2024-02", 100.0)):
        for index in range(10):
            rows.append(
                {
                    "current_accession": f"{month}-{index}",
                    "current_accepted_at": pd.Timestamp(f"{month}-15", tz="UTC"),
                    "cosine_similarity": shift + index,
                }
            )
    assigned, ledger = assign_release_month_quintiles(
        pd.DataFrame(rows),
        score_column="cosine_similarity",
        minimum_monthly_observations=10,
    )
    january = assigned.loc[assigned["release_month"].astype(str).eq("2024-01")]
    february = assigned.loc[assigned["release_month"].astype(str).eq("2024-02")]
    assert january["formation_month"].astype(str).unique().tolist() == ["2024-02"]
    assert february["formation_month"].astype(str).unique().tolist() == ["2024-03"]
    assert january.loc[january["cosine_similarity"].eq(0), "similarity_quintile"].iloc[0] == 1
    assert january.loc[january["cosine_similarity"].eq(9), "similarity_quintile"].iloc[0] == 5
    assert ledger["eligible"].all()


def test_monthly_quintiles_fail_closed_on_ties_or_small_cohort():
    scores = pd.DataFrame(
        {
            "current_accession": [f"a{i}" for i in range(5)],
            "current_accepted_at": [pd.Timestamp("2024-01-15", tz="UTC")] * 5,
            "jaccard_similarity": [0.5] * 5,
        }
    )
    assigned, ledger = assign_release_month_quintiles(
        scores, score_column="jaccard_similarity", minimum_monthly_observations=5
    )
    assert not assigned["sort_eligible"].any()
    assert not ledger.iloc[0]["eligible"]
    assert ledger.iloc[0]["reason"] == "non_strict_monthly_breakpoints"
