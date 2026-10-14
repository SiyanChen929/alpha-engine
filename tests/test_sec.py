import pandas as pd

import pytest

from alpha_factor_engine.data.sec import SecCompanyFactsClient


def test_company_facts_client_requires_real_contact_user_agent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="real contact email"):
        SecCompanyFactsClient(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="real contact email"):
        SecCompanyFactsClient(
            cache_dir=tmp_path,
            user_agent="AlphaFactorEngine research-contact@example.invalid",
        )

    client = SecCompanyFactsClient(
        cache_dir=tmp_path,
        user_agent="AlphaFactorEngine/0.1 research-team@university.edu",
    )
    assert client.session.headers["User-Agent"].endswith("@university.edu")


def test_sec_annual_fact_uses_filing_time_and_keeps_amendment():
    payload = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "filed": "2024-02-15",
                                "form": "10-K",
                                "fp": "FY",
                                "fy": 2023,
                                "accn": "a",
                                "val": 100,
                            },
                            {
                                "end": "2023-12-31",
                                "filed": "2024-03-01",
                                "form": "10-K/A",
                                "fp": "FY",
                                "fy": 2023,
                                "accn": "b",
                                "val": 105,
                            },
                        ]
                    }
                }
            }
        }
    }
    acceptance = {
        "a": pd.Timestamp("2024-02-15T21:01:00Z"),
        "b": pd.Timestamp("2024-03-01T14:00:00Z"),
    }
    frame = SecCompanyFactsClient.normalize_annual_concepts("ABC", 1, payload, acceptance)
    assert frame["value"].tolist() == [100.0, 105.0]
    assert frame["available_at"].min() > frame["effective_at"].min()
