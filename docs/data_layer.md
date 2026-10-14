# Data layer and point-in-time contract

The engine uses one rule across every source: a record is eligible only when `available_at <= decision_time`. Economic period, market event, filing, vendor receipt, and ingestion time remain separate.

## Storage layout

```text
data/cache/                         immutable-ish raw response cache, excluded from Git
  yahoo/                            public chart payloads
  marketdata/confirmed/             completed MarketData sessions
  marketdata/provisional/           same-day or uncertain sessions
  marketdata/options/{confirmed,provisional}/
  fred/                             FRED CSV and ALFRED responses
  sec/{companyfacts,submissions}/   compressed raw EDGAR JSON

artifacts/public_research/data/     normalized run snapshot
  daily_prices.parquet
  sec_annual_concepts.parquet
  sec_filing_events.parquet
  company_metadata.parquet
  fred_*.parquet
  monthly_feature_panel.parquet
```

`data_manifest.json` records retrieval time, source URL, content hash, coverage, finality, and revision policy. A production lake should make the raw tier append-only and attach entitlements and parser versions to every normalized object.

## Source matrix

| Domain | Adapter | Current state | Required production upgrade |
|---|---|---|---|
| Yahoo Finance | `YahooFinanceClient` | Working public daily-bar fallback; preserves raw payload hashes, dividends, splits, and confirmed/provisional status | Replace current-ticker universe with a permanent-ID security master and delisting-complete source; retain Yahoo only for reconciliation |
| MarketData stocks | `MarketDataClient.fetch_candles` | Working authenticated candle path with separate confirmed/provisional caches | Add exchange calendar completeness, bulk endpoints, entitlement metadata, and live quote/NBBO adapter |
| MarketData options | `MarketDataClient.fetch_option_chain` | Current/historical EOD chain retrieval; current and historical caches separated | Historical chains are not OPRA execution replay; add NBBO/trades, OCC adjustment history, quote conditions, and contract master |
| SEC EDGAR | `SecCompanyFactsClient` | Working ticker/CIK, Company Facts, submissions, acceptance-time mapping, amendments, and filing event tape | Add filing document/text archive, custom taxonomy harmonization, exact acceptance mapping for older submission shards, and restatement QA |
| FRED | `FredClient.fetch_latest_series` | Working current-vintage series with explicit availability lag and `latest_vintage` label | Use only for market series with low revision risk or ex-post attribution |
| ALFRED | `FredClient.fetch_alfred_vintages` | Working API-key path retaining `realtime_start`/`realtime_end` | Supply `FRED_API_KEY`; validate each release calendar and time of day |
| CRSP | `CRSPAdapter` | Strict Parquet PIT schema and validation | Supply licensed stock, name, distribution, and delisting files; key by PERMNO/PERMCO |
| Compustat PIT | `CompustatPITAdapter` | Strict PIT schema and validation | Supply vendor knowledge timestamps, CCM link history, quarterly/annual precedence, units, and restatements |
| I/B/E/S | `IBESAdapter` | Strict PIT schema and validation | Supply detail/summary history, estimate announcement timestamps, actuals, split basis, and forecast-period mapping |
| TAQ | `TAQAdapter` | Strict event-time/SIP-time schema and validation | Supply trades, quotes, conditions, corrections, sequence ordering, and NBBO reconstruction |
| OPRA/OCC | `OPRAOptionsAdapter` | Strict quote/contract schema and validation | Supply historical quotes/trades/sizes, symbology, adjustments, rates, dividends, and settlement conventions |
| Borrow/locate | `BorrowAdapter` | Strict fee/inventory schema and validation | Supply borrower-side fee, inventory, locate result, utilization, recall, and dividend compensation |
| Events | `EventTapeAdapter` plus SEC filing events | SEC filing events are populated; generic revision-aware event contract is ready | Add earnings-calendar vintages, guidance, M&A, index, macro, and corporate-action announcement/revision tapes |

Official references: [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [FRED/ALFRED API](https://fred.stlouisfed.org/docs/api/fred/overview.html), [ALFRED vintages](https://fred.stlouisfed.org/docs/api/fred/alfred.html), [MarketData stock candles](https://www.marketdata.app/docs/api/stocks/candles/), and [MarketData option chains](https://www.marketdata.app/docs/api/options/chain/).

## Core clocks

| Field | Meaning |
|---|---|
| `effective_at` | Economic or market time described by the observation |
| `reported_at` | Issuer-reported time, when available |
| `filed_at` / `accepted_at` | Regulator receipt and acceptance |
| `available_at` | Earliest time the strategy may use the row |
| `vendor_observed_at` | Vendor snapshot or dissemination time |
| `ingested_at` | Engine receipt time |
| `valid_from` / `valid_to` | Reference-data validity interval |
| `realtime_start` / `realtime_end` | ALFRED vintage interval |
| `revision_id` | Source revision/restatement identity |

## Required identifier spine

The production security master must maintain validity-dated links among CRSP `PERMNO/PERMCO`, Compustat `GVKEY/IID`, SEC `CIK/accession`, I/B/E/S identifier, exchange listing identifier, OCC option root/contract, ticker/class, and issuer. Ticker is an attribute, never the permanent primary key.

## Fail-closed examples

- A March 31 fiscal period cannot be used on March 31 unless the filing was already public.
- An after-close filing cannot affect that close's signal.
- An amended filing changes only decisions after the amendment's `available_at`.
- A revised macro observation requires the ALFRED vintage visible on the decision date.
- A current or uncertain price bar never enters the confirmed historical cache.
- A historical option-chain midpoint is not labeled executable without contemporaneous OPRA NBBO.
- A generic 200 bps borrow charge is a sensitivity assumption, not evidence that a short was locatable.

