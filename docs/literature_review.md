# Literature review: cross-sectional alpha, nonlinear ML, and evidence gates

_Research cutoff: 2026-09-01. This is a design review, not evidence that the repository already contains a profitable strategy._

## Executive conclusion

The literature supports a disciplined nonlinear alpha engine, but not a claim that one model is reliably “very powerful.” The strongest recurring result is narrower: trees and neural networks can improve return forecasts by combining familiar signals—especially momentum, liquidity, volatility, value, profitability, and investment—in nonlinear ways. Individual-stock returns remain extremely noisy, reported gains are sensitive to the stock universe and portfolio construction, and many paper alphas weaken after publication or after realistic costs.

A credible engine therefore has four coupled parts:

1. a survivorship-free, point-in-time information set with an explicit `known_at` timestamp;
2. economically motivated candidate signals and nonlinear interactions;
3. nested, calendar-time out-of-sample validation that records every model and factor tried; and
4. portfolio construction that prices spread, market impact, borrow, financing, delay, and capacity before capital is assigned.

The practical recommendation is a tabular ensemble—regularized linear model, boosted trees, random forest, and a small neural network—over a common point-in-time feature store. Use LLMs primarily to structure filings and news, create frozen embeddings, and propose auditable hypotheses. Keep LLM-generated formulas outside the production path until they pass the same sealed-holdout and execution tests as human-designed factors.

## 1. What the cross-sectional literature actually supports

### 1.1 Requested accounting factors

These signals are useful priors for a joint model. They should not be treated as ten independent sources of alpha: several are alternate views of value, profitability, investment, or financing risk.

| Signal | Auditable construction at formation date | Mechanism and evidence | Main fragility |
|---|---|---|---|
| Book-to-market | point-in-time common book equity / current market equity; exclude or separately model nonpositive book equity | Value can reflect compensation for distress/duration risk or investor over-extrapolation. [Fama–French (1992)](https://doi.org/10.1111/j.1540-6261.1992.tb04398.x) find book-to-market and size summarize much of the historical cross-section; [Lakonishok, Shleifer, and Vishny (1994)](https://doi.org/10.1111/j.1540-6261.1994.tb04772.x) provide a behavioral interpretation. | Accounting definitions, intangibles, stale book values, financial-sector comparability, and long value drawdowns. |
| Earnings yield | trailing or forward earnings / price; keep trailing and I/B/E/S-forward versions distinct | A cheap price relative to earnings may compensate for risk or reverse excessive optimism/pessimism. [Basu (1977)](https://doi.org/10.1111/j.1540-6261.1977.tb01979.x) documents the historical low-P/E return relation; the signal overlaps heavily with book-to-market and cash-flow yield. | Negative or transitory earnings, one-off items, fiscal-year mismatch, and analyst-estimate revisions. |
| ROE | earnings available to common / lagged common equity; preferably quarterly and annual variants | High expected profitability raises the return on investment in the investment-based model. The q-factor model uses profitability/ROE and investment and prices many anomaly portfolios ([Hou, Xue, and Zhang, 2015](https://www.nber.org/papers/w18435)). | Small or negative equity can explode the ratio; leverage mechanically raises ROE. Pair with ROA, leverage, and an equity-denominator flag. |
| ROA | income or operating profit / lagged total assets | Measures operating quality with less direct leverage amplification than ROE; it is part of financial-strength screens such as [Piotroski (2000)](https://doi.org/10.2307/2672906). | Acquisition accounting, asset write-downs, banks, and denominator drift. |
| Gross profitability | (revenue − cost of goods sold) / lagged assets | Gross profits-to-assets has historically had predictive power comparable to book-to-market and improves value portfolios, including among large liquid stocks ([Novy-Marx, 2013](https://doi.org/10.1016/j.jfineco.2013.01.003)). | Sector accounting, missing/heterogeneous COGS, revenue recognition, and later Compustat restatements. |
| Debt-to-equity | total interest-bearing debt / common equity; also retain debt/assets and net debt/EBITDA | [Bhandari (1988)](https://doi.org/10.1111/j.1540-6261.1988.tb03952.x) documents a positive historical relation controlling for beta and size, but the sign is not structurally fixed: leverage can proxy for priced risk or for deteriorating quality and costly distress. | Negative equity, sector capital structures, refinancing regimes, and nonlinear distress. Model interactions with profitability, rates, maturity, and liquidity rather than imposing one sign. |
| Asset growth | (assets_t − assets_t−1) / assets_t−1 using reports known at the signal time | Firms investing aggressively have historically earned lower subsequent returns; [Cooper, Gulen, and Schill (2008)](https://doi.org/10.1111/j.1540-6261.2008.01370.x) document broad predictive power. Investment-based pricing gives a coherent expected-return channel. | M&A, divestitures, FX translation, accounting changes, and point-in-time predecessor mappings. |
| Sales growth | multi-year and recent sales growth, with acquisition and FX flags | High historical growth combined with rich valuation can identify “glamour” extrapolation; [Lakonishok, Shleifer, and Vishny (1994)](https://doi.org/10.1111/j.1540-6261.1994.tb04772.x) use sales growth in value/glamour tests. Recent positive sales news may instead be momentum. | Its sign depends on valuation, surprise, profitability, and horizon. Raw growth alone mixes organic growth, acquisitions, inflation, and recovery from a small base. |
| Accruals | preferably cash-based operating accruals; retain balance-sheet and cash-flow definitions, scaled by average assets | Earnings backed by cash are more persistent. [Sloan (1996)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2598) finds that prices historically underweighted the lower persistence of accruals. | Statement-of-cash-flows coverage, mergers, discontinued operations, restatements, and definition sensitivity. |
| Free-cash-flow yield | (cash from operations − capital expenditures) / enterprise value and / market equity, with separate maintenance/growth-capex proxies | A value-plus-quality signal: cheap claims on durable cash generation. Cash-flow-to-price belongs to the value family studied by [Lakonishok, Shleifer, and Vishny (1994)](https://doi.org/10.1111/j.1540-6261.1994.tb04772.x). | Lumpy capex, working-capital reversals, capitalized R&D, financial firms, and numerator/denominator timing. |

Implementation details matter more than cosmetic formula precision. Store the raw line items, denominator flags, report age, filing accession, and transformation version. Cross-sectionally rank or robustly scale features at each formation date; [Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009) map characteristic ranks to `[-1, 1]`, which limits outlier leverage and makes measurements more comparable through time.

### 1.2 Complementary signal families

| Family | Mechanism and useful evidence | Required data and implementation warning |
|---|---|---|
| Momentum/reversal | Medium-horizon momentum is documented by [Jegadeesh and Titman (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x). Slow diffusion and underreaction are plausible channels; short-horizon reversal often reflects liquidity provision. | CRSP or equivalent survivorship-free total returns; skip the most recent month for classic 12–2 momentum; model splits, dividends, halts, and bid-ask bounce. |
| Liquidity/volatility | Expected returns can include an illiquidity premium ([Amihud, 2002](https://doi.org/10.1016/S1386-4181(01)00024-6)); high idiosyncratic volatility has historically predicted low returns in many markets, though replication is sensitive to microcaps and construction. | Daily prices/volume for coarse measures; TAQ/NBBO for executable spread and impact. A feature can predict returns precisely because it is costly to trade. |
| Earnings/events | Analyst-based earnings surprises historically produce larger post-announcement drift than time-series surprises ([Livnat and Mendenhall, 2006](https://doi.org/10.1111/j.1475-679X.2006.00196.x)); crowded announcement days strengthen underreaction ([Hirshleifer, Lim, and Teoh, 2009](https://doi.org/10.1111/j.1540-6261.2009.01501.x)). | I/B/E/S point-in-time estimates and actuals, exact announcement timestamps, SEC filings, exchange calendars, and executable post-release quotes. Never treat the closing price before an after-hours release as tradable after the release. |
| Options | Put/call volume can reveal informed trading ([Pan and Poteshman, 2006](https://doi.org/10.1093/rfs/hhj024)); equity-option skew historically predicts underlying returns ([Xing, Zhang, and Zhao, 2010](https://doi.org/10.1017/S0022109009990644)). | Historical OPRA quotes/trades, option symbology, rates, dividends, corporate actions, open interest, and contemporaneous NBBO. End-of-day theoretical mids are not executable option P&L. |
| Borrow/short constraints | Expensive-to-borrow stocks can remain overpriced ([Jones and Lamont, 2002](https://www.nber.org/papers/w8494)). Recent broad evidence finds that borrower-side fees eliminate the average abnormal return of many long-short anomaly portfolios ([Muravyev, Pearson, and Pollet, 2025](https://doi.org/10.1111/jofi.13501)). | Daily borrower-side fee, lendable quantity, utilization, locate status, recall, and dividend compensation. A public short-interest snapshot is not a substitute for historical borrow availability and price. |
| Macro regimes | Rates, inflation, credit, growth, and volatility can condition the payoff and risk of firm characteristics. Nonlinear models can gate value, leverage, duration, and liquidity exposures by macro state. | Use ALFRED vintages, release timestamps, and survey calendars. Revised FRED history creates look-ahead bias. |
| Text | News and filings reveal information not captured by numerical ratios and can measure attention, tone, risk, competition, and narrative changes. | Archive the exact document, source timestamp, first-public timestamp, entity mapping, model/weights hash, prompt, and embedding. Training-corpus leakage is a distinct PIT problem. |

## 2. Why nonlinear models can help—and what the result does not prove

[Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009) compare linear, penalized, dimension-reduction, tree, and neural-network methods for U.S. stock return prediction. Trees and neural networks perform best in their study; the authors trace the improvement to nonlinear interactions, and the methods agree that momentum, liquidity, and volatility variants are dominant signals. Their economic result is important, while the scale should be interpreted correctly: stock-level out-of-sample predictability is small, and portfolio performance aggregates weak forecasts across many stocks.

The mechanism is credible because the accounting and market signals above are conditional:

- cheap plus profitable is different from cheap and deteriorating;
- high ROE produced by operating strength is different from high ROE produced by a tiny equity base;
- aggressive investment with weak profitability is different from investment supported by strong marginal returns;
- momentum in liquid, low-volatility names is cheaper and more scalable than momentum concentrated in distressed microcaps;
- high sales growth at a low valuation can signal improving fundamentals, while high sales growth at an extreme valuation can signal extrapolation;
- earnings/news signals should be stronger when attention and liquidity are low, but those same conditions raise execution cost;
- leverage has different implications under low refinancing stress and under a tightening credit regime.

Related work strengthens the case for structure, not for unconstrained complexity. [Kelly, Pruitt, and Su (2019)](https://doi.org/10.1016/j.jfineco.2019.05.001) use instrumented PCA to let characteristics govern time-varying factor loadings. [Freyberger, Neuhierl, and Weber (2020)](https://doi.org/10.1093/rfs/hhz123) estimate characteristic effects nonparametrically. [Chen, Pelger, and Zhu (2024)](https://doi.org/10.1287/mnsc.2023.4695) embed no-arbitrage moments and macro states in a deep model and report better out-of-sample pricing and investment metrics. The industry review by [Israel, Kelly, and Moskowitz (2020)](https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking-2Q19-Can-Machines-Learn-Finance.pdf) emphasizes finance's weak signals, correlated observations, short effective histories, and the value of economic restrictions.

Recommended model stack:

1. **Elastic net / Huber linear baseline.** Establishes whether nonlinear complexity adds incremental information and exposes sign instability.
2. **Gradient-boosted trees.** Natural main model for missing, ranked tabular features and threshold interactions.
3. **Random forest or extremely randomized trees.** Diversifies boosting's sequential fitting and is a useful stability check.
4. **Small multilayer perceptron.** Captures smooth high-order interactions; use strong regularization, early stopping, and seed ensembles.
5. **Optional structured latent-factor model.** IPCA or an economically constrained autoencoder for risk representation, evaluated separately from pure return forecasting.

Ensemble predictions only from inner-validation performance known before the test month. A deep model that wins one fixed historical split should not receive a discretionary override.

## 3. The factor zoo, decay, and conflicting replication evidence

The literature does not support selecting factors by published t-statistic and combining them unchanged.

| Study | Main result | Design implication |
|---|---|---|
| [Harvey, Liu, and Zhu (2016)](https://doi.org/10.1093/rfs/hhv059) | Hundreds of tested factors make the conventional `t > 2` hurdle too weak; they argue new discoveries generally need a hurdle above 3. | Maintain a complete experiment ledger and control false discoveries across factors, interactions, labels, universes, and hyperparameters. |
| [Hou, Xue, and Zhang (2020)](https://doi.org/10.1093/rfs/hhy131) | With NYSE breakpoints and value weighting, 286 of 447 anomalies (64%) are insignificant at 5%; 380 (85%) are below a t-statistic of 3. | Require performance outside microcaps and under implementable weights. |
| [Green, Hand, and Zhang (2017)](https://doi.org/10.1093/rfs/hhx019) | Of 94 characteristics, 12 are independent determinants in non-microcaps over 1980–2014, but only two remain after 2003 in their design. | Report modern-era and liquidity-segment results; a long historical average can hide decay. |
| [McLean and Pontiff (2016)](https://doi.org/10.1111/jofi.12365) | Returns of 97 predictors are 26% lower out of sample and 58% lower after publication. | Haircut research expectations and include post-publication tests. |
| [Linnainmaa and Roberts (2018)](https://www.nber.org/papers/w22894) | Many accounting anomalies weaken when tested before and after original samples. | Backward, forward, and international validation are stronger than another in-sample specification. |
| [Chen and Zimmermann (2022)](https://doi.org/10.17016/FEDS.2021.037) | Open-source reconstruction reproduces a large body of published predictors under paper-faithful definitions. | Reproduction and scientific/implementable replication are separate tests. Code fidelity does not establish live alpha. |
| [Jensen, Kelly, and Pedersen (2023)](https://doi.org/10.1111/jofi.13249) | Using 153 factors across 93 countries and a Bayesian model, they argue most economically grounded factor themes replicate; their result differs sharply from Hou–Xue–Zhang because the estimand, construction, and inference differ. | Treat the disagreement as evidence of design sensitivity. Pre-register breakpoints, weighting, microcap rules, benchmark, and statistical test. |
| [Feng, Giglio, and Xiu (2020)](https://www.nber.org/papers/w25481) | Many proposed factors are redundant after a high-dimensional set of existing factors is considered; profitability retains incremental power in their tests. | Evaluate marginal contribution to the existing ensemble, not standalone spread returns. |

The synthesis is utilitarian: factor families remain valuable economic priors and benchmarks, but reported anomaly returns are not a production forecast. A candidate earns promotion only when it improves a frozen incumbent after costs, in calendar-time out of sample, without relying on microcaps or a changing short book.

## 4. Trading costs and portfolio construction are part of the model

Execution can reverse model rankings. [Novy-Marx and Velikov (2016)](https://doi.org/10.1093/rfs/hhv063) find that a buy/hold spread—harder entry than retention—is an effective turnover-control device; most anomalies below 50% monthly turnover remain significant in their cost-mitigated construction, while few higher-turnover anomalies do. [Korajczyk and Sadka (2004)](https://doi.org/10.1111/j.1540-6261.2004.00656.x) show that equal-weight momentum looks best gross and worst after costs, while liquidity-aware weights support much greater capacity. Using large institutional trading records, [Frazzini, Israel, and Moskowitz](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini%2C%20Israel%20and%20Moskowitz.pdf) estimate materially lower costs and greater capacity than older effective-spread approaches. The difference is a warning: cost estimates depend on trader, execution skill, order size, market, and period.

Shorting requires a separate cost process. [Muravyev, Pearson, and Pollet (2025)](https://doi.org/10.1111/jofi.13501) report that borrower-side stock-loan fees eliminate average abnormal returns across 162 anomaly portfolios in their recent sample. Their paper also explains why lender revenue, ordinal “hard-to-borrow” scores, and stale average fee buckets understate the cost paid by an end borrower.

Portfolio optimization also overfits. [DeMiguel, Garlappi, and Uppal (2009)](https://doi.org/10.1093/rfs/hhm075) find that 14 optimized allocations do not consistently beat `1/N` out of sample across their datasets, illustrating how estimation error can dominate theoretical efficiency. [Brandt, Santa-Clara, and Valkanov (2009)](https://www.nber.org/papers/w10996) instead parameterize weights directly as functions of characteristics. [DeMiguel et al. (2020)](https://doi.org/10.1093/rfs/hhz085) show that combining signals can lower costs because offsetting trades cancel, so evaluating each factor in isolation can understate the benefit of a joint portfolio.

A research backtest should therefore solve, at every rebalance, a constrained objective such as expected alpha minus risk, turnover, impact, and borrow penalties. Minimum requirements:

- beta, sector, country, and style-exposure bounds;
- single-name, ADV participation, and gross/net exposure limits;
- a no-trade region and hysteresis;
- spread plus nonlinear impact calibrated from TAQ or live fills, not one universal basis-point charge;
- daily borrow fee and availability on every short, plus recall and dividend compensation;
- financing and cash return;
- delisting returns, halts, corporate actions, and stale-price rules;
- a fill timestamp and price rule that could have been executed with information then available;
- capacity curves at several AUM and participation assumptions.

Report long-only and long-short results separately. A signal may forecast the short leg yet be economically unusable once borrow is incorporated.

## 5. Point-in-time data requirements and leakage controls

Every observation should carry `entity_id`, `valid_from`, `valid_to`, `known_at`, `ingested_at`, `source_version`, and `revision_id`. Feature jobs may query only records with `known_at <= decision_time`. Current ticker strings are display fields, not join keys.

| Layer | Research-grade role | PIT/execution gate and primary documentation |
|---|---|---|
| CRSP | Survivorship-free prices, shares, distributions, exchange codes, delistings, and permanent security/company IDs | Include delisting returns and historical identifiers; Shumway documents the historical bias from missing negative delisting returns ([paper](https://www.tylergshumway.org/Shumway-DelistingBiasCRSP-1997.pdf)). CRSP defines delisting-return fields in its [official data guide](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/). |
| Compustat PIT | Standardized historical fundamentals, original/revised observations, and restatement history | Join the version available at decision time, not the latest value for an old fiscal period. S&P describes Compustat as historical and point-in-time data ([product document](https://www.spglobal.com/marketintelligence/en/documents/compustat-brochure_digital.pdf)). If PIT is unavailable, conservative fixed lags are a sensitivity test, not a replacement for observed availability. |
| SEC EDGAR | Raw filings, acceptance timestamps, XBRL facts, exhibits, 8-K events, and original text | The SEC APIs update submissions and XBRL throughout the day ([documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)). Preserve each accession and acceptance time. `companyfacts` aggregates later filings; reconstruct historical state from accessions and amendments rather than reading today's aggregate as if it existed historically. |
| I/B/E/S PIT | Analyst-level and consensus estimates, revisions, actuals, announcement dates, and dispersion | Use the snapshot or detail record known then, with split-adjustment state. LSEG describes an I/B/E/S PIT product with dated historical snapshots ([quant-research brochure](https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf)). Monthly history delivered today is not automatically PIT. |
| FRED/ALFRED | Macro series, release calendars, and data vintages | ALFRED records when values were released and later revised ([official overview](https://fred.stlouisfed.org/docs/api/fred/alfred.html)); query the real-time period known at each forecast date ([API guide](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)). |
| TAQ | Trades, quotes, NBBO, sale conditions, and intraday liquidity/impact | NYSE Daily TAQ includes consolidated trades, quotes, NBBO, and master files ([official catalog](https://www.nyse.com/data-products/catalog/daily-taq)). Apply quote/trade condition filters, exchange timestamp rules, crossed-market handling, and an executable arrival price. |
| Options/OPRA | Historical option trades/quotes, surface, skew, volume, open interest, and event-implied distributions | OPRA is the consolidated options data plan ([official site](https://www.opraplan.com/)); Cboe provides historical OPRA trade files ([catalog](https://datashop.cboe.com/option-trades)). Preserve series mappings, corporate actions, NBBO, rates, dividends, and American-exercise conventions. |
| Borrow | Daily fee paid by the borrower, supply, on-loan quantity, utilization, locate/recall state | S&P Securities Finance describes daily supply, demand, and fee history ([product page](https://www.spglobal.com/market-intelligence/en/solutions/products/securities-finance)). Do not backfill later fee estimates into an earlier locate decision. |
| Events | Earnings, guidance, dividends, M&A, offerings, buybacks, management changes, index events, macro releases | Store first-public timestamp, amendments, source document, entity/security mapping, trading session, and whether the release was scheduled. SEC filing date alone is too coarse for intraday decisions. |
| MarketData/Yahoo prototype feed | Rapid OHLCV and option prototyping | Useful for ingestion tests and provisional research. It does not replace CRSP's historical universe/delistings, Compustat/IBES PIT, TAQ execution, or borrow. Validate adjustments and keep provider/raw payload/version. |

Random train/test splits by stock are invalid because all stocks in a month share the same market state and often the same macro inputs. Split by calendar time; where labels overlap, purge overlapping training labels around validation boundaries and embargo any period required by the holding horizon.

## 6. Text and LLM alpha: evidence, opportunity, and special leakage risk

Textual alpha predates LLMs. [Tetlock (2007)](https://doi.org/10.1111/j.1540-6261.2007.01232.x) links media pessimism to temporary price pressure and volume. [Loughran and McDonald (2011)](https://doi.org/10.1111/j.1540-6261.2010.01625.x) show that general dictionaries misclassify financial language and build finance-specific categories. [Ke, Kelly, and Xiu (2019)](https://www.nber.org/papers/w26186) supervise a news sentiment representation directly on returns rather than importing a generic sentiment score. [Bybee, Kelly, Manela, and Xiu (2024)](https://www.nber.org/papers/w29344) extract interpretable news topics that summarize macroeconomic state.

Recent LLM results are promising but less mature:

- [Chen, Kelly, and Xiu, “Expected Returns and Large Language Models”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687) report that contextual news embeddings add information beyond technical and firm characteristics across multiple markets and languages. It remains a working paper and requires an independently reconstructed news archive and costed test.
- [Lopez-Lira and Tang](https://doi.org/10.1016/j.jfineco.2026.104335) use post-knowledge-cutoff headlines and find that advanced LLM scores capture immediate reactions and subsequent drift, especially for smaller stocks and negative news. Their published discussion distinguishes the largely non-tradable initial reaction from the potentially tradable later drift; this distinction should be preserved in any replication.
- [Glasserman and Lin (2024)](https://arxiv.org/abs/2309.17322) show that overlapping LLM training and backtest periods can create look-ahead concerns and that company knowledge can also distract sentiment measurement.
- [Sarkar et al., “Large Language Models: An Applied Econometric Framework”](https://doi.org/10.1146/annurev-economics-120925-105620) make no-training-leakage a condition for valid prediction inference.
- [He et al. (2025)](https://arxiv.org/abs/2502.21206) build chronologically consistent language models. [Kelly et al. (2026), “Scaling Point-in-Time Language Models”](https://www.nber.org/papers/w35247) train monthly checkpoints on chronologically filtered corpora and report that PIT embeddings retain economically useful signal. These provide a more defensible research pattern than querying a current closed model on old text.

Recommended LLM role:

1. parse each filing/news item into a versioned schema: event, direction, magnitude, horizon, uncertainty, affected entities, and quoted evidence spans;
2. compute frozen PIT embeddings from a documented checkpoint;
3. feed those features to the same tabular forecasting stack as accounting and market data;
4. keep prompts, model weights/hash, tokenizer, sampling parameters, raw outputs, and retry policy;
5. require deterministic or seed-ensembled scoring and test prompt/model drift;
6. anonymize issuer identifiers as a robustness check, not as the sole leakage remedy;
7. never let a current model's narrative about a historical company enter a PIT backtest unless the checkpoint's training cutoff is proven.

LLMs are also useful hypothesis generators, literature assistants, and code reviewers. Those uses still increase the number of tried strategies, so every generated factor counts in the multiple-testing budget.

### Assessment of the cited ACL 2026 CogAlpha paper

[Liu et al., “Cognitive Alpha Mining via LLM-Driven Code-Based Evolution”](https://aclanthology.org/2026.acl-long.538.pdf) is relevant as a candidate-generation system. It uses 21 specialized agents, iterative mutation/crossover, code checks, and predictive-metric feedback to create interpretable OHLCV formulas. The paper uses chronological train/validation/test windows, evaluates several markets and horizons, and reports for CSI 300 a test AER of 16.39% and IR of 1.90 for a top-50/drop-5 Qlib portfolio. Its appendix assumes 5 bp opening cost, 15 bp closing cost, and a CNY 5 minimum fee.

Those figures are research claims, not a production expectation. Important unresolved gates are:

- thousands of adaptively generated candidates create a large search/multiple-testing problem; the reported analysis does not provide a deflated Sharpe ratio, probability of backtest overfitting, or family-wise test over the evolutionary search;
- the paper documents one fixed train/validation/test split per market, not nested rolling retraining or a never-touched final holdout after framework development;
- its Qlib/Yahoo datasets are not documented as point-in-time constituent universes with delisting returns and historical identifier reconstruction;
- fixed proportional costs do not model spread, opening-auction availability, nonlinear impact, borrow, recall, or capacity;
- its U.S. S&P 500 test stops in 2020, limiting evidence about recent regimes;
- the authors explicitly state that Qlib simulation may not match live conditions and leave live-trading validation to future work.

Use CogAlpha behind a research firewall: it may propose code and economic rationales using training data only. Deduplicate candidates by return correlation and syntax tree; assess them in an inner validation loop; count every proposal as a trial; and allow only a frozen, selected set to touch the outer test. It does not replace the GKY-style supervised benchmark or the PIT data layer.

## 7. Realistic out-of-sample validation protocol

1. **Freeze the decision.** Specify universe, forecast timestamp, target horizon, label, rebalance rule, benchmark, costs, and capacity assumptions before reading results.
2. **Build an availability ledger.** An automated test should fail if any feature's `known_at` exceeds the forecast timestamp. Run synthetic time-travel tests for revisions, amendments, ticker changes, and after-hours releases.
3. **Use an eligible PIT universe.** Include dead and delisted securities, historical share classes, exchange/price/liquidity rules, and identifiers known then. Report performance with and without microcaps and penny stocks.
4. **Nest model selection in time.** Example: expanding training window, a later contiguous inner-validation window for feature/model/hyperparameter selection, and the next month or quarter as the outer test. Roll forward without revisiting past decisions.
5. **Purge overlap.** If a ten-day target begins at day `t`, remove training labels whose realization interval overlaps validation/test information, and embargo around the split.
6. **Seal a final holdout.** Use it once after the research protocol, costs, and ensemble are frozen. Further tuning converts it into validation and requires a new holdout.
7. **Track every trial.** Factor definitions, generated formulas, transformations, universes, labels, seeds, prompts, and hyperparameters all count. Apply the [White Reality Check](https://doi.org/10.1111/1468-0262.00152), false-discovery controls, the [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), and/or [probability of backtest overfitting](https://escholarship.org/uc/item/4w1110bb) as complementary diagnostics.
8. **Use date-aware inference.** Report time-series t-statistics on portfolio returns with autocorrelation-robust errors and block bootstrap confidence intervals. Stock-month rows do not create millions of independent observations; calendar dates are the independent economic units.
9. **Benchmark broadly.** Compare with zero, market, sector-neutral value-weighting, simple factor composites, elastic net, trees, and the incumbent production model. Measure incremental alpha after standard factor exposures.
10. **Report forecasts and portfolios.** OOS `R²`, Pearson/Spearman IC, calibration, top-minus-bottom spread, long and short legs, hit rate, turnover, drawdown, tail loss, beta/style/sector exposures, and gross/net performance. A high IC is insufficient if it is concentrated in untradeable names.
11. **Stress implementation.** At least three spread/impact levels, borrower-side fee shocks, one-bar and one-day delays, missed fills, ADV/AUM capacity curves, and a no-short/long-only variant.
12. **Test stability.** Pre/post-publication, pre/post-2003, recessions, inflation/rate regimes, high/low volatility, large/small stocks, liquid/illiquid stocks, sectors, and international markets. Signs and interaction shapes should be economically explainable.
13. **Shadow before live use.** Record untouched forward predictions, intended orders, available borrow, and executable quotes for several months. Promotion requires a predeclared threshold on net information ratio, drawdown, exposure drift, and forecast decay.

## 8. Highest-value research sequence

1. **Auditable monthly tabular baseline.** CRSP + Compustat PIT + ALFRED; requested accounting factors plus momentum, liquidity, volatility, size, beta, issuance, and age/staleness controls. Compare elastic net, LightGBM, random forest, and a small MLP.
2. **Cost-aware joint portfolio.** Add TAQ-calibrated spread/impact and daily borrow. Optimize all signals jointly with a no-trade region. This test determines whether apparent short-leg and microcap alpha survives.
3. **Earnings/event overlay.** I/B/E/S PIT + EDGAR acceptance-time features; model 1–5 day drift separately from the monthly cross-section. Use exact release-to-fill timing.
4. **PIT text overlay.** Frozen chronological embeddings and structured filing/news changes; prove incremental net value over tabular/event features.
5. **Options overlay.** OPRA skew, volume imbalance, implied/realized volatility, and event surface features. Keep underlying-return prediction separate from an option-P&L backtest until historical option NBBO and fill rules exist.
6. **LLM factor evolution sandbox.** CogAlpha-like generation only after the core pipeline records trials, deduplicates factors, and enforces nested holdouts.

The go/no-go statistic is not the best backtest Sharpe. It is the lower confidence bound on net, capacity-adjusted performance of a frozen process in outer walk-forward tests and then in untouched shadow trading.
