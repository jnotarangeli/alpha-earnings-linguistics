# alpha-earnings-linguistics

An equity alpha signal derived from linguistic analysis of earnings call transcripts. The pipeline extracts features capturing executive evasiveness, scripting patterns, sentiment tone, and readability from earnings calls of the top 500 US stocks by market cap, then uses an expanding-window LightGBM model to predict 30-day and 90-day forward returns. Factor spanning regressions confirm the signal delivers alpha beyond the Fama-French 5-factor model, momentum, and AQR Quality Minus Junk (QMJ).

## Research Hypothesis

Executives reveal private information through *how* they speak, not just *what* they say. Managers who deflect analyst questions, rely on scripted Q&A responses, or use unusually positive tone relative to fundamentals are signaling information asymmetry. These linguistic patterns predict future stock returns, particularly among firms with low analyst coverage where such signals are least likely to be arbitraged away.

## Data Sources

All raw data files are stored locally and excluded from version control.

| Source | WRDS Table / Dataset | Description |
|---|---|---|
| Capital IQ Transcripts | `ciq.ciqtranscriptcomponent`, `ciq.ciqtranscript` | Earnings call transcripts (presenter speech, Q&A) for top-500 US stocks |
| CRSP | `crsp.msf`, `crsp.msenames` | Monthly stock returns and market cap (common stocks, share codes 10/11) |
| Compustat | `comp.fundq` | Quarterly EPS for earnings surprise (SUE) construction |
| IBES | `ibes.statsumu_epsus` | Analyst EPS estimates and coverage count per firm-quarter |
| CRSP/CCM | `crsp_a_ccm.ccmxpf_lnkhist` | Compustat gvkey → CRSP permno point-in-time linking |
| Fama-French | `ff.fivefactors_monthly` | MKT-RF, SMB, HML, RMW, CMA, RF (monthly) |
| AQR Data Library | QMJ Factors (US, monthly) | Quality Minus Junk factor for spanning regressions |

## Pipeline Overview

### `earnings_call_signal.ipynb` — Main Signal Notebook

1. **Data Acquisition** — Load cached Capital IQ transcript data (pulled via WRDS). Universe: top-500 US stocks by market cap; transcripts assembled for the 200 most active companies (2020–2024). Component types: presenter speech (MD&A) and Q&A (questions + answers). Data cached as `.parquet` after the first WRDS pull.

2. **Linguistic Feature Extraction** — Compute features from transcript text using the `ling_features` package and custom implementations:

   | Feature | Method | Hypothesis |
   |---|---|---|
   | **Non-answer rate** | `ling_features.non_answers()` — regex-based detection of refusals, deferrals, evasions (Gow et al. 2021) | High evasion → bad news concealment → negative future returns |
   | **Scripting score** | TF-IDF cosine similarity between MD&A and Q&A portions (Lee 2016) | High scripting → lack of spontaneity → negative signal |
   | **LM Sentiment** | `ling_features.tone_count()` — Loughran-McDonald positive/negative word counts | Net tone captures managerial optimism or pessimism |
   | **Fog Readability** | `ling_features.fog()` — Gunning Fog index | Complexity as obfuscation signal |
   | **Forward-looking statements** | `ling_features.fls()` — sentence-level classification | Forward-looking language density |

3. **Additional Feature Engineering**
   - **Tone surprise**: residual of tone regressed on earnings surprise (SUE) — captures tone beyond what fundamentals explain
   - **Delta features**: quarter-over-quarter change in non-answer rate, tone, and scripting score
   - **Structural features**: transcript length, Q&A share, number of analyst questions
   - **Cross-sectional rank normalization**: all features rank-transformed to [0, 1] each month

4. **CRSP, Compustat & IBES Pull** — Monthly returns and market cap from CRSP; quarterly EPS for SUE from Compustat; analyst coverage count from IBES (used to define the low-coverage sub-universe).

5. **ML Prediction** — Expanding-window `LightGBMRegressor` predicting 30-day and 90-day forward returns. OOS from 2022, retrained every 6 months. Run on two universes: full sample and low-analyst-coverage sub-universe (≤ median analyst count).

6. **Portfolio Construction** — Quintile long-short portfolios (Q5 − Q1), equal- and value-weighted, for both universes and both return horizons.

7. **Factor Spanning** — Newey-West HAC regressions of the L/S return on CAPM → FF3+MOM → FF5+MOM → FF5+MOM+QMJ to test signal independence.

## Key Results

- The linguistic signal generates positive out-of-sample alpha, particularly at the 90-day horizon.
- Alpha persists after controlling for market, size, value, profitability, investment, momentum, and quality factors.
- The signal is strongest among stocks with low analyst coverage, consistent with an information-asymmetry story.
- Top predictive features include scripting score, net tone, and non-answer rate.

## Requirements

- **WRDS account** with access to CRSP, Compustat, IBES, Fama-French, and Capital IQ Transcripts
- Python 3.9+
- Key packages: `pandas`, `numpy`, `lightgbm`, `statsmodels`, `wrds`, `ling_features`, `nltk`, `scikit-learn`, `matplotlib`, `pyarrow`

Install dependencies:
```bash
pip install "pyarrow==17.0.0" "pandas>=2.2,<2.3" ling_features nltk lightgbm statsmodels scikit-learn wrds matplotlib
```

## How to Run

1. Ensure your WRDS credentials are configured (via `~/.pgpass` or environment variables).
2. Update `WRDS_USER` and `DATA_DIR` in the first code cell of `earnings_call_signal.ipynb`.
3. Run the notebook end-to-end. All data is cached as `.parquet` files after the first WRDS pull — subsequent runs skip the database queries.

## References

- Gow, I. D., Larcker, D. F., & Zakolyukina, A. A. (2021). Non-Answers During Conference Calls. *Journal of Accounting Research*.
- Lee, J. (2016). Can Investors Detect Managers' Lack of Spontaneity? Adherence to Predetermined Scripts during Earnings Conference Calls. *The Accounting Review*.
- Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*.
