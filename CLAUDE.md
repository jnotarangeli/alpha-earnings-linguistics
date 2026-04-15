# CLAUDE.md — alpha-earnings-linguistics

## What This Project Is
An equity alpha signal pipeline that extracts linguistic features from earnings call transcripts (Capital IQ via WRDS) — including executive evasiveness, scripting patterns, sentiment tone, and readability — and trains an expanding-window LightGBM model to predict 30-day and 90-day forward stock returns. The signal is evaluated via quintile portfolios and factor spanning regressions against FF5+MOM+QMJ. Analysis runs on the full universe and a low-analyst-coverage sub-universe.

## Python Stack
- `pandas`, `numpy` — data manipulation
- `wrds` — WRDS database access (CRSP, Compustat, IBES, Capital IQ, Fama-French)
- `ling_features` — linguistic feature extraction (non-answer rate, tone counts, Fog index, FLS classification)
- `nltk` — tokenization and NLP utilities (required by `ling_features`)
- `sklearn.feature_extraction.text.TfidfVectorizer` — scripting score (cosine similarity between MD&A and Q&A)
- `lightgbm` — gradient boosting ML model (`LGBMRegressor`)
- `scikit-learn` — `QuantileTransformer`, permutation importance
- `statsmodels` — OLS with Newey-West HAC standard errors
- `matplotlib` — charting (presentation-style, mirrors patent notebook)
- `pyarrow` — parquet I/O (pin to `pyarrow==17.0.0` for pandas 2.2 compatibility)

## Data Sources
Raw data lives **locally only** and is never committed to version control:
- `transcript_detail_top500.parquet` (~1.95 GB) — CIQ transcript components
- `transcript_meta_top500.parquet` — CIQ transcript metadata
- `transcripts_assembled.parquet` (~394 MB) — assembled MD&A + Q&A text
- `ciq_gvkey.parquet`, `ciq_ticker.parquet` — CIQ company identifier mappings
- `crsp_monthly.parquet` — CRSP monthly returns
- `compustat_q.parquet` — Compustat quarterly fundamentals
- `ibes_coverage.parquet` — IBES analyst coverage
- `ccm_link.parquet` — CCM gvkey → permno linkage
- `ff_factors.parquet` — Fama-French 5 factors + momentum
- `linguistic_features.parquet` — cached feature extraction output

## Files to NEVER Commit
*.parquet, *.csv, *.xlsx, *.png, *.jpg, *.pdf, *.svg, .env
Data/, data/, figures/, output/, Old/, .venv/

## Coding Conventions
- Vectorized operations only — no `iterrows()` or Python loops over DataFrames
- Cross-sectional operations use `.groupby().transform()` or `.groupby().apply()`
- All date alignment to month-end via `pd.offsets.MonthEnd(0)` before merging
- Feature names follow `snake_case`; rank-normalized versions use `_rank` suffix
- Follow existing notebook cell structure and naming conventions (mirrors patent_signal.ipynb)
- `pyarrow` must be pinned to `==17.0.0` — do not upgrade without testing pandas compatibility

## Session Start
1. Open `earnings_call_signal.ipynb` — this is the main entry point
2. Set `WRDS_USER` and `DATA_DIR` in the first code cell
3. All heavy data (transcripts, CRSP, Compustat, IBES) is cached as `.parquet` — subsequent runs skip WRDS queries
4. If `pyarrow` was recently changed, restart the kernel before re-running
