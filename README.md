# Earnings-Call Linguistics Alpha Signal

An equity alpha signal derived from linguistic analysis of earnings call transcripts. The pipeline extracts features like executive evasiveness, scripting patterns, and sentiment tone from S&P 500 earnings calls, then uses machine learning to predict forward stock returns.

## Pipeline Overview

1. **Data Acquisition** — Pull earnings call transcripts (Capital IQ via WRDS), CRSP monthly returns, Compustat fundamentals, and IBES analyst coverage.
2. **Linguistic Feature Extraction** — Compute features from transcript text:
   - **Non-answer rate** (Gow, Larcker & Zakolyukina 2021) — detects executive evasions and deflections during Q&A
   - **Scripting score** (Lee 2016) — TF-IDF cosine similarity between prepared remarks and Q&A responses
   - **LM Sentiment** — Loughran-McDonald positive/negative word counts and net tone
   - **Fog Readability** — Gunning Fog index measuring linguistic complexity
   - **Forward-looking statements** — proportion of forward-looking language
3. **Additional Feature Engineering** — Tone surprise (residual vs. earnings surprise), delta features (change vs. prior quarter), cross-sectional rank normalization.
4. **ML Prediction** — Expanding-window LightGBM predicting 30-day and 90-day forward returns, retrained every 12 months.
5. **Portfolio Construction** — Quintile long/short portfolios (Q5 − Q1), equal- and value-weighted.
6. **Factor Spanning** — Fama-French 5-factor + Momentum regressions with Newey-West standard errors to test whether the signal delivers alpha beyond standard risk factors.

## Key Results

- The linguistic signal generates positive out-of-sample alpha, particularly at the 90-day horizon.
- Alpha persists after controlling for market, size, value, profitability, investment, and momentum factors.
- The signal is strongest among stocks with low analyst coverage, consistent with an information-asymmetry story.
- Top predictive features include scripting score, net tone, and non-answer rate.

## Requirements

- **WRDS account** with access to CRSP, Compustat, IBES, and Capital IQ Transcripts
- Python 3.9+
- Key packages: `pandas`, `numpy`, `lightgbm`, `statsmodels`, `wrds`, `ling_features`, `nltk`, `scikit-learn`, `matplotlib`, `pyarrow`

Install dependencies (from the notebook):
```bash
pip install "pyarrow==17.0.0" "pandas>=2.2,<2.3" ling_features nltk lightgbm statsmodels scikit-learn wrds matplotlib
```

## Usage

1. Ensure your WRDS credentials are configured (via `~/.pgpass` or environment variables).
2. Update the `WRDS_USER` and `DATA_DIR` variables in the first code cell of `earnings_call_signal.ipynb`.
3. Run the notebook end-to-end. Data is cached as `.parquet` files after the first run, so subsequent runs skip the WRDS queries.

## References

- Gow, I. D., Larcker, D. F., & Zakolyukina, A. A. (2021). Non-Answers During Conference Calls. *Journal of Accounting Research*.
- Lee, J. (2016). Can Investors Detect Managers' Lack of Spontaneity? Adherence to Predetermined Scripts during Earnings Conference Calls. *The Accounting Review*.
- Loughran, T. & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*.

## License

This project is for academic and research purposes.
