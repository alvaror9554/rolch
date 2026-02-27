"""Oil and finance covariate application using ondil Student-t copula.

This script downloads real market data, builds a clean modeling dataset,
fits `MultivariateOnlineDistributionalRegressionPath` with
`BivariateCopulaStudentT`, and writes forecast-ready outputs.

Usage examples:
    python examples/oil_finance_application.py --start 2018-01-01 --end 2026-01-01
    python examples/oil_finance_application.py --input-csv data/market_data.csv

CSV mode expects a `date` column plus price columns for the selected tickers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ondil.distributions import BivariateCopulaStudentT
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath
from ondil.links import FisherZLink, Identity, KendallsTauToParameter, LogShiftTwo


@dataclass
class AppConfig:
    start: str
    end: str
    oil_ticker: str
    benchmark_ticker: str
    covariate_tickers: list[str]
    train_fraction: float
    input_csv: str | None
    output_csv: str


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Oil + covariates forecasting application")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--oil-ticker", default="BZ=F", help="Oil price ticker (Yahoo Finance)")
    parser.add_argument("--benchmark-ticker", default="^GSPC", help="Second response series for bivariate copula")
    parser.add_argument(
        "--covariates",
        default="DX-Y.NYB,^VIX,^TNX,GC=F,EURUSD=X",
        help="Comma-separated Yahoo Finance covariate tickers",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--input-csv", default=None, help="Optional local prices CSV")
    parser.add_argument("--output-csv", default="oil_finance_forecast_output.csv")
    args = parser.parse_args()

    covariates = [ticker.strip() for ticker in args.covariates.split(",") if ticker.strip()]
    return AppConfig(
        start=args.start,
        end=args.end,
        oil_ticker=args.oil_ticker,
        benchmark_ticker=args.benchmark_ticker,
        covariate_tickers=covariates,
        train_fraction=args.train_fraction,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )


def load_prices(config: AppConfig) -> pd.DataFrame:
    tickers = [config.oil_ticker, config.benchmark_ticker, *config.covariate_tickers]
    unique_tickers = list(dict.fromkeys(tickers))

    if config.input_csv is not None:
        prices = pd.read_csv(config.input_csv)
        if "date" in prices.columns:
            prices["date"] = pd.to_datetime(prices["date"])
            prices = prices.set_index("date")
        missing = [ticker for ticker in unique_tickers if ticker not in prices.columns]
        if missing:
            raise ValueError(f"Missing tickers in input CSV: {missing}")
        return prices[unique_tickers].sort_index()

    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError(
            "yfinance is required for online download. Install with: pip install yfinance pandas"
        ) from error

    raw = yf.download(
        unique_tickers,
        start=config.start,
        end=config.end,
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError("No data downloaded. Check ticker symbols and date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_block = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.levels[0][0]
        prices = raw[price_block].copy()
    else:
        prices = raw.to_frame(name=unique_tickers[0])

    return prices[unique_tickers].dropna().sort_index()


def to_pseudo_observations(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average")
    return (ranks / (len(values) + 1.0)).to_numpy()


def build_dataset(prices: pd.DataFrame, config: AppConfig) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    returns = np.log(prices).diff()
    returns.columns = [f"ret_{column}" for column in returns.columns]

    oil_col = f"ret_{config.oil_ticker}"
    benchmark_col = f"ret_{config.benchmark_ticker}"
    cov_cols = [f"ret_{ticker}" for ticker in config.covariate_tickers]

    model_df = pd.DataFrame(index=returns.index)
    model_df["oil_target"] = returns[oil_col].shift(-1)
    model_df["benchmark_target"] = returns[benchmark_col].shift(-1)
    for covariate in cov_cols:
        model_df[covariate] = returns[covariate]

    model_df = model_df.dropna()

    y = np.column_stack(
        [
            to_pseudo_observations(model_df["oil_target"]),
            to_pseudo_observations(model_df["benchmark_target"]),
        ]
    )
    x = model_df[cov_cols].to_numpy()
    return x, y, model_df


def fit_model(x: np.ndarray, y: np.ndarray) -> MultivariateOnlineDistributionalRegressionPath:
    horizon_dim = 2
    equation = {
        0: {h: np.arange(x.shape[1]) for h in range(horizon_dim)},
        1: {0: "intercept"},
    }

    distribution = BivariateCopulaStudentT(
        link_1=FisherZLink(),
        param_link_1=KendallsTauToParameter(),
        link_2=LogShiftTwo(),
        param_link_2=Identity(),
    )

    estimator = MultivariateOnlineDistributionalRegressionPath(
        distribution=distribution,
        equation=equation,
        method="ols",
        early_stopping=False,
        early_stopping_criteria="bic",
        iteration_along_diagonal=False,
        verbose=1,
        max_iterations_inner=10,
        max_iterations_outer=100,
        scale_inputs=False,
    )
    estimator.fit(x, y)
    return estimator


def main() -> None:
    config = parse_args()
    prices = load_prices(config)
    x, y, model_df = build_dataset(prices, config)

    split_index = int(len(x) * config.train_fraction)
    x_train, y_train = x[:split_index], y[:split_index]
    x_test, y_test = x[split_index:], y[split_index:]

    estimator = fit_model(x_train, y_train)
    estimator.update(x_test, y_test)

    nu_raw = float(estimator.coef_[1][0][0][0])
    nu_value = float(np.exp(nu_raw) + 2.0)

    output = model_df.iloc[split_index:].copy()
    output["nu_raw"] = nu_raw
    output["nu"] = nu_value
    output.to_csv(config.output_csv)

    print("=" * 80)
    print("Oil and covariates application completed")
    print("=" * 80)
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Estimated nu (constant in this setup): {nu_value:.4f}")
    print(f"Saved output: {Path(config.output_csv).resolve()}")


if __name__ == "__main__":
    main()
