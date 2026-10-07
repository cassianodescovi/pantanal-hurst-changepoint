"""
01_preprocessing.py

Builds the cleaned daily stage series and the annual-mean series for the
Ladario station (ANA code 66825000) from the raw Hidroweb export.

Input  : data/raw/66825000_Cotas.csv     (not distributed; see data/README.md)
Output : data/processed/daily_stage_ladario.csv
         data/processed/annual_mean_stage_ladario.csv

Steps
  1. Read the Hidroweb export (latin-1, ';' separated, metadata block before the table).
  2. Keep daily-summary rows only (MediaDiaria == 1).
  3. For months with both raw and consistency-reviewed versions, keep the version with
     more non-missing daily values (ties go to the consistency-reviewed version).
  4. Reshape from one row per month (Cota01..Cota31) to one row per calendar day.
  5. Fill short gaps by linear interpolation (first MAX_GAP_DAYS days of each gap).
  6. Compute the daily climatology (mean by day of year) and the anomaly series.
  7. Aggregate to annual means.

Usage
  python scripts/01_preprocessing.py [--raw PATH] [--out DIR] [--end YYYY-MM-DD] [--max-gap N] [--fill-mode MODE]

The series is trimmed at --end (default 2025-12-31, the window analysed in the manuscript),
so that records added to the Hidroweb export after that date, including a partial final
year, do not enter the analysis. --max-gap sets the longest gap (in days) that is filled by
linear interpolation (default 3); --max-gap 0 leaves missing days unfilled, so that annual
means are computed over the available days only. --fill-mode pandas-limit reproduces
pandas' interpolate(limit=N), the default and the mode used for the manuscript: the first N
days of EVERY gap are filled, including gaps longer than N. The stricter mode "gaps" fills only
gaps of at most N days (the difference is one day of one four-day gap in this record).
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATION = "66825000"
MAX_GAP_DAYS = 3

DAY_TAGS = [f"{d:02d}" for d in range(1, 32)]
COTA_COLS = [f"Cota{d}" for d in DAY_TAGS]
STATUS_COLS = [f"Cota{d}Status" for d in DAY_TAGS]


def read_hidroweb(path):
    """Read the Hidroweb export, locating the table header automatically."""
    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if line.startswith("EstacaoCodigo"):
                header_line = i
                break
        else:
            raise ValueError(
                "Header line starting with 'EstacaoCodigo' not found. "
                "Is this a Hidroweb stage ('Cotas') export?"
            )

    df = pd.read_csv(path, encoding="latin-1", sep=";", skiprows=header_line, decimal=",")
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
    df[COTA_COLS + STATUS_COLS] = df[COTA_COLS + STATUS_COLS].apply(pd.to_numeric, errors="coerce")

    codes = df["EstacaoCodigo"].astype(str).unique()
    if list(codes) != [STATION]:
        print(f"WARNING: station codes in file {list(codes)} differ from expected {STATION}.")
    return df


def select_monthly_records(df):
    """Keep one daily-summary row per station-month."""
    df = df[df["MediaDiaria"] == 1].copy()
    df["n_valid"] = df[COTA_COLS].notna().sum(axis=1)
    df = (
        df.sort_values(["n_valid", "NivelConsistencia"], ascending=[False, False])
        .drop_duplicates(subset=["EstacaoCodigo", "Data"], keep="first")
    )
    if df.duplicated(subset=["EstacaoCodigo", "Data"]).any():
        raise RuntimeError("Duplicate station-months remain after selection.")
    return df


def to_daily(df):
    """Reshape one-row-per-month (Cota01..Cota31) into one row per calendar day."""
    stage = df.melt(id_vars=["Data"], value_vars=COTA_COLS, var_name="col", value_name="stage_cm")
    status = df.melt(id_vars=["Data"], value_vars=STATUS_COLS, var_name="col", value_name="status")
    stage["day"] = stage["col"].str.extract(r"(\d+)", expand=False).astype(int)
    status["day"] = status["col"].str.extract(r"(\d+)", expand=False).astype(int)

    daily = stage.merge(status[["Data", "day", "status"]], on=["Data", "day"])
    daily["date"] = pd.to_datetime(
        {"year": daily["Data"].dt.year, "month": daily["Data"].dt.month, "day": daily["day"]},
        errors="coerce",  # e.g., 31 April or 30 February become NaT and are dropped
    )
    daily = daily.dropna(subset=["date"]).sort_values("date").set_index("date")
    daily = daily[["stage_cm", "status"]]

    if not daily.index.is_unique:
        raise RuntimeError("Duplicate dates in the daily series.")

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    absent = full_range.difference(daily.index)
    if len(absent) > 0:
        print(f"WARNING: {len(absent)} calendar dates absent from the export; added as missing.")
    return daily.reindex(full_range).rename_axis("date")


def fill_short_gaps(series, max_gap):
    """Linearly interpolate only runs of at most `max_gap` consecutive missing days."""
    is_na = series.isna()
    run_id = (is_na != is_na.shift(fill_value=False)).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_len <= max_gap)

    interpolated = series.interpolate(method="linear", limit_area="inside")
    filled = series.copy()
    filled[fillable] = interpolated[fillable]
    return filled, fillable


def fill_pandas_limit(series, limit):
    """Same as series.interpolate(method="linear", limit=limit): fills the first `limit`
    days of every gap, including the first days of gaps longer than `limit`."""
    filled = series.interpolate(method="linear", limit=limit)
    return filled, series.isna() & filled.notna()


def main():
    parser = argparse.ArgumentParser(description="Preprocess the Ladario stage record.")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / f"{STATION}_Cotas.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--end", default="2025-12-31", help="last date kept (default: 2025-12-31)")
    parser.add_argument("--fill-mode", choices=["gaps", "pandas-limit"], default="pandas-limit",
                        help="pandas-limit (default, as used in the manuscript): fill the first max-gap days of every gap; gaps: fill only gaps <= max-gap")
    parser.add_argument("--max-gap", type=int, default=MAX_GAP_DAYS,
                        help=f"longest gap filled by interpolation, in days (default: {MAX_GAP_DAYS}; 0 = no filling)")
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(
            f"Input file not found: {args.raw}\n"
            "Download the station file from Hidroweb and place it there (see data/README.md)."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    daily = to_daily(select_monthly_records(read_hidroweb(args.raw)))

    end = pd.Timestamp(args.end)
    if daily.index.max() > end:
        print(f"Trimming the series at {end.date()} (the file extends to {daily.index.max().date()}).")
        daily = daily.loc[:end]

    # Days with an observed stage value but no ANA status flag (mostly recent, not yet
    # reviewed). Computed before gap filling so interpolated days are not counted.
    daily["pending_review"] = daily["status"].isna() & daily["stage_cm"].notna()
    daily["status"] = daily["status"].astype("Int64")

    n_missing = int(daily["stage_cm"].isna().sum())
    fill = fill_short_gaps if args.fill_mode == "gaps" else fill_pandas_limit
    daily["stage_cm"], daily["interpolated"] = fill(daily["stage_cm"], args.max_gap)
    n_remaining = int(daily["stage_cm"].isna().sum())

    # Daily climatology and anomalies (day of year; day 366 exists only in leap years)
    climatology = daily.groupby(daily.index.dayofyear)["stage_cm"].transform("mean")
    daily["anomaly_cm"] = daily["stage_cm"] - climatology

    grouped = daily["stage_cm"].groupby(daily.index.year)
    annual = pd.DataFrame({"stage_cm": grouped.mean(), "n_days": grouped.count()})
    annual.index.name = "year"
    incomplete = annual[annual["n_days"] < 360]
    if len(incomplete):
        print("WARNING: incomplete years in the annual series (annual means are not comparable):")
        print(incomplete.to_string())

    daily.to_csv(args.out / "daily_stage_ladario.csv")
    annual.to_csv(args.out / "annual_mean_stage_ladario.csv")

    n = len(daily)
    n_real = int((daily["status"] == 1).sum())
    print(f"Daily series: {daily.index.min().date()} to {daily.index.max().date()} ({n:,} days)")
    status_counts = {("no flag" if pd.isna(k) else int(k)): int(v)
                     for k, v in daily["status"].value_counts(dropna=False).items()}
    print(f"Status flags (1=real, 2=estimated): {status_counts}")
    print(f"Real-flag share: {100 * n_real / n:.2f}%")
    print(f"Missing stage values: {n_missing}; filled ({args.fill_mode}, max-gap {args.max_gap}): "
          f"{int(daily['interpolated'].sum())}; remaining missing: {n_remaining}")
    print(f"Days pending review (stage present, no status flag): {int(daily['pending_review'].sum())}")
    print(f"Annual series: {len(annual)} years ({annual.index.min()}-{annual.index.max()})")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()