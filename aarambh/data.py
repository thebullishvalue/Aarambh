"""Loading and cleaning the input table.

Rules, in order:
  1. Dates are parsed with one format for the whole column (``parse_dates``).
  2. Rows are sorted by date; a repeated date keeps only its first row.
  3. Text numbers such as ``6.48%`` or ``24,536.60`` become numbers.
  4. Rows with no target value are dropped (the target is never invented).
  5. Predictor gaps are forward-filled only. Nothing is ever back-filled: a value
     from the future must not appear in the past.
  6. Leading rows where a selected predictor has no data yet are trimmed, and the
     report says which predictor set the start date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Candidate formats for text date columns. Order is the tie-break when several
# formats parse equally well: day-first wins over month-first, matching the
# DD/MM/YYYY layout of the default Google Sheet.
DATE_FORMATS = (
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%b-%y",
    "%d/%m/%y", "%m/%d/%y",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
)

_NUMERIC_JUNK = re.compile(r"[,%\s]")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse a date column using one consistent format for every row.

    Plain ``pd.to_datetime`` infers the format from the first value, so a
    DD/MM/YYYY column whose first date is ambiguous (e.g. ``08/06/2006``) is
    read as MM/DD/YYYY: every day > 12 becomes NaT and the rest get day and
    month swapped. Instead, try each candidate format and keep the one that
    parses the most rows, then the one that keeps the series in order, then
    the earlier entry in ``DATE_FORMATS``.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values

    text = values.astype("string").str.strip()
    present = text.notna() & (text != "")
    n_present = int(present.sum())
    if n_present == 0:
        return pd.to_datetime(values, errors="coerce")

    best, best_key = None, None
    for rank, fmt in enumerate(DATE_FORMATS):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        hits = int(parsed.notna().sum())
        if hits == 0:
            continue
        inversions = int((parsed.dropna().diff() < pd.Timedelta(0)).sum())
        key = (hits, -inversions, -rank)
        if best_key is None or key > best_key:
            best, best_key = parsed, key

    if best is None or best_key[0] < n_present:
        # Leftovers in an unexpected format: parse them one by one, day-first.
        fallback = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
        best = fallback if best is None else best.fillna(fallback)
    return best


def to_numeric_series(values: pd.Series) -> pd.Series:
    """Coerce to float, accepting ``6.48%`` (→ 6.48) and ``24,536.60`` (→ 24536.6)."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    text = values.astype("string").str.replace(_NUMERIC_JUNK, "", regex=True)
    return pd.to_numeric(text, errors="coerce").astype(float)


def coerce_numeric_frame(df: pd.DataFrame, min_share: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """Convert text columns that are really numbers (``6.48%``, ``1,234.5``).

    A text column is converted when at least ``min_share`` of its non-empty
    cells parse as numbers. Returns the new frame and the converted column names.
    """
    out = df.copy()
    converted: list[str] = []
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        text = s.dropna().astype(str).str.strip()
        text = text[text != ""]
        if text.empty or not text.str.contains(r"\d").all():
            continue
        parsed = to_numeric_series(text)
        if parsed.notna().mean() >= min_share:
            out[col] = to_numeric_series(s)
            converted.append(str(col))
    return out, converted


def sheet_csv_url(sheet_url: str) -> tuple[str | None, str | None]:
    """Turn a Google Sheets link into its CSV export URL. Returns (url, error)."""
    url = (sheet_url or "").strip()
    if not url:
        return None, "No Google Sheets URL given. Set GOOGLE_SHEET_URL or paste a link."
    gid_match = re.search(r"[#&?]gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    published = re.search(r"/spreadsheets/d/e/([a-zA-Z0-9-_]+)", url)
    if published:
        return f"https://docs.google.com/spreadsheets/d/e/{published.group(1)}/pub?output=csv&gid={gid}", None
    sheet_id = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not sheet_id:
        return None, "Invalid Google Sheets URL: could not find the sheet ID."
    return f"https://docs.google.com/spreadsheets/d/{sheet_id.group(1)}/export?format=csv&gid={gid}", None


def load_sheet(sheet_url: str) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch a Google Sheet tab as a DataFrame with numeric text converted."""
    csv_url, error = sheet_csv_url(sheet_url)
    if error:
        return None, error
    try:
        df = pd.read_csv(csv_url)
    except Exception as exc:  # network, permissions, HTML login page, …
        return None, f"Could not read the sheet ({exc}). Is it shared as 'Anyone with the link'?"
    df, _ = coerce_numeric_frame(df)
    return df, None


def read_upload(name: str, buffer: Any) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file with numeric text converted."""
    df = pd.read_csv(buffer) if name.lower().endswith(".csv") else pd.read_excel(buffer)
    df, _ = coerce_numeric_frame(df)
    return df


@dataclass
class CleanReport:
    """What ``clean_data`` changed, so the app can tell the user."""

    rows_in: int = 0
    rows_out: int = 0
    unparsed_dates: int = 0
    duplicate_dates: list[str] = field(default_factory=list)
    missing_target_rows: int = 0
    leading_rows_trimmed: int = 0
    start_limited_by: str | None = None
    empty_features: list[str] = field(default_factory=list)
    start: Any = None
    end: Any = None


def clean_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str | None = None,
) -> tuple[pd.DataFrame, CleanReport]:
    """Select, order and clean the modelling columns. See the module docstring."""
    report = CleanReport(rows_in=len(df))
    features = [f for f in dict.fromkeys(features) if f != target and f in df.columns]
    if target not in df.columns:
        return pd.DataFrame(), report

    use_date = bool(date_col) and date_col in df.columns
    cols = [target] + features + ([date_col] if use_date else [])
    data = df[cols].copy()

    if use_date:
        raw = data[date_col]
        data[date_col] = parse_dates(raw)
        report.unparsed_dates = int(data[date_col].isna().sum())
        data = data[data[date_col].notna()]
        data = data.sort_values(date_col, kind="stable")
        dup = data[date_col].duplicated(keep="first")
        if dup.any():
            report.duplicate_dates = raw.loc[dup[dup].index].astype(str).unique().tolist()
        data = data[~dup]

    for col in [target] + features:
        data[col] = to_numeric_series(data[col])

    report.empty_features = [f for f in features if data[f].notna().sum() == 0]
    features = [f for f in features if f not in report.empty_features]
    data = data.drop(columns=report.empty_features)

    if features:
        data[features] = data[features].ffill()

    has_target = data[target].notna()
    report.missing_target_rows = int((~has_target).sum())
    data = data[has_target]

    if features and len(data):
        starts = {f: data[f].notna().to_numpy().argmax() if data[f].notna().any() else len(data) for f in features}
        limiter = max(starts, key=starts.get)
        if starts[limiter] > 0:
            report.leading_rows_trimmed = int(starts[limiter])
            report.start_limited_by = limiter
            data = data.iloc[starts[limiter]:]

    data = data.reset_index(drop=True)
    report.rows_out = len(data)
    if use_date and len(data):
        report.start, report.end = data[date_col].iloc[0], data[date_col].iloc[-1]
    return data, report
