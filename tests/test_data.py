import numpy as np
import pandas as pd
import pytest

from aarambh.data import clean_data, coerce_numeric_frame, parse_dates, sheet_csv_url, to_numeric_series


def _check_dates(values, expected):
    got = parse_dates(pd.Series(values, dtype=object)).reset_index(drop=True)
    exp = pd.to_datetime(pd.Series(expected))
    assert got.equals(exp.astype(got.dtype)), got.tolist()


class TestParseDates:
    def test_day_first_sheet_layout(self):
        # The source sheet's first date is ambiguous; plain pd.to_datetime reads it month-first.
        _check_dates(["08/06/2006", "09/06/2006", "13/06/2006"], ["2006-06-08", "2006-06-09", "2006-06-13"])

    def test_genuine_month_first(self):
        _check_dates(["01/05/2024", "01/31/2024"], ["2024-01-05", "2024-01-31"])

    def test_ambiguous_prefers_day_first(self):
        _check_dates(["05/01/2024", "06/01/2024"], ["2024-01-05", "2024-01-06"])

    def test_iso_missing_and_unpadded(self):
        _check_dates(["2024-01-05", None, ""], ["2024-01-05", None, None])
        _check_dates(["8/6/2006", "13/6/2006"], ["2006-06-08", "2006-06-13"])

    def test_datetime_column_untouched(self):
        s = pd.Series(pd.to_datetime(["2024-01-05", "2024-02-05"]))
        assert parse_dates(s) is s

    def test_mixed_objects(self):
        _check_dates([pd.Timestamp("2024-03-13"), "14/03/2024"], ["2024-03-13", "2024-03-14"])


class TestNumeric:
    def test_percent_and_thousands(self):
        out = to_numeric_series(pd.Series(["6.48%", "24,536.60", " 7 ", "n/a"]))
        assert out.iloc[:3].tolist() == [6.48, 24536.6, 7.0] and np.isnan(out.iloc[3])

    def test_coerce_frame_converts_only_numeric_text(self):
        df = pd.DataFrame({"EY": ["6.48%", "6.16%"], "DATE": ["08/06/2006", "09/06/2006"],
                           "NOTE": ["VARIABLES", "24.5"], "X": [1.0, 2.0]})
        out, converted = coerce_numeric_frame(df)
        assert converted == ["EY"]
        assert out["EY"].tolist() == [6.48, 6.16]
        assert out["DATE"].tolist() == df["DATE"].tolist()
        assert out["NOTE"].tolist() == df["NOTE"].tolist()


class TestCleanData:
    def _frame(self):
        return pd.DataFrame({
            "DATE": ["03/01/2020", "01/01/2020", "02/01/2020", "04/01/2020", "05/01/2020", "02/01/2020"],
            "Y": [3.0, 1.0, 2.0, np.nan, 5.0, 99.0],
            "A": [30.0, 10.0, 20.0, 40.0, np.nan, 999.0],
            "LATE": [np.nan, np.nan, 7.0, 8.0, 9.0, 0.0],
        })

    def test_sorted_deduplicated_no_backfill(self):
        data, rep = clean_data(self._frame(), "Y", ["A", "LATE"], "DATE")
        # Sorted, the repeated 02/01 keeps its first row (Y=2), Y-less 04/01 dropped,
        # LATE starts 02/01 so 01/01 is trimmed rather than back-filled.
        assert data["DATE"].dt.day.tolist() == [2, 3, 5]
        assert data["Y"].tolist() == [2.0, 3.0, 5.0]
        assert data["A"].tolist() == [20.0, 30.0, 40.0]       # 05/01 forward-filled from 04/01
        assert data["LATE"].tolist() == [7.0, 7.0, 9.0]       # 03/01 forward-filled from 02/01
        assert rep.duplicate_dates == ["02/01/2020"]
        assert rep.missing_target_rows == 1
        assert rep.leading_rows_trimmed == 1 and rep.start_limited_by == "LATE"
        assert rep.rows_out == 3

    def test_target_excluded_from_features_and_empty_features_reported(self):
        df = self._frame().assign(EMPTY=np.nan)
        data, rep = clean_data(df, "Y", ["Y", "A", "EMPTY"], "DATE")
        assert "EMPTY" not in data.columns and rep.empty_features == ["EMPTY"]
        assert list(data.columns) == ["Y", "A", "DATE"]

    def test_no_date_column_keeps_file_order(self):
        data, _ = clean_data(pd.DataFrame({"Y": [3.0, 1.0, 2.0], "A": [1.0, 2.0, 3.0]}), "Y", ["A"], None)
        assert data["Y"].tolist() == [3.0, 1.0, 2.0]


@pytest.mark.parametrize("url,expected", [
    ("https://docs.google.com/spreadsheets/d/AbC-1_2/edit?pli=1&gid=990005192#gid=990005192",
     "https://docs.google.com/spreadsheets/d/AbC-1_2/export?format=csv&gid=990005192"),
    ("https://docs.google.com/spreadsheets/d/AbC/edit",
     "https://docs.google.com/spreadsheets/d/AbC/export?format=csv&gid=0"),
    ("https://docs.google.com/spreadsheets/d/e/2PACX-xyz/pubhtml?gid=5",
     "https://docs.google.com/spreadsheets/d/e/2PACX-xyz/pub?output=csv&gid=5"),
])
def test_sheet_csv_url(url, expected):
    assert sheet_csv_url(url) == (expected, None)


def test_sheet_csv_url_errors():
    assert sheet_csv_url("")[1] is not None
    assert sheet_csv_url("https://example.com/not-a-sheet")[1] is not None
