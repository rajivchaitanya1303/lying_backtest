"""
Statistical test of alignment between model failure windows and known
market crisis periods.

Uses Fisher's exact test on the 2x2 contingency table:

                       crisis-adjacent   non-crisis
    red backtest              a             b
    green/yellow              c             d

Under the null, red-window placement is uniform across the calendar. A
significant one-sided p-value indicates that failure windows cluster in
crisis periods rather than being evenly distributed.

The default crisis-year list covers documented equity-market stress
episodes intersecting the sample:
    - 1997 (Asian financial crisis)
    - 2007, 2008, 2009 (Global Financial Crisis)
    - 2020 (COVID crash)

A test window is treated as "crisis-adjacent" if its start-year OR
start-year + 1 is in the crisis-year set (so a window starting in late
2019 and running through 2020 counts).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


DEFAULT_CRISIS_YEARS = frozenset({1997, 2007, 2008, 2009, 2020})


@dataclass
class CrisisAlignmentResult:
    ticker: str
    model: str
    n_windows: int
    a_red_crisis: int
    b_red_noncrisis: int
    c_nored_crisis: int
    d_nored_noncrisis: int
    odds_ratio: float
    fisher_p_one_sided: float
    fisher_p_two_sided: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _tag_crisis(df: pd.DataFrame, crisis_years: frozenset) -> pd.Series:
    y = pd.to_datetime(df["test_start"]).dt.year
    return y.isin(crisis_years) | (y + 1).isin(crisis_years)


def fisher_alignment(basel_windows: pd.DataFrame,
                     raw_forecasts: pd.DataFrame,
                     crisis_years: frozenset = DEFAULT_CRISIS_YEARS,
                     ) -> pd.DataFrame:
    """
    For every (ticker, model), run a Fisher exact test on the
    (red, crisis-adjacent) contingency table. Returns one row per
    (ticker, model).
    """
    dates = (raw_forecasts.groupby(["ticker", "model", "window_id"])["date"]
                          .agg(["min", "max"])
                          .reset_index()
                          .rename(columns={"min": "test_start",
                                           "max": "test_end"}))
    merged = basel_windows.merge(dates, on=["ticker", "model", "window_id"])
    merged["is_crisis"] = _tag_crisis(merged, crisis_years)
    merged["is_red"] = merged["zone"].eq("red")

    rows = []
    for (tkr, mdl), grp in merged.groupby(["ticker", "model"]):
        a = int(((grp.is_red) & (grp.is_crisis)).sum())
        b = int(((grp.is_red) & (~grp.is_crisis)).sum())
        c = int(((~grp.is_red) & (grp.is_crisis)).sum())
        d = int(((~grp.is_red) & (~grp.is_crisis)).sum())
        # scipy odds ratio is (a*d) / (b*c); may be inf if b=0
        if b == 0 or c == 0:
            odds = float("inf") if a > 0 and d > 0 else float("nan")
        else:
            odds = (a * d) / (b * c)
        try:
            _, p_one = fisher_exact([[a, b], [c, d]], alternative="greater")
            _, p_two = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        except ValueError:
            p_one = p_two = float("nan")
        rows.append(CrisisAlignmentResult(
            ticker=tkr, model=mdl, n_windows=len(grp),
            a_red_crisis=a, b_red_noncrisis=b,
            c_nored_crisis=c, d_nored_noncrisis=d,
            odds_ratio=float(odds),
            fisher_p_one_sided=float(p_one),
            fisher_p_two_sided=float(p_two),
        ).as_dict())
    return pd.DataFrame(rows)
