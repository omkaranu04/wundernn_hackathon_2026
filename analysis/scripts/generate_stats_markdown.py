"""Converts results/feature_statistics.csv into a readable Markdown file
(results/feature_statistics.md): a plain-language glossary of what each
statistic means, followed by grouped tables (matching the same feature
groups used elsewhere in this analysis) instead of one giant 114-row table.

The CSV is kept as-is alongside this - compute_statistics.py reads it back
to seed histogram ranges on a rerun (see its docstring), and it's the
easier format for anything programmatic. This Markdown file is for reading.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from data import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GROUPS = {
    "i0 — bid price": [f"i0_p{i}" for i in range(0, 11)],
    "i0 — ask price": [f"i0_p{i}" for i in range(11, 22)],
    "i0 — bid volume": [f"i0_v{i}" for i in range(0, 11)],
    "i0 — ask volume": [f"i0_v{i}" for i in range(11, 22)],
    "i0 — trade price": [f"i0_dp{i}" for i in range(4)],
    "i0 — trade volume": [f"i0_dv{i}" for i in range(4)],
    "i1 — bid price": [f"i1_p{i}" for i in range(0, 11)],
    "i1 — ask price": [f"i1_p{i}" for i in range(11, 22)],
    "i1 — bid volume": [f"i1_v{i}" for i in range(0, 11)],
    "i1 — ask volume": [f"i1_v{i}" for i in range(11, 22)],
    "i1 — trade price": [f"i1_dp{i}" for i in range(4)],
    "i1 — trade volume": [f"i1_dv{i}" for i in range(4)],
    "additional": [f"a{i}" for i in range(8)],
    "targets": list(TARGET_COLUMNS),
}

GLOSSARY = """
## What each statistic means

Plain-language explanation of every column in the table below, in the order
they appear.

- **count** — how many rows this statistic was computed over. Same for every
  column here: all 212,140,000 rows of `train.parquet` (the complete
  dataset, not a sample — see the methodology note above).
- **mean** — the average value. The "typical center" of the distribution,
  but easily skewed by a few extreme values (see *skew* below).
- **std** (standard deviation) — how spread out the values typically are
  around the mean. A small std means most values sit close together; a
  large std means they're scattered widely. Roughly: for a normal-shaped
  distribution, ~68% of values fall within 1 std of the mean.
- **min / max** — the single smallest and largest value seen anywhere in
  the dataset. Sensitive to rare outliers by definition.
- **p1, p5, p25, p50, p75, p95, p99** (percentiles) — the value below which
  that percentage of rows fall. E.g. *p25* is the value where 25% of rows
  are lower and 75% are higher. *p50* is the **median** (the true "middle"
  value — more robust to outliers than the mean). Comparing p1 vs p99 (or
  p5 vs p95) shows how wide the "typical" range is without being thrown off
  by extreme outliers the way min/max are.
- **skew** — measures *asymmetry*. Zero means the distribution is symmetric
  around its center. Positive skew means a longer tail stretching toward
  large positive values (a few very high outliers pull the mean up);
  negative skew is the mirror image (a long tail toward very negative
  values).
- **excess_kurtosis** — measures how "fat-tailed" a distribution is
  compared to a normal (bell-curve) distribution, for which excess kurtosis
  is defined as 0. A large positive value means extreme outliers are far
  more common than a normal distribution would predict — a few rows very
  far from the center. This dataset's targets have very high excess
  kurtosis (see the main README) — a concrete warning sign for outliers.
- **frac_zero / frac_negative / frac_positive** — the fraction (0.0-1.0) of
  rows where this column's value is exactly zero, strictly negative, or
  strictly positive. These three always add up to 1.0. Useful for spotting
  things percentiles alone can hide, like a big pile of exact zeros (see
  the targets' ~22% zero-inflation finding).
- **hist_underflow / hist_overflow** — bookkeeping from how the histograms
  in this analysis were built (a fixed value range was assumed before
  scanning the data; see the main README's methodology section). These
  count how many values fell outside that assumed range — i.e. were more
  extreme than expected. A **nonzero** count here is not an error, just a
  signal that a handful of values are especially extreme outliers, and
  that this column's plotted histogram may not visually show the very
  furthest tail (though the exact min/max/mean/std above are unaffected -
  those come from a separate exact calculation, not the histogram).
"""


def fmt(value: str, decimals: int = 4, is_count: bool = False) -> str:
    try:
        f = float(value)
    except ValueError:
        return value
    if is_count:
        return f"{int(round(f)):,}"
    if abs(f) >= 1000 or (abs(f) < 1e-4 and f != 0):
        return f"{f:.3e}"
    return f"{f:.{decimals}f}"


def main():
    with open(RESULTS_DIR / "feature_statistics.csv") as f:
        rows = {r["column"]: r for r in csv.DictReader(f)}

    display_cols = ["mean", "std", "min", "max", "p1", "p25", "p50", "p75", "p99",
                     "skew", "excess_kurtosis", "frac_zero", "frac_negative", "frac_positive",
                     "hist_underflow", "hist_overflow"]

    lines = []
    lines.append("# Feature & Target Statistics (readable format)\n")
    lines.append(
        "Human-readable version of `feature_statistics.csv` — exact statistics "
        "computed over the complete `train.parquet` (212,140,000 rows, no "
        "sampling; see `../README.md` for methodology). The CSV stays "
        "alongside this file for programmatic use (e.g. `compute_statistics.py` "
        "reads it back to seed histogram ranges on a rerun); this file is for "
        "reading.\n"
    )
    lines.append(GLOSSARY)
    lines.append("\n## Per-column statistics, grouped\n")

    for group_name, columns in GROUPS.items():
        lines.append(f"\n### {group_name}\n")
        header = "| column | " + " | ".join(display_cols) + " |"
        sep = "|---|" + "---:|" * len(display_cols)
        lines.append(header)
        lines.append(sep)
        count_cols = {"hist_underflow", "hist_overflow"}
        for col in columns:
            r = rows[col]
            values = [fmt(r[c], is_count=(c in count_cols)) for c in display_cols]
            lines.append(f"| `{col}` | " + " | ".join(values) + " |")

    out_path = RESULTS_DIR / "feature_statistics.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
