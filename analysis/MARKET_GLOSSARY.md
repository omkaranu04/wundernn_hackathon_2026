# Market Domain Glossary

What the feature *types* (`p`, `v`, `dp`, `dv`, bid/ask) mean in general
financial-market terms. The competition deliberately anonymizes the exact
values (see `PROBLEM_STATEMENT.md` / the FAQ: *"to focus the competition on
the modeling task itself, rather than domain-specific feature engineering"*)
— we don't and can't know the precise definition or scaling of any single
column. What follows is the **general domain knowledge** behind the *kind*
of data this is, which is still useful context for reasoning about the
data's shape, sparsity, and modeling implications.

## The limit order book (the "L11" in the docs)

Most tradeable markets (stocks, futures, crypto, FX) run on a **limit order
book**: a live, continuously-updated list of everyone's outstanding buy and
sell orders for one instrument, organized by price.

- **Bid** = a standing offer to *buy* at a given price. The **best bid** is
  the highest price any buyer is currently willing to pay.
- **Ask** (also called the **offer**) = a standing offer to *sell* at a
  given price. The **best ask** is the lowest price any seller is currently
  willing to accept.
- The best ask is always ≥ the best bid in a functioning market. The gap
  between them is the **spread** — a basic measure of how "tight" or
  liquid the market is right now.
- Orders don't just sit at the best price — they stack up at many price
  levels behind the best one, each level holding some quantity. This is
  the book's **depth**.
- **"L11"** means this dataset gives the **top 11 price levels on each
  side** (11 bid levels + 11 ask levels), not just the single best
  bid/ask. This is a fairly rich view into market depth, not just "the
  current price."

So for each instrument, at each moment, the book looks conceptually like:

```
 ASK side (sellers)          price          BID side (buyers)
        …                                          …
   [level 10 ask]  ←  furthest from best price  →   [level 10 bid]
        …                                          …
   [level 1 ask]   ←     closer to best price   →   [level 1 bid]
   [best ask]      ←        tightest spread     →   [best bid]
```

## What the feature groups likely represent

- **`p` (price-like), 22 columns per instrument** — the price *at* each of
  the 11 bid levels and 11 ask levels (indices 0-10 = bid side, 11-21 =
  ask side, per the docs' column layout). In real order-book datasets
  these are almost never raw absolute prices (which drift arbitrarily over
  time and aren't comparable across different moments or instruments) —
  they're typically expressed as small **offsets from a reference point**
  (e.g. the mid-price, or the best bid/ask). That matches what we found in
  the actual data: these columns have very low variance (~0.15, the
  tightest of any feature group — see `results/feature_statistics.md`) and
  a heavily quantized, multi-modal shape rather than a smooth continuous
  distribution.

- **`v` (volume-like), 22 columns per instrument** — the *quantity*
  resting at each of those same 22 price levels (how many units/shares/
  contracts are waiting to trade there). This is a direct measure of
  **liquidity** or **market depth** at that level: more volume at a level
  means it would take more buying/selling pressure to push the price
  through it. Consistent with volumes generally being non-negative
  quantities, we found these columns skew strongly positive (frac_positive
  ≈ 0.98 for most).

- **`dp` / `dv` (trade price / trade volume), 4 columns each per
  instrument** — these are structurally different from the book columns
  above. The book (`p`/`v`) describes *standing, not-yet-executed* orders
  — intent, which can be cancelled any time. `dp`/`dv` instead describe
  **actual executed trades** — a transaction only happens when a buyer's
  bid and a seller's ask actually match, at which point real money and
  real shares change hands. `dp` is the price a recent trade executed at;
  `dv` is how much was traded in it. Having 4 slots of each suggests a
  short rolling history of the last few trades, not book levels — trades
  are discrete, comparatively rare events (most single-row snapshots see
  *no* new trade), which matches what we found: the `dv` columns are
  dominated by a huge spike at (or near) zero, with a sparse spread of
  nonzero values around it — i.e., "no new trade this row" is the common
  case.

- **Why this combination (book + trades) is used for short-horizon
  prediction** — this is the foundation of what's called **market
  microstructure** analysis. The book's *shape* (how volume is stacked on
  each side) reflects current supply/demand imbalance and can hint at
  near-term price pressure; trade flow (the sequence of actual executions)
  is "harder" information than the book, since it reflects money that's
  already changed hands rather than orders that could be cancelled at any
  moment. Combining both is standard practice for predicting price moves
  over very short horizons (this competition's sequences suggest
  sub-second to low-second granularity, given 20,000 steps per sequence)
  — a fundamentally different problem from longer-horizon
  fundamental/macro forecasting.

## Why two instruments (`i0`, `i1`)?

Only `i0` has prediction targets, but `i1`'s full book + trade data is
given as additional context. A very common setup in real trading is two
**related** instruments — e.g. the same underlying asset traded on two
venues, a derivative and its underlying, or two historically correlated
assets — where price discovery in one can lead or lag the other by a
short window (a **lead-lag** relationship). If that's the case here,
`i1`'s book/trade activity could carry predictive signal about `i0`'s
near-future movement even without `i1` having its own labeled target. Our
correlation analysis (`README.md`, finding 7) found `i1`'s features
correlate much more weakly with the targets than `i0`'s own features do
under a simple *linear* lens — consistent with `i1` contributing mostly
indirect/contextual or nonlinear signal rather than a strong direct
linear predictor, if it contributes meaningfully at all this way.

## The `additional` features (`a0`-`a7`)

Genuinely unknown — the docs don't hint at what these are, and unlike
`p`/`v`/`dp`/`dv` there's no naming convention to lean on. That said, the
*shape* we found for `a5`/`a6`/`a7` (a large point-mass at a fixed extreme
value, plus a separate, genuinely different-looking distribution for the
rest — see finding 5 in `README.md`) is a classic pattern for a
**sentinel-encoded "not currently available" flag** on top of a real
continuous quantity — common for things like "time since the last X
happened" (which needs some placeholder value when no such event has
occurred yet, or fell outside a lookback window). This is a plausible
*shape-based* guess, not a claim about what `a5`/`a6`/`a7` actually
measure.

## Practical takeaway for modeling

None of this changes what's already documented in `README.md`, but it
gives a reason *why* the data looks the way it does:

- Low-variance, quantized `p` columns ≈ small relative offsets, not raw
  prices — consistent with why they're the columns with the tightest,
  most structured distributions.
- Zero-heavy `dv` columns ≈ "no trade this row" being the normal case,
  not missing data.
- Weak `i1` correlations ≈ expected if `i1`'s value is more about
  cross-instrument context/timing than a direct linear predictor of
  `i0`'s own targets.
