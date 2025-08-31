
# Bond Analytics with Real-Time Market Data

This project provides an institutional-style fixed income analytics framework that integrates **real-time Treasury yield curves, credit spreads, volatility measures, and bond ETF data** to generate professional-grade bond analytics. It is designed as both a teaching tool and a practical dashboard for bond risk and return analysis.

---

## Concepts Covered

### Yield Curve

* The U.S. Treasury yield curve is the foundation of fixed income pricing.
* Each maturity (from 1 month to 30 years) represents the "risk-free" rate at which the U.S. government can borrow.
* Corporate bonds are priced as a spread above these yields, reflecting credit risk.

### Credit Spreads

* The yield difference between corporate bonds and Treasuries of the same maturity.
* Higher-rated bonds (AAA, AA) trade with lower spreads.
* Lower-rated (high yield or "junk") bonds trade with higher spreads to compensate investors for credit risk.

### Duration and Convexity (conceptual integration)

* **Duration** measures sensitivity of bond prices to interest rate changes (linear approximation).
* **Convexity** adjusts for the curvature in the price–yield relationship, capturing how sensitivity changes at different yield levels.
* High-duration bonds (e.g., zero-coupon Treasuries) are very sensitive to interest rate changes, while high-convexity bonds gain more on rallies than they lose on sell-offs.

### Volatility

* Measures the variability of bond yields.
* The MOVE Index (bond market equivalent of the VIX) provides implied volatility.
* Realized volatility can also be computed from daily changes in yields.

### Bond Pricing with Live Data

* A bond’s **Yield to Maturity (YTM)** is estimated as:

  ```
  YTM ≈ Risk-Free Treasury Yield + Credit Spread
  ```
* In this implementation, bonds are assumed to price at par (coupon = YTM).
* The interactive module allows exploration of bonds by rating, maturity, and yield curve conditions.

---

## Methodology

1. **Market Data Integration**

   * Treasury yields are fetched from the Federal Reserve Economic Data (FRED).
   * Credit spreads and ratings-based spreads are pulled from FRED series.
   * Volatility measures (realized and MOVE index) provide insight into market risk.
   * Bond ETF data (e.g., LQD, HYG, TLT) from Yahoo Finance are used as proxies for corporate and Treasury sectors.

2. **Analytics Generation**

   * Summarizes risk-free rate, spreads, and volatility.
   * Prices sample bonds of different ratings and maturities using current market conditions.
   * Provides an interactive mode for user-defined bonds and curve trades.

3. **Scenario Exploration**

   * Compares investment grade vs. high yield vs. Treasuries.
   * Demonstrates how spreads and maturities affect yields.
   * Highlights the importance of duration and convexity in rate-sensitive instruments.

---

## What Works Well

* **Real-Time Market Data**: Treasury yields and spreads pulled live from FRED provide a realistic baseline for analysis.
* **Conceptual Clarity**: Simplifies bond pricing into its key drivers (risk-free rate + spread).
* **Interactive Mode**: Users can build custom bonds and explore market-driven yields.
* **Educational Value**: Bridges the gap between theory (duration, convexity, spreads) and practice (live pricing).

---

## Shortcomings and Limitations

* **Spread Data Quality**: FRED series often report very tight spreads (e.g., 1–3 bps), which understate true market values. Actual IG spreads are closer to 100–150 bps, HY around 400–600 bps.
* **ETF Data Scaling**: Current Yahoo Finance fetch returns incorrectly scaled values (e.g., percentages > 100%). These should be normalized to show yields or NAVs.
* **Volatility Data**: The MOVE index fetch sometimes fails, and realized volatility is understated due to limited sampling.
* **Bond Pricing Model**: Currently assumes par bonds with coupon = YTM. A more robust implementation would compute price, duration, and convexity using full discounted cash flow models.

---

## Next Steps

* Correct credit spread and ETF yield parsing for more realistic analytics.
* Implement full bond pricing engine with:

  * Macaulay and modified duration
  * Convexity
  * DV01 (dollar value of a basis point)
* Extend scenario analysis (rate shocks, spread widening, curve steepening/flattening).
* Improve volatility module to reliably fetch MOVE and compute realized vol from longer time horizons.

---

## Usage

Run the script to:

1. Fetch live market data.
2. Generate a bond market dashboard with Treasury yields, spreads, and ETFs.
3. Explore bond analytics via interactive mode.

Example interactive input:

```
Select a calculation option:
1. Custom bond with live yields
2. Compare bonds across ratings
3. Analyze curve trades
4. Update market data
```

