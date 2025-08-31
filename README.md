
# Bond Analytics with Real-Time Market Data

This repository implements an institutional-style fixed income analytics system using **live market data**. It combines:

* U.S. Treasury yield curve (risk-free baseline)
* Credit spreads across ratings
* Volatility measures (realized and implied)
* Bond ETF proxies for major sectors
* Example bond pricing and analytics

The project aims to bridge **bond market theory** (duration, convexity, spreads) with **practical tools** (real-time pricing, scenario analysis, interactive exploration).

---

## Concepts Explored

### 1. Yield Curve (Risk-Free Rates)

The **yield curve** represents the interest rates at which the U.S. Treasury borrows across maturities.

* Short-term yields reflect monetary policy expectations.
* Long-term yields reflect growth and inflation expectations.
* Corporate bonds are priced as **Treasury yield + credit spread**.

### 2. Credit Spreads

Credit spread = Extra yield investors demand over Treasuries to compensate for default risk.

* **Investment Grade (AAA–BBB):** Safer, lower spreads.
* **High Yield (BB and below):** Riskier, higher spreads.
* Example:

  ```
  Corporate YTM = Treasury Yield + Spread
  ```

### 3. Duration and Convexity

* **Macaulay Duration**: Weighted average time to cash flows.
* **Modified Duration (Dmod):** Price sensitivity to yield changes:

  ```
  ΔP / P ≈ -Dmod × Δy
  ```
* **Convexity (C):** Adjusts for curvature in the price-yield relationship:

  ```
  ΔP / P ≈ -Dmod × Δy + 0.5 × C × (Δy)²
  ```
* High convexity → more favorable asymmetry: bond gains more when yields fall than it loses when yields rise.

### 4. DV01 (Dollar Value of a Basis Point)

Measures how much the bond’s price changes if yields move by **1 basis point (0.01%)**.

```
DV01 = (Duration × Price × 0.0001)
```

* Example: A 10-year bond with duration 7.8 and price \$100 → DV01 ≈ \$0.78 per \$100 notional.

### 5. Volatility

* **MOVE Index** = Implied volatility in Treasury options (bond market’s VIX).
* **Realized volatility** = Standard deviation of recent yield changes.
* Important for **VaR (Value-at-Risk)** and risk management.

### 6. Bond Pricing Simplification in this Project

Currently assumes **par bonds**:

* Coupon = Yield to Maturity (YTM).
* Price ≈ 100 at issuance.
* YTM determined as:

  ```
  YTM = Treasury Yield (at maturity) + Credit Spread
  ```

---

## Methodology

1. **Data Fetching**

   * Treasury yields → FRED API.
   * Credit spreads → FRED investment-grade & high-yield indices.
   * Volatility → MOVE index (FRED) + realized yield changes.
   * Bond ETF proxies (LQD, HYG, TLT, IEF, SHY, AGG) → Yahoo Finance.

2. **Analytics Generation**

   * Risk-free 10Y rate, IG and HY spreads, daily vol summary.
   * Bond examples: AAA, A, BB ratings at 5Y, 7Y, 10Y maturities.
   * YTM derived by adding spreads to Treasuries.

3. **Interactive Analysis**

   * Build custom bonds (choose rating and maturity).
   * Compare bonds across ratings.
   * Explore curve trades (steepener/flatteners).
   * Refresh live data.

---

## Worked Example

Suppose the 10-year Treasury yield = **4.22%**.

* **A-rated corporate bond** with 10Y maturity and spread of 80 bp:

  ```
  YTM = 4.22% + 0.80% = 5.02%
  Duration ≈ 8.0
  DV01 = 8.0 × 100 × 0.0001 = $0.80
  ```
* **Zero-coupon Treasury strip** with 20Y maturity:

  ```
  Duration ≈ 20 years
  Convexity very high
  DV01 ≈ 19.6 × 45.29 × 0.0001 = $0.89 (per $100 notional)
  ```

This illustrates why zero-coupon bonds are extremely rate-sensitive compared to coupon bonds.

---

## What Works Well

* **Real-time integration**: Treasury curve and spreads pulled directly from FRED.
* **Conceptual clarity**: Shows how bonds are built from risk-free + spread.
* **Educational framework**: Connects YTM, duration, convexity, and spreads.
* **Interactive exploration**: User can define bonds and analyze on the fly.

---

## Shortcomings and Limitations

* **Credit Spreads Understated**: FRED data fetch sometimes returns unrealistically tight spreads (1–3 bp). True values are often 100–600 bp depending on rating.
* **ETF Data Scaling**: Yahoo Finance pulls are mis-scaled (percentages > 100%). Should be adjusted to reflect yield or NAV.
* **MOVE Index Reliability**: Data fetch may fail; realized vol is understated due to sampling.
* **Bond Pricing Simplified**: Assumes par bonds. No present value discounting of cash flows yet.

---

## Next Steps

* Implement **full bond pricing engine**:

  * Present value of coupon and principal cash flows.
  * Macaulay & modified duration.
  * Convexity and DV01.
* Fix data parsing for spreads and ETF yields.
* Add scenario analysis:

  * Parallel rate shifts.
  * Curve steepening/flattening.
  * Spread widening.
* Expand volatility metrics with robust MOVE integration.

---

## Usage

Clone the repo and run:

```bash
python bond_analytics.py
```

The program will:

1. Fetch live Treasury yields, spreads, volatility, ETF proxies.
2. Print a market summary.
3. Generate example bond analytics.
4. Enter interactive mode for custom exploration.

Example:

```
Select a calculation option:
1. Custom bond with live yields
2. Compare bonds across ratings
3. Analyze curve trades
4. Update market data

Enter choice (1-4): 1
Enter rating (AAA/AA/A/BBB/BB/B) [A]: BBB
Enter maturity in years [10]: 10
```

Output:

```
BOND ANALYSIS WITH LIVE MARKET DATA
Rating: BBB
Maturity: 10 years
YTM (from market): 5.50%
Coupon: 5.50%
Risk-Free Rate: 4.22%
Credit Spread: 128bp
```

---

## Educational Value

This repository doubles as:

* A **learning tool** for students or professionals entering fixed income.
* A **framework** for extending into risk analytics (duration, convexity, DV01, VaR).
* A **prototype** for institutional-style dashboards using public data.
