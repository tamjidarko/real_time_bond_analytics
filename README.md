
# Real-Time Bond Analytics: Yield Curves, Credit Spreads, and Risk Measures

## Project Overview

This project implements **real-time bond analytics** using market data sourced from FRED (Federal Reserve Economic Data). It combines **Treasury yields**, **credit spreads**, and standard **fixed-income risk measures** (duration, convexity, DV01) to provide a holistic view of bond market conditions.

The goal is to replicate the kind of analytics performed in professional fixed-income trading, portfolio management, and risk analysis, but in a reproducible and transparent way.

---

## Objectives

* Fetch **live Treasury rates** across the maturity spectrum (1M–30Y).
* Retrieve **corporate bond spreads** across credit ratings (AAA–CCC).
* Compute **yield-to-price relationships** for Treasuries, Investment Grade, and High Yield corporates.
* Apply **duration and convexity analysis** to measure price sensitivity to interest rate shocks.
* Visualize the **yield curve**, **credit spread ladder**, and **slope spreads**.
* Provide a **market interpretation** of late-cycle signals, curve steepness, and credit conditions.

---

## Background Concepts

### Yield Curve

The **yield curve** plots Treasury yields against maturities. It reflects expectations about future interest rates, inflation, and economic conditions:

* **Normal curve:** Upward sloping, typical of economic expansion.
* **Inverted curve:** Short-term rates higher than long-term, often a recession signal.
* **Flat curve:** Narrow spreads between maturities, common late in the cycle.

### Credit Spreads

Credit spreads measure the extra yield over Treasuries that investors demand for taking credit risk. Spreads widen in stress and tighten in stable environments.

* **Investment Grade (IG):** AAA–BBB, safer issuers.
* **High Yield (HY):** BB and below, riskier issuers.

### Duration & Convexity

* **Modified Duration:** Approximate % change in price for a 1% (100bp) change in yields.
* **Convexity:** Adjusts for curvature in the price–yield relationship; improves estimates for larger shocks.
* **DV01:** Dollar Value of 1 basis point — change in bond price for a 1bp move in rates.

---

## Outputs

### Treasury Yield Curve

![Yield Curve](Figure_final.png)

* Short end (1M–6M): elevated rates (\~4.0–4.4%).
* 2Y at 3.62%, 10Y at 4.22%, 30Y at 4.88%.
* **2s10s spread:** +60bp → suggests steepening relative to historical inversions.
* **2s30s spread:** +126bp → confirms a steep long end.

### Credit Spread Ladder

* **AAA:** 30bp
* **AA:** 44bp
* **A:** 65bp
* **BBB:** 99bp
* **BB:** 172bp
* **B:** 281bp
* **CCC:** 783bp

Ratio of HY to IG spreads = **3.5x**, consistent with stable but cautious credit markets.

### Risk Measures (10Y Treasury Example)

* Yield = 4.22%, Price = 98.22
* Modified Duration = 8.15
* Convexity = 78.5
* DV01 (per \$1MM) = \$800

### Scenario Analysis (+100bp Rate Shock)

| Bond Type         | Duration Est. | Convexity Adj. | Actual | Error |
| ----------------- | ------------- | -------------- | ------ | ----- |
| 10Y Treasury      | -8.15%        | +0.39%         | -7.77% | 0.01% |
| A-Rated Corporate | -7.81%        | +0.37%         | -7.45% | 0.01% |
| BB HY Bond        | -5.53%        | +0.19%         | -5.35% | 0.00% |

**Result:** Model estimates align almost perfectly with repricing, confirming correct implementation of duration + convexity formulas.

---

##  Market Interpretation (as of August 31, 2025)

* **Yield Curve:** Steepening after prolonged inversion → signals potential transition from restrictive Fed policy toward normalization.
* **Credit Market:** IG and HY spreads within normal ranges, suggesting **no acute credit stress**.
* **Implication:** Market pricing reflects late-cycle but stable conditions. Neutral-to-slightly-long duration positioning in **high-quality credits** is favored.

---

## Possible Sources of Error

1. **Treasury curve anomalies**

   * Real-world curves (mid-2025) are often still inverted; FRED data may lag or interpolate differently. Verification with multiple data sources (Treasury.gov, Bloomberg) would be prudent.

2. **Corporate bond assumptions**

   * The pricing engine may assume fixed coupons leading to above-par valuations for high-yield bonds. In reality, HY bonds often price closer to or below par.

3. **Day count & compounding conventions**

   * Slight mismatches may occur if FRED conventions differ from the analytics assumptions.

---

## Room for Improvements

* **Data validation:** Cross-check Treasury yields with Treasury.gov or Bloomberg feeds.
* **Dynamic coupon modeling:** Use actual bond characteristics instead of stylized coupons.
* **Stress testing:** Add -100bp, +200bp, and multi-factor shocks (credit widening + rate moves).
* **Term premium decomposition:** Separate expectations vs. risk premia in the curve.
* **Portfolio analytics:** Extend framework to multi-bond portfolios with weighted duration and convexity.
* **Visualization upgrades:** Add interactive dashboards (Plotly/Dash) for real-time monitoring.

---

## Repository Structure

```
.
├── corrected_bond_calculator.py       
├── duration_and_convexity_calculator_real_data.py   
├── duration_and_convexity_calculator.py   # Simulated data
├── visualizations.py      # Yield curve, spreads, dashboard plots
├── Figure_final.png       # Sample yield curve & spread visualization
└── README.md              # Project documentation
```

---

**This project demonstrates how real-time bond analytics can be replicated programmatically with accurate risk measures, market interpretation, and professional-grade outputs.**


