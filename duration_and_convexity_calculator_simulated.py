"""
Professional Duration and Convexity Calculator
Implements market conventions and Bloomberg-style analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class DayCountConvention(Enum):
    """Market standard day count conventions"""
    ACTUAL_ACTUAL = "Actual/Actual"
    THIRTY_360 = "30/360"
    ACTUAL_360 = "Actual/360"
    ACTUAL_365 = "Actual/365"

@dataclass
class MarketData:
    """Market data container"""
    risk_free_rate: float = 0.045  # Current T-bill rate
    credit_spread: float = 0.0     # Spread over risk-free
    volatility_1d_bp: float = 12   # Daily rate volatility in basis points
    curve_slope_bp: float = 150    # 10Y-2Y spread in bp
    
    @property
    def total_yield(self) -> float:
        return self.risk_free_rate + self.credit_spread


@dataclass
class Bond:
    """Enhanced bond class with market conventions"""
    face_value: float = 1000.0
    coupon_rate: float = 0.05
    years_to_maturity: float = 10.0
    yield_to_maturity: float = 0.05
    frequency: int = 2  # 1=Annual, 2=Semi-annual, 4=Quarterly
    settlement_date: date = field(default_factory=date.today)
    maturity_date: date = None
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    
    def __post_init__(self):
        if self.maturity_date is None:
            # Convert to int to ensure compatibility with timedelta
            days_to_maturity = int(365 * self.years_to_maturity)
            self.maturity_date = self.settlement_date + timedelta(days=days_to_maturity)
    
    @property
    def coupon_payment(self) -> float:
        return self.face_value * self.coupon_rate / self.frequency
    
    @property
    def periods(self) -> int:
        return int(self.years_to_maturity * self.frequency)
    
    @property
    def periodic_yield(self) -> float:
        return self.yield_to_maturity / self.frequency
    
    def days_to_next_coupon(self) -> int:
        """Calculate days until next coupon payment"""
        days_between_coupons = 365 / self.frequency
        # Simplified - in practice would use actual payment schedule
        return int(days_between_coupons / 2)  # Assume middle of period
    
    def accrued_days(self) -> int:
        """Days since last coupon"""
        days_between_coupons = 365 / self.frequency
        return int(days_between_coupons / 2)


class ProfessionalBondCalculator:
    """
    Professional-grade bond calculator with market conventions
    Matches Bloomberg/Reuters terminal functionality
    """
    
    def __init__(self, bond: Bond, market_data: MarketData = None):
        self.bond = bond
        self.market = market_data or MarketData()
        self._price_cache = {}
        
    def calculate_price(self, ytm: float = None, settlement_date: date = None) -> Dict[str, float]:
        """
        Calculate bond price with clean/dirty price distinction
        
        Returns:
            Dictionary with clean_price, accrued_interest, dirty_price
        """
        if ytm is None:
            ytm = self.bond.yield_to_maturity
        
        if settlement_date is None:
            settlement_date = self.bond.settlement_date
        
        # Handle edge case for bonds at or past maturity
        if self.bond.years_to_maturity <= 0:
            return {
                'clean_price': self.bond.face_value,
                'accrued_interest': 0,
                'dirty_price': self.bond.face_value,
                'price_as_percent': 100.0
            }
        
        # PV of cash flows (clean price)
        clean_price = 0
        periodic_yield = ytm / self.bond.frequency
        
        for t in range(1, self.bond.periods + 1):
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            clean_price += cash_flow / (1 + periodic_yield) ** t
        
        # Accrued interest
        accrued_interest = self.calculate_accrued_interest()
        
        return {
            'clean_price': clean_price,
            'accrued_interest': accrued_interest,
            'dirty_price': clean_price + accrued_interest,
            'price_as_percent': (clean_price / self.bond.face_value) * 100
        }
    
    def calculate_accrued_interest(self) -> float:
        """Calculate accrued interest using day count convention"""
        if self.bond.day_count == DayCountConvention.THIRTY_360:
            # Simplified 30/360 calculation
            days_accrued = self.bond.accrued_days()
            days_in_period = 360 / self.bond.frequency
        else:  # Actual/Actual
            days_accrued = self.bond.accrued_days()
            days_in_period = 365 / self.bond.frequency
        
        accrual_fraction = days_accrued / days_in_period
        return self.bond.coupon_payment * accrual_fraction
    
    def calculate_macaulay_duration(self) -> float:
        """Calculate Macaulay Duration"""
        if self.bond.coupon_rate == 0:
            return self.bond.years_to_maturity
        
        # Handle edge case for very short or zero maturity
        if self.bond.years_to_maturity <= 0:
            return 0.0
        
        price_dict = self.calculate_price()
        clean_price = price_dict['clean_price']
        
        # Handle edge case where price is zero or very small
        if abs(clean_price) < 1e-10:
            return 0.0
        
        weighted_cash_flows = 0
        
        for t in range(1, self.bond.periods + 1):
            time_in_years = t / self.bond.frequency
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            
            pv = cash_flow / (1 + self.bond.periodic_yield) ** t
            weighted_cash_flows += time_in_years * pv
        
        return weighted_cash_flows / clean_price
    
    def calculate_modified_duration(self) -> float:
        """Calculate Modified Duration"""
        macaulay_duration = self.calculate_macaulay_duration()
        # Correct formula for discrete compounding
        return macaulay_duration / (1 + self.bond.yield_to_maturity / self.bond.frequency)
    
    def calculate_convexity(self) -> float:
        """
        Calculate convexity with correct scaling
        Formula: (1/P) * (1/(1+y)^2) * Σ[CFt * t * (t+1) / (1+y)^t]
        """
        # Handle edge cases
        if self.bond.years_to_maturity <= 0:
            return 0.0
            
        price_dict = self.calculate_price()
        clean_price = price_dict['clean_price']
        
        # Handle edge case where price is zero or very small
        if abs(clean_price) < 1e-10:
            return 0.0
        
        weighted_sum = 0
        for t in range(1, self.bond.periods + 1):
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            
            # Time in periods, not years
            pv = cash_flow / (1 + self.bond.periodic_yield) ** t
            
            # Correct formula: t(t+1) for period t
            time_factor = t * (t + 1)
            weighted_sum += time_factor * pv
        
        # Scale by frequency squared to convert to annual terms
        convexity = weighted_sum / (clean_price * (1 + self.bond.periodic_yield) ** 2 * self.bond.frequency ** 2)
        
        return convexity
    
    def calculate_dv01(self, notional: float = None) -> Dict[str, float]:
        """
        Calculate DV01 (Dollar Value of a Basis Point)
        Also known as PVBP (Price Value of Basis Point)
        """
        if notional is None:
            notional = self.bond.face_value
        
        # Method 1: Using modified duration
        modified_duration = self.calculate_modified_duration()
        price_dict = self.calculate_price()
        clean_price = price_dict['clean_price']
        
        dv01_duration = modified_duration * clean_price * 0.0001  # 1bp = 0.0001
        
        # Method 2: Actual repricing (more accurate)
        ytm_up = self.bond.yield_to_maturity + 0.0001
        ytm_down = self.bond.yield_to_maturity - 0.0001
        price_up = self.calculate_price(ytm_up)['clean_price']
        price_down = self.calculate_price(ytm_down)['clean_price']
        dv01_repricing = (price_down - price_up) / 2
        
        # Scale to notional
        scaling_factor = notional / self.bond.face_value
        
        return {
            'dv01_duration_method': dv01_duration * scaling_factor,
            'dv01_repricing_method': dv01_repricing * scaling_factor,
            'dv01_per_million': dv01_repricing * (1_000_000 / self.bond.face_value),
            'basis_point_value': dv01_repricing  # Bloomberg terminology
        }
    
    def calculate_key_rate_durations(self, key_rates: List[float] = None) -> pd.DataFrame:
        """
        Calculate key rate durations (KRD) for curve risk
        Used for hedging non-parallel curve shifts
        """
        if key_rates is None:
            key_rates = [2, 5, 10, 30]  # Standard key rate tenors
        
        krds = {}
        mod_dur = self.calculate_modified_duration()
        
        for tenor in key_rates:
            if tenor <= self.bond.years_to_maturity:
                # Simplified KRD - weight by distance to maturity
                distance = abs(self.bond.years_to_maturity - tenor)
                weight = np.exp(-distance / 5)  # Exponential decay
                krds[f'{tenor}Y'] = mod_dur * weight
            else:
                krds[f'{tenor}Y'] = 0.0
        
        # Ensure KRDs sum to total duration
        total_weight = sum(krds.values())
        if total_weight > 0:
            for key in krds:
                krds[key] = (krds[key] / total_weight) * mod_dur
        
        return pd.DataFrame(list(krds.items()), columns=['Tenor', 'KRD'])
    
    def calculate_spread_metrics(self) -> Dict[str, float]:
        """Calculate spread-related risk metrics"""
        
        # Spread duration (same as modified duration for fixed-rate bonds)
        spread_duration = self.calculate_modified_duration()
        
        # Spread DV01
        price_dict = self.calculate_price()
        clean_price = price_dict['clean_price']
        spread_dv01 = spread_duration * clean_price * 0.0001
        
        # OAS (simplified - assumes no optionality)
        oas = self.bond.yield_to_maturity - self.market.risk_free_rate
        
        return {
            'spread_duration': spread_duration,
            'spread_dv01': spread_dv01,
            'oas_bp': oas * 10000,
            'credit_spread_bp': self.market.credit_spread * 10000
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        mod_dur = self.calculate_modified_duration()
        convexity = self.calculate_convexity()
        price_dict = self.calculate_price()
        clean_price = price_dict['clean_price']
        dv01 = self.calculate_dv01()['dv01_repricing_method']
        
        # Value at Risk (parametric)
        confidence_level = 1.96  # 95% confidence
        var_1d = confidence_level * self.market.volatility_1d_bp * dv01
        var_10d = var_1d * np.sqrt(10)
        
        # Conditional VaR (Expected Shortfall)
        cvar_multiplier = 2.33  # 99% confidence
        cvar_1d = cvar_multiplier * self.market.volatility_1d_bp * dv01
        
        # Effective duration (using actual price changes)
        shift = 0.01  # 100bp
        price_up = self.calculate_price(self.bond.yield_to_maturity + shift)['clean_price']
        price_down = self.calculate_price(self.bond.yield_to_maturity - shift)['clean_price']
        effective_duration = (price_down - price_up) / (2 * clean_price * shift)
        
        # Effective convexity
        effective_convexity = (price_up + price_down - 2 * clean_price) / (clean_price * shift ** 2)
        
        return {
            'modified_duration': mod_dur,
            'effective_duration': effective_duration,
            'convexity': convexity,
            'effective_convexity': effective_convexity,
            'dv01': dv01,
            'var_95_1d': var_1d,
            'var_95_10d': var_10d,
            'cvar_99_1d': cvar_1d,
            'duration_times_spread': mod_dur * self.bond.yield_to_maturity,
            'yield_per_unit_duration': self.bond.yield_to_maturity / mod_dur if mod_dur > 0 else 0
        }
    
    def scenario_analysis(self, scenarios: Dict[str, float] = None) -> pd.DataFrame:
        """
        Run comprehensive scenario analysis
        
        Args:
            scenarios: Dictionary of scenario names and yield changes in bp
        """
        if scenarios is None:
            scenarios = {
                'Aggressive Fed Hike': 75,
                'Fed Hike': 25,
                'Unchanged': 0,
                'Fed Cut': -25,
                'Emergency Cut': -75,
                'Crisis Response': -150
            }
        
        results = []
        current_price = self.calculate_price()['clean_price']
        mod_dur = self.calculate_modified_duration()
        convexity = self.calculate_convexity()
        
        for scenario_name, bp_change in scenarios.items():
            yield_change = bp_change / 10000
            new_ytm = self.bond.yield_to_maturity + yield_change
            new_price = self.calculate_price(new_ytm)['clean_price']
            
            # Actual change
            actual_change = new_price - current_price
            actual_change_pct = (actual_change / current_price) * 100
            
            # Duration approximation
            duration_estimate = -mod_dur * yield_change * 100
            
            # Duration + Convexity approximation
            full_estimate = duration_estimate + 0.5 * convexity * (yield_change ** 2) * 100
            
            # Total return including carry (30-day)
            carry_30d = (self.bond.coupon_rate / 12) * 100  # Monthly carry
            total_return_30d = actual_change_pct + carry_30d
            
            results.append({
                'Scenario': scenario_name,
                'Yield Δ (bp)': bp_change,
                'New Price': f'${new_price:.2f}',
                'Price Δ%': f'{actual_change_pct:.2f}%',
                'Duration Est%': f'{duration_estimate:.2f}%',
                'Full Est%': f'{full_estimate:.2f}%',
                'Carry (30d)': f'{carry_30d:.2f}%',
                'Total Return (30d)': f'{total_return_30d:.2f}%'
            })
        
        return pd.DataFrame(results)
    
    def calculate_breakeven_metrics(self) -> Dict[str, float]:
        """Calculate breakeven and relative value metrics"""
        
        mod_dur = self.calculate_modified_duration()
        
        # Breakeven yield change (where price loss = coupon income)
        # For different holding periods
        breakeven_1m = (self.bond.coupon_rate / 12) / mod_dur * 10000  # in bp
        breakeven_3m = (self.bond.coupon_rate / 4) / mod_dur * 10000
        breakeven_1y = self.bond.coupon_rate / mod_dur * 10000
        
        # Yield per unit of duration risk
        yield_per_duration = self.bond.yield_to_maturity / mod_dur * 10000
        
        return {
            'breakeven_1m_bp': breakeven_1m,
            'breakeven_3m_bp': breakeven_3m,
            'breakeven_1y_bp': breakeven_1y,
            'yield_per_duration_bp': yield_per_duration
        }
    
    def generate_bloomberg_style_report(self) -> str:
        """Generate a Bloomberg terminal-style analytics report"""
        
        # Get all metrics
        price_data = self.calculate_price()
        risk_metrics = self.calculate_risk_metrics()
        spread_metrics = self.calculate_spread_metrics()
        breakeven = self.calculate_breakeven_metrics()
        dv01_data = self.calculate_dv01()
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("BLOOMBERG PROFESSIONAL SERVICE - BOND ANALYTICS")
        report.append("=" * 80)
        
        # Security Description
        report.append(f"\nSECURITY DESCRIPTION")
        report.append("-" * 40)
        report.append(f"Coupon:                {self.bond.coupon_rate*100:.3f}%")
        report.append(f"Maturity:              {self.bond.maturity_date.strftime('%m/%d/%Y')}")
        report.append(f"Payment Frequency:     {['Annual', 'Semi-Annual', 'Quarterly'][self.bond.frequency-1]}")
        report.append(f"Day Count:             {self.bond.day_count.value}")
        report.append(f"Settlement:            {self.bond.settlement_date.strftime('%m/%d/%Y')}")
        
        # Pricing
        report.append(f"\nPRICING")
        report.append("-" * 40)
        report.append(f"Clean Price:           {price_data['price_as_percent']:.4f}")
        report.append(f"Accrued Interest:      {price_data['accrued_interest']:.2f}")
        report.append(f"Dirty Price:           {price_data['dirty_price']:.4f}")
        report.append(f"Yield to Maturity:     {self.bond.yield_to_maturity*100:.3f}%")
        
        # Risk Analytics
        report.append(f"\nRISK ANALYTICS")
        report.append("-" * 40)
        report.append(f"Modified Duration:     {risk_metrics['modified_duration']:.4f}")
        report.append(f"Effective Duration:    {risk_metrics['effective_duration']:.4f}")
        report.append(f"Convexity:             {risk_metrics['convexity']:.2f}")
        report.append(f"Effective Convexity:   {risk_metrics['effective_convexity']:.2f}")
        
        # DV01 Metrics
        report.append(f"\nDV01 ANALYTICS")
        report.append("-" * 40)
        report.append(f"DV01 (Duration):       ${dv01_data['dv01_duration_method']:.4f}")
        report.append(f"DV01 (Repricing):      ${dv01_data['dv01_repricing_method']:.4f}")
        report.append(f"DV01 per $1MM:         ${dv01_data['dv01_per_million']:.2f}")
        
        # Spread Analytics
        report.append(f"\nSPREAD ANALYTICS")
        report.append("-" * 40)
        report.append(f"OAS:                   {spread_metrics['oas_bp']:.0f}bp")
        report.append(f"Spread Duration:       {spread_metrics['spread_duration']:.4f}")
        report.append(f"Spread DV01:           ${spread_metrics['spread_dv01']:.4f}")
        
        # Risk Metrics
        report.append(f"\nRISK METRICS")
        report.append("-" * 40)
        report.append(f"VaR (95%, 1-day):      ${risk_metrics['var_95_1d']:.2f}")
        report.append(f"VaR (95%, 10-day):     ${risk_metrics['var_95_10d']:.2f}")
        report.append(f"CVaR (99%, 1-day):     ${risk_metrics['cvar_99_1d']:.2f}")
        
        # Breakeven Analysis
        report.append(f"\nBREAKEVEN ANALYSIS")
        report.append("-" * 40)
        report.append(f"1-Month Breakeven:     {breakeven['breakeven_1m_bp']:.1f}bp")
        report.append(f"3-Month Breakeven:     {breakeven['breakeven_3m_bp']:.1f}bp")
        report.append(f"1-Year Breakeven:      {breakeven['breakeven_1y_bp']:.1f}bp")
        report.append(f"Yield/Duration:        {breakeven['yield_per_duration_bp']:.1f}bp")
        
        # Key Rate Durations
        report.append(f"\nKEY RATE DURATIONS")
        report.append("-" * 40)
        krd_df = self.calculate_key_rate_durations()
        for _, row in krd_df.iterrows():
            report.append(f"{row['Tenor']:6s}:                {row['KRD']:.4f}")
        
        return "\n".join(report)


def create_professional_visualizations(calc: ProfessionalBondCalculator):
    """Create institutional-quality visualizations"""
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Price-Yield with Greeks Overlay
    ax1 = plt.subplot(3, 3, 1)
    yields = np.linspace(0.01, 0.09, 100)
    prices = []
    durations = []
    convexities = []
    
    for y in yields:
        calc.bond.yield_to_maturity = y
        temp_calc = ProfessionalBondCalculator(calc.bond, calc.market)
        prices.append(temp_calc.calculate_price()['clean_price'])
        durations.append(temp_calc.calculate_modified_duration())
        convexities.append(temp_calc.calculate_convexity())
    
    calc.bond.yield_to_maturity = 0.05  # Reset
    
    color = 'tab:blue'
    ax1.set_xlabel('Yield (%)')
    ax1.set_ylabel('Price ($)', color=color)
    ax1.plot(yields * 100, prices, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax1b = ax1.twinx()
    color = 'tab:red'
    ax1b.set_ylabel('Duration', color=color)
    ax1b.plot(yields * 100, durations, color=color, linestyle='--', alpha=0.7)
    ax1b.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title('Price-Yield Curve with Duration Overlay', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. DV01 Heatmap
    ax2 = plt.subplot(3, 3, 2)
    maturities = np.arange(1, 31, 2)
    coupons = np.arange(0, 0.11, 0.01)
    
    dv01_grid = np.zeros((len(coupons), len(maturities)))
    
    for i, coupon in enumerate(coupons):
        for j, maturity in enumerate(maturities):
            # Convert numpy types to Python native types
            temp_bond = Bond(1000, float(coupon), float(maturity), 0.05, 2)
            temp_calc = ProfessionalBondCalculator(temp_bond)
            dv01_grid[i, j] = temp_calc.calculate_dv01()['dv01_per_million']
    
    im = ax2.imshow(dv01_grid, aspect='auto', cmap='RdYlBu_r')
    ax2.set_xticks(range(len(maturities)))
    ax2.set_xticklabels(maturities)
    ax2.set_yticks(range(len(coupons)))
    ax2.set_yticklabels([f'{c*100:.0f}%' for c in coupons])
    ax2.set_xlabel('Maturity (Years)')
    ax2.set_ylabel('Coupon Rate')
    ax2.set_title('DV01 per $1MM Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    # 3. Scenario Analysis Waterfall
    ax3 = plt.subplot(3, 3, 3)
    scenarios = calc.scenario_analysis()
    colors = ['red' if float(r['Price Δ%'][:-1]) < 0 else 'green' 
              for r in scenarios.to_dict('records')]
    
    x_pos = np.arange(len(scenarios))
    price_changes = [float(r['Price Δ%'][:-1]) for r in scenarios.to_dict('records')]
    
    ax3.bar(x_pos, price_changes, color=colors, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([r['Scenario'] for r in scenarios.to_dict('records')], 
                        rotation=45, ha='right')
    ax3.set_ylabel('Price Change (%)')
    ax3.set_title('Scenario Analysis', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Key Rate Duration Profile
    ax4 = plt.subplot(3, 3, 4)
    krd_df = calc.calculate_key_rate_durations()
    ax4.bar(krd_df['Tenor'], krd_df['KRD'], color='purple', alpha=0.7)
    ax4.set_xlabel('Key Rate Tenor')
    ax4.set_ylabel('Key Rate Duration')
    ax4.set_title('Key Rate Duration Profile', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. VaR Distribution
    ax5 = plt.subplot(3, 3, 5)
    risk_metrics = calc.calculate_risk_metrics()
    
    # Simulate returns distribution
    np.random.seed(42)
    daily_vol = calc.market.volatility_1d_bp / 10000
    returns = np.random.normal(-0.0001, daily_vol, 10000)
    price_changes = returns * risk_metrics['modified_duration'] * 100
    
    ax5.hist(price_changes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Mark VaR and CVaR
    var_95 = np.percentile(price_changes, 5)
    cvar_99 = np.percentile(price_changes, 1)
    
    ax5.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
    ax5.axvline(cvar_99, color='red', linestyle='--', linewidth=2, label=f'CVaR 99%: {cvar_99:.2f}%')
    
    ax5.set_xlabel('Daily Return (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Value at Risk Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Duration vs Convexity Scatter
    ax6 = plt.subplot(3, 3, 6)
    
    # Generate portfolio of bonds
    np.random.seed(42)
    portfolio_bonds = []
    portfolio_durations = []
    portfolio_convexities = []
    portfolio_yields = []
    
    for _ in range(50):
        coupon = np.random.uniform(0.01, 0.08)
        maturity = np.random.uniform(2, 30)
        ytm = np.random.uniform(0.02, 0.07)
        
        # Convert numpy types to Python native types
        temp_bond = Bond(1000, float(coupon), float(maturity), float(ytm), 2)
        temp_calc = ProfessionalBondCalculator(temp_bond)
        
        portfolio_durations.append(temp_calc.calculate_modified_duration())
        portfolio_convexities.append(temp_calc.calculate_convexity())
        portfolio_yields.append(ytm * 100)
    
    scatter = ax6.scatter(portfolio_durations, portfolio_convexities, 
                         c=portfolio_yields, cmap='viridis', s=50, alpha=0.6)
    
    # Highlight current bond
    current_dur = calc.calculate_modified_duration()
    current_conv = calc.calculate_convexity()
    ax6.scatter(current_dur, current_conv, color='red', s=200, 
               marker='*', edgecolor='black', linewidth=2, label='Current Bond')
    
    ax6.set_xlabel('Modified Duration')
    ax6.set_ylabel('Convexity')
    ax6.set_title('Duration-Convexity Frontier', fontweight='bold')
    plt.colorbar(scatter, ax=ax6, label='Yield (%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Breakeven Analysis
    ax7 = plt.subplot(3, 3, 7)
    holding_periods = np.arange(1, 13)  # Monthly periods
    breakevens = []
    
    for months in holding_periods:
        carry = (calc.bond.coupon_rate * months / 12) * 100
        mod_dur = calc.calculate_modified_duration()
        breakeven_bp = (carry / mod_dur) * 100
        breakevens.append(breakeven_bp)
    
    ax7.plot(holding_periods, breakevens, 'go-', linewidth=2, markersize=8)
    ax7.fill_between(holding_periods, 0, breakevens, alpha=0.3, color='green')
    ax7.set_xlabel('Holding Period (Months)')
    ax7.set_ylabel('Breakeven Yield Rise (bp)')
    ax7.set_title('Breakeven Analysis by Holding Period', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Greeks Sensitivity
    ax8 = plt.subplot(3, 3, 8)
    yield_shifts = np.linspace(-200, 200, 41)
    actual_changes = []
    duration_estimates = []
    full_estimates = []
    
    current_price = calc.calculate_price()['clean_price']
    mod_dur = calc.calculate_modified_duration()
    convexity = calc.calculate_convexity()
    
    for bp in yield_shifts:
        shift = bp / 10000
        new_price = calc.calculate_price(calc.bond.yield_to_maturity + shift)['clean_price']
        actual = ((new_price - current_price) / current_price) * 100
        dur_est = -mod_dur * shift * 100
        full_est = dur_est + 0.5 * convexity * (shift ** 2) * 100
        
        actual_changes.append(actual)
        duration_estimates.append(dur_est)
        full_estimates.append(full_est)
    
    ax8.plot(yield_shifts, actual_changes, 'b-', linewidth=2, label='Actual')
    ax8.plot(yield_shifts, duration_estimates, 'r--', linewidth=1.5, label='Duration Only')
    ax8.plot(yield_shifts, full_estimates, 'g:', linewidth=1.5, label='Duration + Convexity')
    ax8.set_xlabel('Yield Change (bp)')
    ax8.set_ylabel('Price Change (%)')
    ax8.set_title('Price Approximation Accuracy', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 9. Time Decay Analysis
    ax9 = plt.subplot(3, 3, 9)
    time_horizons = np.linspace(0, calc.bond.years_to_maturity, 50)
    durations_over_time = []
    
    for t in time_horizons:
        remaining_maturity = calc.bond.years_to_maturity - t
        if remaining_maturity > 0.1:  # Avoid very small maturities
            temp_bond = Bond(calc.bond.face_value, calc.bond.coupon_rate,
                           float(remaining_maturity), calc.bond.yield_to_maturity, calc.bond.frequency)
            temp_calc = ProfessionalBondCalculator(temp_bond)
            durations_over_time.append(temp_calc.calculate_modified_duration())
        else:
            durations_over_time.append(0)
    
    ax9.plot(time_horizons, durations_over_time, 'purple', linewidth=2)
    ax9.fill_between(time_horizons, 0, durations_over_time, alpha=0.3, color='purple')
    ax9.set_xlabel('Time Elapsed (Years)')
    ax9.set_ylabel('Modified Duration')
    ax9.set_title('Duration Decay Over Time', fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Professional Bond Analytics Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration of professional bond analytics"""
    
    print("\n" + "=" * 80)
    print("PROFESSIONAL BOND ANALYTICS SYSTEM")
    print("Institutional-Grade Fixed Income Risk Management")
    print("=" * 80)
    
    # Market data
    market = MarketData(
        risk_free_rate=0.045,
        credit_spread=0.001,
        volatility_1d_bp=12,
        curve_slope_bp=150
    )
    
    # Example 1: Investment Grade Corporate Bond
    print("\n" + "=" * 80)
    print("EXAMPLE 1: INVESTMENT GRADE CORPORATE BOND")
    print("=" * 80)
    
    corp_bond = Bond(
        face_value=1000,
        coupon_rate=0.05,
        years_to_maturity=10,
        yield_to_maturity=0.052,  # 20bp spread
        frequency=2,
        day_count=DayCountConvention.THIRTY_360
    )
    
    corp_calc = ProfessionalBondCalculator(corp_bond, market)
    print(corp_calc.generate_bloomberg_style_report())
    
    # Scenario Analysis
    print("\n" + "=" * 80)
    print("SCENARIO ANALYSIS")
    print("=" * 80)
    scenarios_df = corp_calc.scenario_analysis()
    print(scenarios_df.to_string(index=False))
    
    # Example 2: High Yield Bond
    print("\n" + "=" * 80)
    print("EXAMPLE 2: HIGH YIELD BOND")
    print("=" * 80)
    
    hy_bond = Bond(
        face_value=1000,
        coupon_rate=0.08,
        years_to_maturity=7,
        yield_to_maturity=0.095,  # 500bp spread
        frequency=2
    )
    
    hy_calc = ProfessionalBondCalculator(hy_bond, market)
    
    print(f"\nHIGH YIELD ANALYTICS")
    print("-" * 40)
    hy_price = hy_calc.calculate_price()
    hy_risk = hy_calc.calculate_risk_metrics()
    hy_spread = hy_calc.calculate_spread_metrics()
    
    print(f"Price (% of Par):      {hy_price['price_as_percent']:.3f}")
    print(f"Modified Duration:     {hy_risk['modified_duration']:.3f}")
    print(f"Convexity:             {hy_risk['convexity']:.2f}")
    print(f"Spread Duration:       {hy_spread['spread_duration']:.3f}")
    print(f"OAS:                   {hy_spread['oas_bp']:.0f}bp")
    print(f"DV01 per $1MM:         ${hy_risk['dv01'] * 1000:.2f}")
    
    # Example 3: Zero Coupon Treasury
    print("\n" + "=" * 80)
    print("EXAMPLE 3: ZERO COUPON TREASURY STRIP")
    print("=" * 80)
    
    zero_bond = Bond(
        face_value=1000,
        coupon_rate=0.0,
        years_to_maturity=20,
        yield_to_maturity=0.04,
        frequency=2
    )
    
    zero_calc = ProfessionalBondCalculator(zero_bond, market)
    zero_price = zero_calc.calculate_price()
    zero_risk = zero_calc.calculate_risk_metrics()
    
    print(f"\nZERO COUPON ANALYTICS")
    print("-" * 40)
    print(f"Price:                 ${zero_price['clean_price']:.2f}")
    print(f"Price (% of Par):      {zero_price['price_as_percent']:.3f}")
    print(f"Macaulay Duration:     {zero_calc.calculate_macaulay_duration():.2f} years")
    print(f"Modified Duration:     {zero_risk['modified_duration']:.2f}")
    print(f"Convexity:             {zero_risk['convexity']:.2f}")
    print(f"DV01 per $1MM:         ${zero_calc.calculate_dv01()['dv01_per_million']:.2f}")
    
    # Comparative Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    comparison_data = {
        'Bond Type': ['IG Corporate', 'High Yield', 'Zero Treasury'],
        'Yield': [f"{b.yield_to_maturity*100:.2f}%" for b in [corp_bond, hy_bond, zero_bond]],
        'Duration': [f"{c.calculate_modified_duration():.2f}" 
                    for c in [corp_calc, hy_calc, zero_calc]],
        'Convexity': [f"{c.calculate_convexity():.1f}" 
                     for c in [corp_calc, hy_calc, zero_calc]],
        'DV01/$1MM': [f"${c.calculate_dv01()['dv01_per_million']:.0f}" 
                     for c in [corp_calc, hy_calc, zero_calc]]
    }
    
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    
    # Generate visualizations
    print("\n📊 Generating Professional Analytics Dashboard...")
    create_professional_visualizations(corp_calc)
    
    # Interactive Mode
    print("\n" + "=" * 80)
    print("INTERACTIVE ANALYSIS MODE")
    print("=" * 80)
    
    try:
        print("\nEnter bond parameters for custom analysis:")
        print("(Press Enter for defaults)")
        
        face = float(input("Face Value [1000]: ") or 1000)
        coupon = float(input("Coupon Rate (decimal, e.g., 0.05) [0.05]: ") or 0.05)
        maturity = float(input("Years to Maturity [10]: ") or 10)
        ytm = float(input("Yield to Maturity (decimal) [0.05]: ") or 0.05)
        freq = int(input("Payment Frequency (1=Annual, 2=Semi) [2]: ") or 2)
        
        # Optional market data
        if input("\nCustomize market data? (y/n) [n]: ").lower() == 'y':
            rf_rate = float(input("Risk-free rate (decimal) [0.045]: ") or 0.045)
            vol_bp = float(input("Daily volatility (bp) [12]: ") or 12)
            custom_market = MarketData(risk_free_rate=rf_rate, volatility_1d_bp=vol_bp)
        else:
            custom_market = market
        
        # Create custom bond
        custom_bond = Bond(face, coupon, maturity, ytm, freq)
        custom_calc = ProfessionalBondCalculator(custom_bond, custom_market)
        
        # Display results
        print(custom_calc.generate_bloomberg_style_report())
        
        # Optional visualizations
        if input("\nGenerate analytics dashboard? (y/n): ").lower() == 'y':
            create_professional_visualizations(custom_calc)
        
        # Optional scenario analysis
        if input("\nRun scenario analysis? (y/n): ").lower() == 'y':
            print("\n" + "=" * 80)
            print("CUSTOM SCENARIO ANALYSIS")
            print("=" * 80)
            scenarios_df = custom_calc.scenario_analysis()
            print(scenarios_df.to_string(index=False))
            
    except ValueError as e:
        print(f"Invalid input: {e}")
        print("Using default values...")


if __name__ == "__main__":
    main()