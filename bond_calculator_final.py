"""
Corrected Real-Time Bond Analytics with Accurate Market Data
Focused on Treasury Curve and Credit Spreads with proper calculations
"""

import numpy as np
import pandas as pd
import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Bond:
    """Bond characteristics"""
    face_value: float = 1000.0
    coupon_rate: float = 0.05  # Annual coupon rate
    years_to_maturity: float = 10.0
    yield_to_maturity: float = 0.05  # Annual YTM
    frequency: int = 2  # Payment frequency (2 = semi-annual)
    
    @property
    def coupon_payment(self) -> float:
        """Periodic coupon payment"""
        return self.face_value * self.coupon_rate / self.frequency
    
    @property
    def periods(self) -> int:
        """Total number of payment periods"""
        return int(self.years_to_maturity * self.frequency)
    
    @property
    def periodic_yield(self) -> float:
        """Yield per period"""
        return self.yield_to_maturity / self.frequency


class BondCalculator:
    """Bond calculator with correct duration and convexity formulas"""
    
    def __init__(self, bond: Bond):
        self.bond = bond
    
    def calculate_price(self, ytm: float = None) -> float:
        """Calculate bond price using discounted cash flows"""
        if ytm is None:
            ytm = self.bond.yield_to_maturity
        
        periodic_yield = ytm / self.bond.frequency
        price = 0
        
        # Present value of all cash flows
        for t in range(1, self.bond.periods + 1):
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            price += cash_flow / (1 + periodic_yield) ** t
        
        return price
    
    def calculate_macaulay_duration(self) -> float:
        """
        Calculate Macaulay Duration
        Formula: Σ(t * PV(CFt)) / Price
        """
        price = self.calculate_price()
        if price == 0:
            return 0
        
        weighted_cash_flows = 0
        periodic_yield = self.bond.periodic_yield
        
        for t in range(1, self.bond.periods + 1):
            # Time in years
            time_years = t / self.bond.frequency
            
            # Cash flow at time t
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            
            # Present value of cash flow
            pv = cash_flow / (1 + periodic_yield) ** t
            
            # Weight by time
            weighted_cash_flows += time_years * pv
        
        return weighted_cash_flows / price
    
    def calculate_modified_duration(self) -> float:
        """
        Calculate Modified Duration
        Formula: Macaulay Duration / (1 + y/n)
        """
        macaulay_duration = self.calculate_macaulay_duration()
        return macaulay_duration / (1 + self.bond.periodic_yield)
    
    def calculate_convexity(self) -> float:
        """
        Calculate Convexity with CORRECT formula
        Formula: [Σ(t*(t+1) * CFt/(1+y)^t)] / [Price * (1+y)^2 * frequency^2]
        
        Where:
        - t is the period number
        - CFt is cash flow at period t
        - y is periodic yield
        - frequency converts to annual terms
        """
        price = self.calculate_price()
        if price == 0:
            return 0
        
        periodic_yield = self.bond.periodic_yield
        convexity_sum = 0
        
        for t in range(1, self.bond.periods + 1):
            # Cash flow at time t
            cash_flow = self.bond.coupon_payment
            if t == self.bond.periods:
                cash_flow += self.bond.face_value
            
            # t*(t+1) term for convexity
            time_factor = t * (t + 1)
            
            # Present value of weighted cash flow
            pv_weighted = (time_factor * cash_flow) / ((1 + periodic_yield) ** t)
            
            convexity_sum += pv_weighted
        
        # Complete convexity formula with proper scaling
        convexity = convexity_sum / (price * ((1 + periodic_yield) ** 2) * (self.bond.frequency ** 2))
        
        return convexity
    
    def calculate_dv01(self) -> Dict[str, float]:
        """Calculate Dollar Value of a Basis Point (DV01)"""
        
        # Method 1: Using modified duration
        mod_duration = self.calculate_modified_duration()
        price = self.calculate_price()
        dv01_duration = mod_duration * price * 0.0001
        
        # Method 2: Actual repricing (more accurate)
        price_up = self.calculate_price(self.bond.yield_to_maturity + 0.0001)
        price_down = self.calculate_price(self.bond.yield_to_maturity - 0.0001)
        dv01_repricing = (price_down - price_up) / 2
        
        return {
            'dv01_duration_method': dv01_duration,
            'dv01_repricing_method': dv01_repricing,
            'dv01_per_million': dv01_repricing * (1_000_000 / self.bond.face_value)
        }
    
    def price_change_analysis(self, basis_points: int) -> Dict[str, float]:
        """Analyze price changes with duration and convexity contributions"""
        
        yield_change = basis_points / 10000
        
        # Current metrics
        current_price = self.calculate_price()
        mod_duration = self.calculate_modified_duration()
        convexity = self.calculate_convexity()
        
        # New price after yield change
        new_price = self.calculate_price(self.bond.yield_to_maturity + yield_change)
        
        # Actual change
        actual_change = new_price - current_price
        actual_change_pct = (actual_change / current_price) * 100
        
        # Duration approximation (first-order)
        duration_estimate_pct = -mod_duration * yield_change * 100
        
        # Convexity contribution (second-order)
        convexity_contribution_pct = 0.5 * convexity * (yield_change ** 2) * 100
        
        # Combined estimate
        total_estimate_pct = duration_estimate_pct + convexity_contribution_pct
        
        # Approximation error
        error_pct = abs(total_estimate_pct - actual_change_pct)
        
        return {
            'actual_change_pct': actual_change_pct,
            'duration_estimate_pct': duration_estimate_pct,
            'convexity_contribution_pct': convexity_contribution_pct,
            'total_estimate_pct': total_estimate_pct,
            'approximation_error_pct': error_pct
        }


class FREDDataFetcher:
    """Fetch real-time data from FRED with corrected spread calculations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self._cache = {}
        self._cache_time = {}
        self.cache_duration = 3600  # Cache for 1 hour
    
    def _fetch_series(self, series_id: str, limit: int = 1) -> List[Dict]:
        """Fetch data from FRED API with caching"""
        
        # Check cache
        cache_key = f"{series_id}_{limit}"
        if cache_key in self._cache:
            if (datetime.now() - self._cache_time[cache_key]).seconds < self.cache_duration:
                return self._cache[cache_key]
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': limit,
            'sort_order': 'desc'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' in data:
                self._cache[cache_key] = data['observations']
                self._cache_time[cache_key] = datetime.now()
                return data['observations']
            else:
                print(f"Warning: No data found for {series_id}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {series_id}: {e}")
            return []
    
    def get_treasury_curve(self) -> Dict[str, float]:
        """Fetch current Treasury yield curve"""
        
        print("Fetching live Treasury rates from FRED...")
        
        # FRED series IDs for Treasury yields
        treasury_series = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '3Y': 'DGS3',
            '5Y': 'DGS5',
            '7Y': 'DGS7',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        
        curve = {}
        for tenor, series_id in treasury_series.items():
            observations = self._fetch_series(series_id)
            if observations and observations[0]['value'] != '.':
                try:
                    rate = float(observations[0]['value']) / 100
                    curve[tenor] = rate
                    print(f"  {tenor}: {rate*100:.3f}%")
                except ValueError:
                    print(f"  {tenor}: No data available")
        
        return curve
    
    def get_credit_spreads(self) -> Dict[str, float]:
        """Fetch corporate bond spreads - CORRECTED to show in basis points"""
        
        print("\nFetching credit spreads from FRED...")
        
        spreads = {}
        
        # These FRED series return data in percentage points, we need to convert to basis points
        spread_series = {
            'investment_grade': 'BAMLC0A0CM',      # IG Master
            'high_yield': 'BAMLH0A0HYM2',          # HY Master II
            'aaa': 'BAMLC0A1CAAA',                 # AAA
            'aa': 'BAMLC0A2CAA',                   # AA
            'a': 'BAMLC0A3CA',                     # A
            'bbb': 'BAMLC0A4CBBB',                 # BBB
            'bb': 'BAMLH0A1HYBB',                  # BB
            'b': 'BAMLH0A2HYB',                    # B
            'ccc': 'BAMLH0A3HYC'                   # CCC & Lower
        }
        
        for name, series_id in spread_series.items():
            data = self._fetch_series(series_id)
            if data and data[0]['value'] != '.':
                # FRED returns these as percentage points (e.g., 1.23 for 123bp)
                # We keep as decimal for calculations but display as bp
                spread_decimal = float(data[0]['value']) / 100
                spreads[name] = spread_decimal
                print(f"  {name.upper()}: {spread_decimal*10000:.0f}bp")
        
        return spreads
    
    def get_historical_treasury_for_vol(self, series_id: str = 'DGS10', days: int = 30) -> List[float]:
        """Get historical Treasury data for volatility calculation"""
        
        observations = self._fetch_series(series_id, limit=days)
        rates = []
        
        for obs in observations:
            if obs['value'] != '.':
                rates.append(float(obs['value']))
        
        return rates


@dataclass
class LiveMarketData:
    """Container for real-time market data with corrected calculations"""
    
    # Treasury rates
    treasury_curve: Dict[str, float] = field(default_factory=dict)
    
    # Credit spreads (stored as decimals, displayed as bp)
    credit_spreads: Dict[str, float] = field(default_factory=dict)
    
    # Key derived metrics
    risk_free_rate: float = 0.045  # 10Y Treasury
    curve_2s10s: float = 0  # 2Y-10Y spread in bp
    curve_2s30s: float = 0  # 2Y-30Y spread in bp
    curve_5s30s: float = 0  # 5Y-30Y spread in bp
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_curve_metrics(self):
        """Calculate yield curve slope metrics"""
        
        if '2Y' in self.treasury_curve and '10Y' in self.treasury_curve:
            self.curve_2s10s = (self.treasury_curve['10Y'] - self.treasury_curve['2Y']) * 10000  # Convert to bp
            
        if '2Y' in self.treasury_curve and '30Y' in self.treasury_curve:
            self.curve_2s30s = (self.treasury_curve['30Y'] - self.treasury_curve['2Y']) * 10000
            
        if '5Y' in self.treasury_curve and '30Y' in self.treasury_curve:
            self.curve_5s30s = (self.treasury_curve['30Y'] - self.treasury_curve['5Y']) * 10000
            
        if '10Y' in self.treasury_curve:
            self.risk_free_rate = self.treasury_curve['10Y']


def create_focused_visualizations(market_data: LiveMarketData):
    """Create focused, accurate visualizations of Treasury curve and credit spreads"""
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Treasury Yield Curve with Key Points
    ax1 = plt.subplot(2, 2, 1)
    
    # Sort tenors by maturity
    tenor_order = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    available_tenors = [t for t in tenor_order if t in market_data.treasury_curve]
    
    if available_tenors:
        maturities = []
        yields = []
        
        # Convert tenor strings to years
        tenor_to_years = {
            '1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1,
            '2Y': 2, '3Y': 3, '5Y': 5, '7Y': 7,
            '10Y': 10, '20Y': 20, '30Y': 30
        }
        
        for tenor in available_tenors:
            maturities.append(tenor_to_years[tenor])
            yields.append(market_data.treasury_curve[tenor] * 100)
        
        # Plot curve
        ax1.plot(maturities, yields, 'b-', linewidth=3, alpha=0.7)
        ax1.scatter(maturities, yields, color='darkblue', s=100, zorder=5)
        
        # Fill under curve
        ax1.fill_between(maturities, min(yields)-0.5, yields, alpha=0.3, color='lightblue')
        
        # Annotate key points
        key_points = {'2Y': 2, '10Y': 10, '30Y': 30}
        for label, years in key_points.items():
            if label in market_data.treasury_curve:
                y_val = market_data.treasury_curve[label] * 100
                ax1.annotate(f'{label}: {y_val:.2f}%', 
                           xy=(years, y_val),
                           xytext=(10, 10), 
                           textcoords='offset points',
                           fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Highlight inversion zone if present
        if '2Y' in market_data.treasury_curve and '10Y' in market_data.treasury_curve:
            if market_data.treasury_curve['2Y'] > market_data.treasury_curve['10Y']:
                ax1.axhspan(min(yields)-0.5, max(yields)+0.5, 
                          xmin=0.1, xmax=0.5, alpha=0.1, color='red')
                ax1.text(5, min(yields), 'INVERTED', fontsize=16, 
                        color='red', fontweight='bold', alpha=0.5)
        
        ax1.set_xlabel('Maturity (Years)', fontsize=12)
        ax1.set_ylabel('Yield (%)', fontsize=12)
        ax1.set_title(f'U.S. Treasury Yield Curve - {datetime.now().strftime("%B %d, %Y")}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, 31)
        ax1.set_ylim(min(yields)-0.5, max(yields)+0.5)
    
    # 2. Yield Curve Slopes
    ax2 = plt.subplot(2, 2, 2)
    
    slopes = {
        '2s10s': market_data.curve_2s10s,
        '2s30s': market_data.curve_2s30s,
        '5s30s': market_data.curve_5s30s
    }
    
    # Create bar chart of slopes
    slope_names = list(slopes.keys())
    slope_values = list(slopes.values())
    colors = ['green' if v > 0 else 'red' for v in slope_values]
    
    bars = ax2.bar(slope_names, slope_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, slope_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (5 if height > 0 else -15),
                f'{value:.0f}bp', 
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=11)
    
    # Add reference line at zero
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add interpretation text
    if market_data.curve_2s10s < 0:
        interpretation = "⚠️ INVERTED CURVE - Recession Signal"
        color = 'red'
    elif market_data.curve_2s10s < 50:
        interpretation = "📊 FLAT CURVE - Late Cycle"
        color = 'orange'
    else:
        interpretation = "✓ NORMAL CURVE - Expansion"
        color = 'green'
    
    ax2.text(0.5, 0.95, interpretation, 
            transform=ax2.transAxes,
            fontsize=12, fontweight='bold', 
            color=color,
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax2.set_ylabel('Spread (basis points)', fontsize=12)
    ax2.set_title('Yield Curve Slopes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(min(slope_values)-50 if min(slope_values) < 0 else -10, 
                 max(slope_values)+50)
    
    # 3. Credit Spreads by Rating
    ax3 = plt.subplot(2, 2, 3)
    
    # Order ratings from best to worst
    rating_order = ['aaa', 'aa', 'a', 'bbb', 'bb', 'b', 'ccc']
    
    ratings = []
    spreads_bp = []
    colors_list = []
    
    # Color mapping for ratings
    color_map = {
        'aaa': '#004d00',  # Dark green
        'aa': '#006600',   # Green
        'a': '#009900',    # Light green
        'bbb': '#ffcc00',  # Yellow
        'bb': '#ff9900',   # Orange
        'b': '#ff6600',    # Dark orange
        'ccc': '#cc0000'   # Red
    }
    
    for rating in rating_order:
        if rating in market_data.credit_spreads:
            ratings.append(rating.upper())
            spreads_bp.append(market_data.credit_spreads[rating] * 10000)  # Convert to bp
            colors_list.append(color_map[rating])
    
    if ratings:
        bars = ax3.bar(ratings, spreads_bp, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, spread in zip(bars, spreads_bp):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{spread:.0f}bp', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        
        # Add IG/HY dividing line
        if 'bbb' in market_data.credit_spreads and 'bb' in market_data.credit_spreads:
            ig_index = ratings.index('BBB') if 'BBB' in ratings else -1
            if ig_index >= 0:
                ax3.axvline(x=ig_index + 0.5, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
                ax3.text(ig_index + 0.5, max(spreads_bp) * 0.9, 
                        'IG | HY', ha='center', fontsize=10,
                        color='red', fontweight='bold')
    
    ax3.set_ylabel('Spread to Treasury (bp)', fontsize=12)
    ax3.set_xlabel('Credit Rating', fontsize=12)
    ax3.set_title('Corporate Bond Credit Spreads', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Historical Context and Market Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create comprehensive summary
    summary_text = f"""
    📊 BOND MARKET SUMMARY - {datetime.now().strftime("%B %d, %Y %I:%M %p")}
    {'='*60}
    
    TREASURY YIELDS:
    • 2-Year:   {market_data.treasury_curve.get('2Y', 0)*100:>6.3f}%
    • 5-Year:   {market_data.treasury_curve.get('5Y', 0)*100:>6.3f}%
    • 10-Year:  {market_data.treasury_curve.get('10Y', 0)*100:>6.3f}%
    • 30-Year:  {market_data.treasury_curve.get('30Y', 0)*100:>6.3f}%
    
    CURVE ANALYSIS:
    • 2s10s Spread: {market_data.curve_2s10s:>+6.0f}bp {' (INVERTED!)' if market_data.curve_2s10s < 0 else ''}
    • 2s30s Spread: {market_data.curve_2s30s:>+6.0f}bp
    • 5s30s Spread: {market_data.curve_5s30s:>+6.0f}bp
    
    CREDIT SPREADS:
    • Investment Grade: {market_data.credit_spreads.get('investment_grade', 0)*10000:>6.0f}bp
    • High Yield:       {market_data.credit_spreads.get('high_yield', 0)*10000:>6.0f}bp
    • IG/HY Ratio:      {market_data.credit_spreads.get('high_yield', 0)/market_data.credit_spreads.get('investment_grade', 1):>6.1f}x
    
    MARKET SIGNALS:
    """
    
    # Add market interpretation
    signals = []
    
    if market_data.curve_2s10s < 0:
        signals.append("⚠️  Inverted curve suggests recession within 12-18 months")
    elif market_data.curve_2s10s < 50:
        signals.append("📊 Flat curve indicates late economic cycle")
    else:
        signals.append("✓  Normal curve suggests healthy economic expansion")
    
    if market_data.credit_spreads.get('high_yield', 0) * 10000 > 500:
        signals.append("⚠️  Wide HY spreads indicate credit stress")
    elif market_data.credit_spreads.get('high_yield', 0) * 10000 < 300:
        signals.append("✓  Tight HY spreads suggest risk-on sentiment")
    
    summary_text += '\n'.join(f"    {signal}" for signal in signals)
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Real-Time Fixed Income Market Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def fetch_all_market_data(api_key: str) -> LiveMarketData:
    """Fetch all market data with corrected calculations"""
    
    print("=" * 60)
    print("FETCHING REAL-TIME MARKET DATA")
    print("=" * 60)
    
    # Initialize data fetcher
    fred = FREDDataFetcher(api_key)
    
    # Create market data container
    market_data = LiveMarketData()
    
    # Fetch Treasury curve
    market_data.treasury_curve = fred.get_treasury_curve()
    
    # Fetch credit spreads (now correctly scaled)
    market_data.credit_spreads = fred.get_credit_spreads()
    
    # Calculate curve metrics
    market_data.calculate_curve_metrics()
    
    print("\n" + "=" * 60)
    print("MARKET DATA SUMMARY")
    print("=" * 60)
    print(f"Risk-Free Rate (10Y): {market_data.risk_free_rate*100:.3f}%")
    print(f"2s10s Spread: {market_data.curve_2s10s:.0f}bp")
    print(f"2s30s Spread: {market_data.curve_2s30s:.0f}bp")
    
    if 'investment_grade' in market_data.credit_spreads:
        print(f"IG Spread: {market_data.credit_spreads['investment_grade']*10000:.0f}bp")
    if 'high_yield' in market_data.credit_spreads:
        print(f"HY Spread: {market_data.credit_spreads['high_yield']*10000:.0f}bp")
    
    print(f"Last Updated: {market_data.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return market_data


def main():
    """Main function with focused, accurate analysis"""
    
    print("\n" + "=" * 60)
    print("PROFESSIONAL BOND ANALYTICS WITH REAL-TIME DATA")
    print("=" * 60)
    
    # Your FRED API key
    FRED_API_KEY = "8f28060ba4e2b7a5f855b66efc0d72ae"
    
    # Fetch market data
    market_data = fetch_all_market_data(FRED_API_KEY)
    
    # Create visualizations
    print("\n📊 Generating Real-Time Market Dashboard...")
    create_focused_visualizations(market_data)
    
    # Demonstrate bond calculations with live data
    print("\n" + "=" * 60)
    print("BOND CALCULATIONS WITH LIVE MARKET DATA")
    print("=" * 60)
    
    # Example 1: 10-Year Treasury Bond
    print("\n📈 10-YEAR TREASURY BOND ANALYSIS")
    print("-" * 40)
    
    treasury_10y = Bond(
        face_value=1000,
        coupon_rate=0.04,  # 4% coupon
        years_to_maturity=10,
        yield_to_maturity=market_data.risk_free_rate,  # Live 10Y rate
        frequency=2
    )
    
    calc_treasury = BondCalculator(treasury_10y)
    
    print(f"Yield (from market):     {treasury_10y.yield_to_maturity*100:.3f}%")
    print(f"Price:                   ${calc_treasury.calculate_price():.2f}")
    print(f"Macaulay Duration:       {calc_treasury.calculate_macaulay_duration():.3f} years")
    print(f"Modified Duration:       {calc_treasury.calculate_modified_duration():.3f}")
    print(f"Convexity:               {calc_treasury.calculate_convexity():.2f}")
    
    dv01 = calc_treasury.calculate_dv01()
    print(f"DV01 (per bond):         ${dv01['dv01_repricing_method']:.4f}")
    print(f"DV01 (per $1MM):         ${dv01['dv01_per_million']:.2f}")
    
    # Example 2: Investment Grade Corporate Bond
    print("\n📈 INVESTMENT GRADE CORPORATE BOND (A-RATED)")
    print("-" * 40)
    
    # A-rated bond yields Treasury + IG spread
    ig_yield = market_data.risk_free_rate
    if 'a' in market_data.credit_spreads:
        ig_yield += market_data.credit_spreads['a']
    
    ig_bond = Bond(
        face_value=1000,
        coupon_rate=0.05,  # 5% coupon
        years_to_maturity=10,
        yield_to_maturity=ig_yield,
        frequency=2
    )
    
    calc_ig = BondCalculator(ig_bond)
    
    print(f"Treasury Yield:          {market_data.risk_free_rate*100:.3f}%")
    print(f"Credit Spread:           {market_data.credit_spreads.get('a', 0)*10000:.0f}bp")
    print(f"Total Yield:             {ig_yield*100:.3f}%")
    print(f"Price:                   ${calc_ig.calculate_price():.2f}")
    print(f"Modified Duration:       {calc_ig.calculate_modified_duration():.3f}")
    print(f"Convexity:               {calc_ig.calculate_convexity():.2f}")
    print(f"DV01 (per $1MM):         ${calc_ig.calculate_dv01()['dv01_per_million']:.2f}")
    
    # Example 3: High Yield Bond
    print("\n📈 HIGH YIELD BOND (BB-RATED)")
    print("-" * 40)
    
    hy_yield = market_data.risk_free_rate
    if 'bb' in market_data.credit_spreads:
        hy_yield += market_data.credit_spreads['bb']
    
    hy_bond = Bond(
        face_value=1000,
        coupon_rate=0.07,  # 7% coupon
        years_to_maturity=7,
        yield_to_maturity=hy_yield,
        frequency=2
    )
    
    calc_hy = BondCalculator(hy_bond)
    
    print(f"Treasury Yield:          {market_data.risk_free_rate*100:.3f}%")
    print(f"Credit Spread:           {market_data.credit_spreads.get('bb', 0)*10000:.0f}bp")
    print(f"Total Yield:             {hy_yield*100:.3f}%")
    print(f"Price:                   ${calc_hy.calculate_price():.2f}")
    print(f"Modified Duration:       {calc_hy.calculate_modified_duration():.3f}")
    print(f"Convexity:               {calc_hy.calculate_convexity():.2f}")
    print(f"DV01 (per $1MM):         ${calc_hy.calculate_dv01()['dv01_per_million']:.2f}")
    
    # Scenario Analysis
    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS: +100bp RATE SHOCK")
    print("=" * 60)
    
    scenarios = [
        ("10Y Treasury", calc_treasury),
        ("A-Rated Corp", calc_ig),
        ("BB-Rated HY", calc_hy)
    ]
    
    print(f"{'Bond Type':<15} {'Actual Δ%':>10} {'Duration Est':>12} {'Convexity Cont':>15} {'Total Est':>10} {'Error':>8}")
    print("-" * 80)
    
    for name, calculator in scenarios:
        analysis = calculator.price_change_analysis(100)
        print(f"{name:<15} {analysis['actual_change_pct']:>9.3f}% "
              f"{analysis['duration_estimate_pct']:>11.3f}% "
              f"{analysis['convexity_contribution_pct']:>14.3f}% "
              f"{analysis['total_estimate_pct']:>9.3f}% "
              f"{analysis['approximation_error_pct']:>7.3f}%")
    
    # Verify convexity formula
    print("\n" + "=" * 60)
    print("CONVEXITY FORMULA VERIFICATION")
    print("=" * 60)
    
    print("\nUsing formula: Convexity = Σ[t*(t+1) * CF_t/(1+y)^t] / [Price * (1+y)^2 * frequency^2]")
    print("\nFor the 10Y Treasury bond:")
    print(f"  Calculated Convexity: {calc_treasury.calculate_convexity():.4f}")
    
    # Show convexity effect for different yield changes
    print("\nConvexity contribution to price change:")
    for bp in [50, 100, 200, 300]:
        analysis = calc_treasury.price_change_analysis(bp)
        print(f"  {bp:3d}bp move: {analysis['convexity_contribution_pct']:>6.3f}% "
              f"(vs duration only: {analysis['duration_estimate_pct']:>7.3f}%)")
    
    # Analyze current market conditions
    print("\n" + "=" * 60)
    print("MARKET ANALYSIS")
    print("=" * 60)
    
    # Curve shape analysis
    if market_data.curve_2s10s < 0:
        print("\n🔴 YIELD CURVE INVERTED")
        print(f"   2s10s spread: {market_data.curve_2s10s:.0f}bp")
        print("   Historical signal: Recession typically follows within 12-18 months")
        print("   Recommended positioning: Reduce credit risk, extend duration")
    elif market_data.curve_2s10s < 80:
        print("\n🟡 YIELD CURVE FLAT") 
        print(f"   2s10s spread: {market_data.curve_2s10s:.0f}bp")
        print("   Signal: Late cycle dynamics")
        print("   Recommended positioning: Neutral duration, high quality credits")

    else:
        print("\n🟢 YIELD CURVE NORMAL")
        print(f"   2s10s spread: {market_data.curve_2s10s:.0f}bp")
        print("   Signal: Healthy economic expansion")
        print("   Recommended positioning: Credit overweight, curve steepeners")
    
    # Credit spread analysis
    if 'investment_grade' in market_data.credit_spreads and 'high_yield' in market_data.credit_spreads:
        ig_spread = market_data.credit_spreads['investment_grade'] * 10000
        hy_spread = market_data.credit_spreads['high_yield'] * 10000
        
        print(f"\n📈 CREDIT MARKET CONDITIONS")
        print(f"   IG Spreads: {ig_spread:.0f}bp")
        print(f"   HY Spreads: {hy_spread:.0f}bp")
        print(f"   HY/IG Ratio: {hy_spread/ig_spread:.1f}x")
        
        if hy_spread < 250:
            print("   Signal: Tight spreads, search for yield")
        elif hy_spread > 400:
            print("   Signal: Wide spreads, credit stress present")
        else:
            print("   Signal: Normal credit conditions")
    
    print("\n✅ Analysis complete!")
    
if __name__ == "__main__":
    main()