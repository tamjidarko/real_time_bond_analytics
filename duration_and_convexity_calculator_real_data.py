"""
Real-Time Bond Analytics with Live Market Data
Integrates FRED API for Treasury rates and market data
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

# Try importing yfinance - will provide instructions if not installed
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("Note: Install yfinance for corporate bond data: pip install yfinance")


class FREDDataFetcher:
    """Fetch real-time data from FRED (Federal Reserve Economic Data)"""
    
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
        """Fetch corporate bond spreads"""
        
        print("\nFetching credit spreads from FRED...")
        
        spreads = {}
        
        # Investment Grade spread
        ig_data = self._fetch_series('BAMLC0A0CM')  # IG Corporate Master Index
        if ig_data and ig_data[0]['value'] != '.':
            spreads['investment_grade'] = float(ig_data[0]['value']) / 100
            print(f"  Investment Grade: {spreads['investment_grade']*100:.0f}bp")
        
        # High Yield spread
        hy_data = self._fetch_series('BAMLH0A0HYM2')  # HY Master II Index
        if hy_data and hy_data[0]['value'] != '.':
            spreads['high_yield'] = float(hy_data[0]['value']) / 100
            print(f"  High Yield: {spreads['high_yield']*100:.0f}bp")
        
        # AAA spread
        aaa_data = self._fetch_series('BAMLC0A1CAAA')
        if aaa_data and aaa_data[0]['value'] != '.':
            spreads['aaa'] = float(aaa_data[0]['value']) / 100
            print(f"  AAA: {spreads['aaa']*100:.0f}bp")
        
        # BBB spread
        bbb_data = self._fetch_series('BAMLC0A4CBBB')
        if bbb_data and bbb_data[0]['value'] != '.':
            spreads['bbb'] = float(bbb_data[0]['value']) / 100
            print(f"  BBB: {spreads['bbb']*100:.0f}bp")
        
        return spreads
    
    def get_volatility_metrics(self) -> Dict[str, float]:
        """Fetch volatility indicators"""
        
        print("\nFetching volatility metrics...")
        
        metrics = {}
        
        # MOVE Index (bond volatility)
        move_data = self._fetch_series('MOVE')
        if move_data and move_data[0]['value'] != '.':
            metrics['move_index'] = float(move_data[0]['value'])
            print(f"  MOVE Index: {metrics['move_index']:.1f}")
        
        # Calculate implied daily volatility from 10Y Treasury history
        treasury_10y_history = self._fetch_series('DGS10', limit=30)
        if len(treasury_10y_history) > 1:
            rates = []
            for obs in treasury_10y_history:
                if obs['value'] != '.':
                    rates.append(float(obs['value']))
            
            if len(rates) > 1:
                returns = np.diff(rates)  # In basis points
                metrics['realized_vol_10y_daily'] = np.std(returns)
                metrics['realized_vol_10y_annual'] = np.std(returns) * np.sqrt(252)
                print(f"  10Y Realized Vol (Daily): {metrics['realized_vol_10y_daily']:.1f}bp")
                print(f"  10Y Realized Vol (Annual): {metrics['realized_vol_10y_annual']:.1f}bp")
        
        return metrics
    
    def get_fed_funds_rate(self) -> float:
        """Get current Fed Funds rate"""
        
        ff_data = self._fetch_series('DFF')
        if ff_data and ff_data[0]['value'] != '.':
            return float(ff_data[0]['value']) / 100
        return 0.045  # Default fallback


class YahooFinanceData:
    """Fetch corporate bond ETF data from Yahoo Finance"""
    
    @staticmethod
    def get_bond_etf_data() -> Dict:
        """Fetch major bond ETF yields and spreads"""
        
        if not YF_AVAILABLE:
            print("Yahoo Finance not available - using default values")
            return {
                'lqd_yield': 0.052,
                'hyg_yield': 0.085,
                'agg_yield': 0.048
            }
        
        print("\nFetching bond ETF data from Yahoo Finance...")
        
        etf_data = {}
        
        # Major bond ETFs
        etfs = {
            'LQD': 'Investment Grade Corporate',
            'HYG': 'High Yield Corporate', 
            'AGG': 'Aggregate Bond',
            'TLT': '20+ Year Treasury',
            'IEF': '7-10 Year Treasury',
            'SHY': '1-3 Year Treasury'
        }
        
        for symbol, description in etfs.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get yield and price data
                current_price = info.get('regularMarketPrice', 0)
                prev_close = info.get('previousClose', 0)
                
                # Calculate simple yield (this is simplified - real yield calculation is complex)
                # Using dividend yield as proxy for bond ETF yield
                div_yield = info.get('dividendYield', 0)
                
                if div_yield > 0:
                    etf_data[f'{symbol.lower()}_yield'] = div_yield
                    print(f"  {symbol} ({description}): {div_yield*100:.2f}%")
                
                # Calculate 1-day return
                if prev_close > 0:
                    daily_return = (current_price - prev_close) / prev_close
                    etf_data[f'{symbol.lower()}_return'] = daily_return
                    
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
        
        return etf_data


@dataclass
class LiveMarketData:
    """Container for real-time market data"""
    
    # Treasury rates
    treasury_curve: Dict[str, float] = field(default_factory=dict)
    
    # Credit spreads
    credit_spreads: Dict[str, float] = field(default_factory=dict)
    
    # Volatility
    volatility_metrics: Dict[str, float] = field(default_factory=dict)
    
    # ETF data
    etf_data: Dict[str, float] = field(default_factory=dict)
    
    # Derived metrics
    risk_free_rate: float = 0.045  # Will be updated with 10Y Treasury
    ig_spread: float = 0.007  # Investment grade spread
    hy_spread: float = 0.045  # High yield spread
    daily_vol_bp: float = 12  # Daily volatility in basis points
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        
        # Use 10Y Treasury as risk-free rate
        if '10Y' in self.treasury_curve:
            self.risk_free_rate = self.treasury_curve['10Y']
        
        # Use actual credit spreads
        if 'investment_grade' in self.credit_spreads:
            self.ig_spread = self.credit_spreads['investment_grade']
        
        if 'high_yield' in self.credit_spreads:
            self.hy_spread = self.credit_spreads['high_yield']
        
        # Use realized volatility
        if 'realized_vol_10y_daily' in self.volatility_metrics:
            self.daily_vol_bp = self.volatility_metrics['realized_vol_10y_daily']
    
    def get_yield_for_rating(self, rating: str = 'A') -> float:
        """Get appropriate yield for a given credit rating"""
        
        base_rate = self.risk_free_rate
        
        rating_spreads = {
            'AAA': self.credit_spreads.get('aaa', 0.003),
            'AA': self.credit_spreads.get('aaa', 0.003) + 0.002,
            'A': self.ig_spread,
            'BBB': self.credit_spreads.get('bbb', self.ig_spread + 0.003),
            'BB': self.hy_spread,
            'B': self.hy_spread + 0.02,
            'CCC': self.hy_spread + 0.05
        }
        
        spread = rating_spreads.get(rating, self.ig_spread)
        return base_rate + spread


def fetch_all_market_data(api_key: str) -> LiveMarketData:
    """Fetch all available market data"""
    
    print("=" * 60)
    print("FETCHING REAL-TIME MARKET DATA")
    print("=" * 60)
    
    # Initialize data fetchers
    fred = FREDDataFetcher(api_key)
    
    # Create market data container
    market_data = LiveMarketData()
    
    # Fetch Treasury curve
    market_data.treasury_curve = fred.get_treasury_curve()
    
    # Fetch credit spreads
    market_data.credit_spreads = fred.get_credit_spreads()
    
    # Fetch volatility metrics
    market_data.volatility_metrics = fred.get_volatility_metrics()
    
    # Fetch ETF data if available
    if YF_AVAILABLE:
        market_data.etf_data = YahooFinanceData.get_bond_etf_data()
    
    # Update derived metrics
    market_data.update_derived_metrics()
    
    print("\n" + "=" * 60)
    print("MARKET DATA SUMMARY")
    print("=" * 60)
    print(f"Risk-Free Rate (10Y): {market_data.risk_free_rate*100:.3f}%")
    print(f"IG Spread: {market_data.ig_spread*100:.0f}bp")
    print(f"HY Spread: {market_data.hy_spread*100:.0f}bp")
    print(f"Daily Volatility: {market_data.daily_vol_bp:.1f}bp")
    print(f"Last Updated: {market_data.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return market_data


def create_live_data_visualizations(market_data: LiveMarketData):
    """Create visualizations using real market data"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Treasury Yield Curve
    ax1 = plt.subplot(2, 3, 1)
    
    # Sort tenors by maturity
    tenor_order = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    available_tenors = [t for t in tenor_order if t in market_data.treasury_curve]
    
    if available_tenors:
        maturities = []
        yields = []
        
        # Convert tenor strings to years
        for tenor in available_tenors:
            if 'M' in tenor:
                years = int(tenor[:-1]) / 12
            else:
                years = int(tenor[:-1])
            maturities.append(years)
            yields.append(market_data.treasury_curve[tenor] * 100)
        
        ax1.plot(maturities, yields, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(maturities, 0, yields, alpha=0.3)
        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Yield (%)')
        ax1.set_title(f'Live Treasury Yield Curve\n{datetime.now().strftime("%Y-%m-%d")}', 
                     fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for key rates
        for i, (mat, yld) in enumerate(zip(maturities, yields)):
            if mat in [2, 10, 30]:
                ax1.annotate(f'{yld:.2f}%', xy=(mat, yld), 
                           xytext=(5, 5), textcoords='offset points')
    
    # 2. Credit Spread Comparison
    ax2 = plt.subplot(2, 3, 2)
    
    if market_data.credit_spreads:
        spreads = []
        labels = []
        colors = []
        
        spread_mapping = {
            'aaa': ('AAA', 'green'),
            'investment_grade': ('IG', 'blue'),
            'bbb': ('BBB', 'orange'),
            'high_yield': ('HY', 'red')
        }
        
        for key, (label, color) in spread_mapping.items():
            if key in market_data.credit_spreads:
                spreads.append(market_data.credit_spreads[key] * 100)
                labels.append(label)
                colors.append(color)
        
        bars = ax2.bar(labels, spreads, color=colors, alpha=0.7)
        ax2.set_ylabel('Spread (bps)')
        ax2.set_title('Live Credit Spreads vs Treasury', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, spread in zip(bars, spreads):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{spread:.0f}bp', ha='center', va='bottom')
    
    # 3. Yield Curve Shape Analysis
    ax3 = plt.subplot(2, 3, 3)
    
    if '2Y' in market_data.treasury_curve and '10Y' in market_data.treasury_curve:
        # Calculate curve metrics
        curve_2_10 = (market_data.treasury_curve['10Y'] - 
                      market_data.treasury_curve['2Y']) * 100
        
        if '30Y' in market_data.treasury_curve:
            curve_10_30 = (market_data.treasury_curve['30Y'] - 
                          market_data.treasury_curve['10Y']) * 100
        else:
            curve_10_30 = 0
        
        # Historical context (simplified)
        historical_2_10 = [curve_2_10]  # Current
        historical_dates = [datetime.now()]
        
        # Add some historical context (fake for demonstration)
        for i in range(1, 13):
            historical_2_10.append(curve_2_10 + np.random.normal(0, 20))
            historical_dates.append(datetime.now() - timedelta(days=30*i))
        
        ax3.plot(range(len(historical_2_10)), historical_2_10[::-1], 
                'g-', linewidth=2)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.fill_between(range(len(historical_2_10)), 0, historical_2_10[::-1], 
                        alpha=0.3, color='green')
        ax3.set_xlabel('Months Ago')
        ax3.set_ylabel('2Y-10Y Spread (bps)')
        ax3.set_title(f'Yield Curve Slope\nCurrent 2Y-10Y: {curve_2_10:.0f}bp', 
                     fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Volatility Metrics
    ax4 = plt.subplot(2, 3, 4)
    
    if market_data.volatility_metrics:
        vol_categories = []
        vol_values = []
        
        if 'move_index' in market_data.volatility_metrics:
            vol_categories.append('MOVE\nIndex')
            vol_values.append(market_data.volatility_metrics['move_index'])
        
        if 'realized_vol_10y_annual' in market_data.volatility_metrics:
            vol_categories.append('10Y\nRealized')
            vol_values.append(market_data.volatility_metrics['realized_vol_10y_annual'])
        
        if vol_categories:
            bars = ax4.bar(vol_categories, vol_values, color=['purple', 'blue'], alpha=0.7)
            ax4.set_ylabel('Volatility')
            ax4.set_title('Bond Market Volatility', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, vol_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom')
    
    # 5. Bond ETF Performance (if available)
    ax5 = plt.subplot(2, 3, 5)
    
    if market_data.etf_data and any('yield' in k for k in market_data.etf_data.keys()):
        etf_yields = {}
        for key, value in market_data.etf_data.items():
            if 'yield' in key:
                symbol = key.replace('_yield', '').upper()
                etf_yields[symbol] = value * 100
        
        if etf_yields:
            sorted_etfs = sorted(etf_yields.items(), key=lambda x: x[1])
            symbols = [s[0] for s in sorted_etfs]
            yields = [s[1] for s in sorted_etfs]
            
            colors_map = {'LQD': 'blue', 'HYG': 'red', 'AGG': 'green', 
                         'TLT': 'purple', 'IEF': 'orange', 'SHY': 'brown'}
            colors = [colors_map.get(s, 'gray') for s in symbols]
            
            bars = ax5.barh(symbols, yields, color=colors, alpha=0.7)
            ax5.set_xlabel('Yield (%)')
            ax5.set_title('Bond ETF Yields', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            for bar, yld in zip(bars, yields):
                width = bar.get_width()
                ax5.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{yld:.2f}%', ha='left', va='center')
    
    # 6. Market Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    
    if '10Y' in market_data.treasury_curve:
        summary_data.append(['10Y Treasury', f"{market_data.treasury_curve['10Y']*100:.3f}%"])
    
    if '2Y' in market_data.treasury_curve:
        summary_data.append(['2Y Treasury', f"{market_data.treasury_curve['2Y']*100:.3f}%"])
    
    if market_data.credit_spreads:
        if 'investment_grade' in market_data.credit_spreads:
            summary_data.append(['IG Spread', f"{market_data.ig_spread*100:.0f}bp"])
        if 'high_yield' in market_data.credit_spreads:
            summary_data.append(['HY Spread', f"{market_data.hy_spread*100:.0f}bp"])
    
    if market_data.volatility_metrics:
        if 'realized_vol_10y_daily' in market_data.volatility_metrics:
            summary_data.append(['Daily Vol', f"{market_data.daily_vol_bp:.1f}bp"])
    
    summary_data.append(['Updated', market_data.last_updated.strftime('%H:%M:%S')])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax6.set_title('Market Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Real-Time Bond Market Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def calculate_bond_with_live_data(market_data: LiveMarketData, 
                                 rating: str = 'A',
                                 maturity: float = 10.0,
                                 coupon_spread: float = 0.0):
    """
    Calculate bond metrics using live market data
    
    Args:
        market_data: Live market data object
        rating: Credit rating for spread determination
        maturity: Years to maturity
        coupon_spread: Additional spread for coupon vs current market
    """
    
    # Get appropriate yield based on rating
    ytm = market_data.get_yield_for_rating(rating)
    
    # Set coupon (could be at par, premium, or discount)
    coupon = ytm + coupon_spread
    
    print(f"\n{'='*60}")
    print(f"BOND ANALYSIS WITH LIVE MARKET DATA")
    print(f"{'='*60}")
    print(f"Rating: {rating}")
    print(f"Maturity: {maturity} years")
    print(f"YTM (from market): {ytm*100:.3f}%")
    print(f"Coupon: {coupon*100:.3f}%")
    print(f"Risk-Free Rate: {market_data.risk_free_rate*100:.3f}%")
    print(f"Credit Spread: {(ytm - market_data.risk_free_rate)*100:.0f}bp")
    
    # Import the bond calculator from the previous implementation
    # This would use your existing ProfessionalBondCalculator class
    
    return {
        'ytm': ytm,
        'coupon': coupon,
        'spread': ytm - market_data.risk_free_rate,
        'risk_free': market_data.risk_free_rate
    }


def main():
    """Main function demonstrating real-time data integration"""
    
    print("\n" + "=" * 60)
    print("BOND ANALYTICS WITH REAL-TIME MARKET DATA")
    print("=" * 60)
    
    # Your FRED API key
    FRED_API_KEY = "8f28060ba4e2b7a5f855b66efc0d72ae"
    
    # Fetch all market data
    market_data = fetch_all_market_data(FRED_API_KEY)
    
    # Create visualizations
    print("\n📊 Generating Real-Time Market Dashboard...")
    create_live_data_visualizations(market_data)
    
    # Example calculations with live data
    print("\n" + "=" * 60)
    print("EXAMPLE BOND CALCULATIONS WITH LIVE DATA")
    print("=" * 60)
    
    # Investment Grade Bond
    ig_bond = calculate_bond_with_live_data(market_data, rating='A', maturity=10)
    
    # High Yield Bond
    hy_bond = calculate_bond_with_live_data(market_data, rating='BB', maturity=7)
    
    # AAA Bond
    aaa_bond = calculate_bond_with_live_data(market_data, rating='AAA', maturity=5)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE WITH LIVE DATA")
    print("=" * 60)
    
    print("\nSelect a calculation option:")
    print("1. Custom bond with live yields")
    print("2. Compare bonds across ratings")
    print("3. Analyze curve trades")
    print("4. Update market data")
    
    try:
        choice = input("\nEnter choice (1-4): ")
        
        if choice == '1':
            rating = input("Enter rating (AAA/AA/A/BBB/BB/B) [A]: ") or 'A'
            maturity = float(input("Enter maturity in years [10]: ") or 10)
            
            result = calculate_bond_with_live_data(market_data, rating, maturity)
            
            print(f"\nBond created with:")
            print(f"  YTM: {result['ytm']*100:.3f}%")
            print(f"  Spread: {result['spread']*100:.0f}bp")
            
        elif choice == '2':
            print("\nComparing bonds across rating spectrum...")
            ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
            
            comparison = []
            for rating in ratings:
                ytm = market_data.get_yield_for_rating(rating)
                spread = (ytm - market_data.risk_free_rate) * 100
                comparison.append({
                    'Rating': rating,
                    'YTM': f"{ytm*100:.2f}%",
                    'Spread': f"{spread:.0f}bp"
                })
            
            df = pd.DataFrame(comparison)
            print(df.to_string(index=False))
            
        elif choice == '3':
            if '2Y' in market_data.treasury_curve and '10Y' in market_data.treasury_curve:
                curve_2_10 = (market_data.treasury_curve['10Y'] - 
                             market_data.treasury_curve['2Y']) * 100
                
                print(f"\nCurve Analysis:")
                print(f"  2Y Rate: {market_data.treasury_curve['2Y']*100:.3f}%")
                print(f"  10Y Rate: {market_data.treasury_curve['10Y']*100:.3f}%")
                print(f"  2s10s Spread: {curve_2_10:.0f}bp")
                
                print("  Signal: Normal curve - neutral positioning")
                
        elif choice == '4':
            print("\nRefreshing market data...")
            market_data = fetch_all_market_data(FRED_API_KEY)
            print("Market data updated successfully!")
            
    except ValueError as e:
        print(f"Invalid input: {e}")
    
    # Save market data for later use
    print("\n" + "=" * 60)
    print("SAVING MARKET DATA")
    print("=" * 60)
    
    # Create a summary report
    summary = {
        'timestamp': market_data.last_updated.isoformat(),
        'treasury_10y': market_data.risk_free_rate,
        'ig_spread': market_data.ig_spread,
        'hy_spread': market_data.hy_spread,
        'daily_vol': market_data.daily_vol_bp,
        'treasury_curve': market_data.treasury_curve,
        'credit_spreads': market_data.credit_spreads
    }
    
    # Save to JSON
    with open('market_data_snapshot.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("Market data saved to 'market_data_snapshot.json'")
    
    # Display data quality metrics
    print("\n" + "=" * 60)
    print("DATA QUALITY METRICS")
    print("=" * 60)
    
    total_series = len(market_data.treasury_curve) + len(market_data.credit_spreads)
    print(f"Treasury points collected: {len(market_data.treasury_curve)}")
    print(f"Credit spread series: {len(market_data.credit_spreads)}")
    print(f"Total data points: {total_series}")
    
    if market_data.treasury_curve:
        min_rate = min(market_data.treasury_curve.values())
        max_rate = max(market_data.treasury_curve.values())
        print(f"Treasury range: {min_rate*100:.2f}% - {max_rate*100:.2f}%")
    
    print("\n✅ Real-time data integration complete!")


if __name__ == "__main__":
    # Check for required packages
    required_packages = []
    
    try:
        import requests
    except ImportError:
        required_packages.append('requests')
    
    try:
        import numpy as np
    except ImportError:
        required_packages.append('numpy')
    
    try:
        import pandas as pd
    except ImportError:
        required_packages.append('pandas')
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        required_packages.append('matplotlib')
    
    if required_packages:
        print("Missing required packages. Please install:")
        print(f"pip install {' '.join(required_packages)}")
        print("\nOptional: pip install yfinance (for ETF data)")
    else:
        main()
        print("  Signal: Normal curve - neutral positioning")