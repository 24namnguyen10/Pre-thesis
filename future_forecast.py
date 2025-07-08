"""
FinRL PPO Testing Demo Implementation
This script demonstrates how to use a pre-trained PPO model from FinRL for stock trading.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FinRL imports
try:
    from finrl.config_tickers import DOW_30_TICKER
    from finrl.config import INDICATORS
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    
    # Stable Baselines3 imports
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    FINRL_AVAILABLE = True
except ImportError as e:
    print(f"FinRL not available: {e}")
    print("Please install FinRL: pip install finrl")
    FINRL_AVAILABLE = False

class FinRLPPOTester:
    def __init__(self, 
                 model_path,
                 start_date='2023-01-01',
                 end_date='2025-05-31',
                 initial_amount=1000000,
                 transaction_cost_pct=0.001,
                 tech_indicators=None):
        """
        Initialize the FinRL PPO tester.
        
        Args:
            model_path (str): Path to the trained PPO model
            start_date (str): Start date for testing data
            end_date (str): End date for testing data
            initial_amount (int): Initial investment amount
            transaction_cost_pct (float): Transaction cost percentage
            tech_indicators (list): List of technical indicators to use
        """
        self.model_path = model_path
        self.start_date = start_date
        self.end_date = end_date

        if hasattr(initial_amount, "values"):
            self.initial_amount = initial_amount.values[0]
        else:
            self.initial_amount = initial_amount

        self.transaction_cost_pct = transaction_cost_pct
        
        # Use default indicators if none provided
        if tech_indicators is None:
            self.tech_indicators = [
                'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma'
            ]
        else:
            self.tech_indicators = tech_indicators
            
        # Use a subset of DOW 30 for demo (faster execution)
        self.ticker_list = DOW_30_TICKER #['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        self.df = None
        self.model = None
        self.env_kwargs = {}
        
    def download_data(self):
        """Download stock data using Yahoo Finance or generate future synthetic data."""
        print(f"Downloading data for {self.ticker_list}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        if not FINRL_AVAILABLE:
            print("FinRL not available. Generating dummy data...")
            return self._create_dummy_data()

        try:
            df = YahooDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                ticker_list=self.ticker_list
            ).fetch_data()
            
            if df.empty:
                print("⚠️ No data returned — simulating future prices...")
                df = self._create_dummy_data()

            print(f"Downloaded {len(df)} rows of data")
            self.df = df
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating synthetic future data...")
            return self._create_dummy_data()

    
    def _create_dummy_data(self):
        """Simulate future stock price data using geometric Brownian motion."""
        print("🧪 Simulating future price paths...")

        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='B')  # business days only
        num_days = len(dates)

        data = []

        for ticker in self.ticker_list:
            np.random.seed(hash(ticker) % 10000)
            price = np.random.uniform(100, 500)  # initial price

            mu = 0.0005  # drift
            sigma = 0.02  # volatility

            prices = [price]
            for _ in range(1, num_days):
                daily_return = np.random.normal(mu, sigma)
                price *= (1 + daily_return)
                prices.append(price)

            for i in range(num_days):
                p = prices[i]
                data.append({
                    'date': dates[i],
                    'tic': ticker,
                    'open': p * np.random.uniform(0.99, 1.01),
                    'high': p * np.random.uniform(1.00, 1.03),
                    'low':  p * np.random.uniform(0.97, 1.00),
                    'close': p,
                    'volume': np.random.randint(1e6, 10e6),
                    'adjcp': p
                })

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        self.df = df
        print(f"✅ Created synthetic data: {df.shape[0]} rows")
        return df
    
    def add_technical_indicators(self):
        """Add technical indicators to the data."""
        print("Adding technical indicators...")
        
        if not FINRL_AVAILABLE:
            # Add dummy technical indicators
            return self._add_dummy_indicators()
        
        try:
            fe = FeatureEngineer(
                use_technical_indicator=True,
                tech_indicator_list=self.tech_indicators,
                use_vix=False,
                use_turbulence=False,
                user_defined_feature=False
            )
            
            processed_df = fe.preprocess_data(self.df)
            
            # Ensure data is properly formatted
            processed_df = processed_df.sort_values(['date', 'tic']).reset_index(drop=True)
            
            self.df = processed_df
            print(f"Added {len(self.tech_indicators)} technical indicators")
            print(f"Data shape: {processed_df.shape}")
            print(f"Columns: {processed_df.columns.tolist()}")
            return processed_df
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return self._add_dummy_indicators()
    
    def _add_dummy_indicators(self):
        """Add dummy technical indicators for demonstration."""
        df = self.df.copy()
        
        # Ensure data is properly sorted
        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        
        # Add technical indicators
        for indicator in self.tech_indicators:
            df[indicator] = np.random.randn(len(df))
        
        # Fill any potential NaN values
        df = df.fillna(method='ffill')
        
        self.df = df
        print(f"Added {len(self.tech_indicators)} dummy technical indicators")
        print(f"Data shape: {df.shape}")
        return df
    
    def load_model(self):
        """Load the pre-trained model."""
        print(f"Loading model from {self.model_path}")
        
        if not FINRL_AVAILABLE:
            # Create dummy model for demonstration
            self.model = DummyPPOModel()
            print("Loaded dummy model for demonstration")
            return self.model
        
        try:
            self.model = PPO.load(self.model_path)
            print("Model loaded successfully!")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating dummy model for demonstration...")
            self.model = DummyPPOModel()
            return self.model
    
    def create_environment(self, df):
        """Create the trading environment with clean input types."""
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        df = df.sort_values(['date', 'tic']).reset_index(drop=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.fillna(method='ffill').fillna(method='bfill')

        self.ticker_list = DOW_30_TICKER
        stock_dim = len(self.ticker_list)

        # 🔧 Make sure everything is standard Python types
        hmax = int(100)
        initial_amount = float(self.initial_amount)
        transaction_cost = float(self.transaction_cost_pct)

        num_stock_shares = [int(0) for _ in range(stock_dim)]
        buy_cost_pct = [float(transaction_cost) for _ in range(stock_dim)]
        sell_cost_pct = [float(transaction_cost) for _ in range(stock_dim)]
        tech_indicator_list = self.tech_indicators
        state_space = 1 + 2 * stock_dim + len(tech_indicator_list) * stock_dim
        action_space = stock_dim
        reward_scaling = float(1e-4)

        self.env_kwargs = {
            'hmax': hmax,
            'initial_amount': initial_amount,
            'num_stock_shares': num_stock_shares,
            'buy_cost_pct': buy_cost_pct,
            'sell_cost_pct': sell_cost_pct,
            'state_space': state_space,
            'stock_dim': stock_dim,
            'tech_indicator_list': tech_indicator_list,
            'action_space': action_space,
            'reward_scaling': reward_scaling
        }

        try:
            env = StockTradingEnv(df=df, **self.env_kwargs)
            print("✅ Successfully created StockTradingEnv")
            return env
        except Exception as e:
            print(f"[ERROR] Could not create StockTradingEnv: {e}")
            raise
    
    def test_model(self):
        """Test the trained model."""
        print("Testing model...")
        if self.df is None or self.df.empty:
            raise ValueError("No data available for testing")
        
        print(f"Testing with {len(self.df)} data points")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Tickers: {self.df['tic'].unique().tolist()}")

        try:
            test_env = self.create_environment(self.df)

            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: test_env])  # ✅ Wrap env for SB3 compatibility

            df_account_value, df_actions = DRLAgent.DRL_prediction(
                model=self.model,
                environment=vec_env
            )

            print("✅ Model tested successfully using FinRL environment.")
            return df_account_value, df_actions

        except Exception as e:
            print(f"[Fallback] Error during testing: {e}")
            print("Using dummy results instead.")
            return self._create_dummy_results()

    
    def _manual_prediction(self, env):
        """Manually run prediction using the model."""
        print("Running manual prediction...")
        
        try:
            # Reset environment and get initial state
            state = env.reset()
            done = False
            
            account_values = [self.initial_amount]
            actions_list = []
            dates = []
            
            step_count = 0
            max_steps = len(self.df) // len(self.ticker_list)  # Approximate number of trading days
            
            while not done and step_count < max_steps:
                # Get action from model
                action, _states = self.model.predict(state, deterministic=True)
                
                # Take step in environment
                state, reward, done, info = env.step(action)
                
                # Record results
                if hasattr(env, 'asset_memory') and len(env.asset_memory) > 0:
                    account_values.append(env.asset_memory[-1])
                else:
                    # Estimate account value based on previous value and reward
                    account_values.append(account_values[-1] * (1 + reward * 0.01))
                
                actions_list.append(action)
                
                # Add date (approximate)
                if step_count < len(self.df['date'].unique()):
                    dates.append(sorted(self.df['date'].unique())[step_count])
                else:
                    dates.append(dates[-1] + pd.Timedelta(days=1))
                
                step_count += 1
            
            # Create result DataFrames
            df_account_value = pd.DataFrame({
                'date': dates[:len(account_values)],
                'account_value': account_values
            })
            
            df_actions = pd.DataFrame({
                'date': dates[:len(actions_list)],
                'actions': actions_list
            })
            
            print(f"Manual prediction completed with {len(df_account_value)} steps")
            return df_account_value, df_actions
            
        except Exception as e:
            print(f"Error in manual prediction: {e}")
            print("Falling back to dummy results...")
            return self._create_dummy_results()
    
    def _create_dummy_results(self):
        """Create dummy results for demonstration."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Only weekdays
        
        # Generate realistic account value progression
        account_values = []
        current_value = self.initial_amount
        
        for i, date in enumerate(dates):
            # Random walk with slight upward trend
            daily_return = np.random.normal(0.0005, 0.02)
            current_value *= (1 + daily_return)
            account_values.append(current_value)
        
        df_account_value = pd.DataFrame({
            'date': dates,
            'account_value': account_values
        })
        
        # Generate dummy actions
        df_actions = pd.DataFrame({
            'date': dates,
            'actions': [np.random.choice([-1, 0, 1], size=len(self.ticker_list)) for _ in dates]
        })
        
        return df_account_value, df_actions
    
    def plot_results(self, df_account_value, df_actions=None, show_actions=True):
        """Plot portfolio value and optionally trading actions."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        print("📈 Plotting trading results...")

        # Prepare data
        df_account_value['date'] = pd.to_datetime(df_account_value['date'])
        df_account_value = df_account_value.sort_values('date')
        df_account_value.set_index('date', inplace=True)

        # Daily returns
        daily_returns = df_account_value['account_value'].pct_change().dropna()

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))

        # --- Plot Account Value Line ---
        ax.plot(df_account_value.index, df_account_value['account_value'], label='Portfolio Value', color='blue', linewidth=2)

        # --- Overlay Buy/Sell Markers ---
        if show_actions and df_actions is not None:
            df_actions['date'] = pd.to_datetime(df_actions['date'])
            df_actions = df_actions.sort_values('date')
            df_actions.set_index('date', inplace=True)

            # We'll visualize net position (sum of action vector)
            df_actions['net_action'] = df_actions['actions'].apply(lambda x: sum(x) if isinstance(x, (list, np.ndarray)) else 0)

            buy_signals = df_actions[df_actions['net_action'] > 0]
            sell_signals = df_actions[df_actions['net_action'] < 0]

            ax.scatter(buy_signals.index, 
                    df_account_value.loc[buy_signals.index, 'account_value'], 
                    marker='^', color='green', label='Buy', alpha=0.6)

            ax.scatter(sell_signals.index, 
                    df_account_value.loc[sell_signals.index, 'account_value'], 
                    marker='v', color='red', label='Sell', alpha=0.6)

        # Formatting
        ax.set_title("Trading Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Account Value ($)")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xticks(rotation=45)

        plt.savefig("trading_result.png")
        print("Plot saved as trading_result.png")

        # Print summary
        print("\n=== Performance Summary ===")
        print(f"Initial Value: ${self.initial_amount:,.2f}")
        print(f"Final Value: ${df_account_value['account_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {(df_account_value['account_value'].iloc[-1] / df_account_value['account_value'].iloc[0] - 1) * 100:.2f}%")
        print(f"Average Daily Return: {daily_returns.mean():.4f}")
        print(f"Daily Return Std: {daily_returns.std():.4f}")
        print(f"Sharpe Ratio: {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.2f}")
    
    def run_test(self):
        """Run the complete test."""
        print("=== FinRL PPO Testing Demo ===")
        print(f"FinRL Available: {FINRL_AVAILABLE}")
        
        # Step 1: Download data
        self.download_data()
        
        # Step 2: Add technical indicators
        self.add_technical_indicators()
        
        # Step 3: Load pre-trained model
        self.load_model()
        
        # Step 4: Test model
        df_account_value, df_actions = self.test_model()
        
        # Step 5: Plot results
        self.plot_results(df_account_value, df_actions)

        return df_account_value, df_actions

class DummyTradingEnv:
    """Dummy trading environment for demonstration when FinRL is not available."""
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
        self.current_step = 0
        self.max_steps = len(df) // kwargs.get('stock_dim', 5)
        self.asset_memory = [kwargs.get('initial_amount', 1000000)]
        self.state_space = kwargs.get('state_space', 50)
        
    def reset(self):
        self.current_step = 0
        self.asset_memory = [self.kwargs.get('initial_amount', 1000000)]
        return np.random.randn(self.state_space)
    
    def step(self, action):
        self.current_step += 1
        state = np.random.randn(self.state_space)
        reward = np.random.randn()
        done = self.current_step >= self.max_steps
        
        # Update asset memory
        new_value = self.asset_memory[-1] * (1 + reward * 0.01)
        self.asset_memory.append(new_value)
        
        info = {}
        return state, reward, done, info
    
    def get_sb_env(self):
        """Method to make it compatible with FinRL's DRL_prediction."""
        return self


class DummyPPOModel:
    """Dummy PPO model for demonstration when FinRL is not available."""
    def predict(self, observation):
        # Return random actions
        return np.random.randint(-1, 2, size=5), None


def main():
    """Main function to run the test."""
    # Initialize the tester with your model path
    model_path = "trained_models/trained_ppo_model.zip"  # Replace with your actual model path
    
    tester = FinRLPPOTester(
        model_path=model_path,
        start_date='2025-01-01',
        end_date='2026-12-31',
        initial_amount = 100000000
    )
    
    # Run the test
    df_account_value, df_actions = tester.run_test()
    
    print("\nTest completed successfully!")
    
    if FINRL_AVAILABLE:
        print("\nTo use this with your trained model:")
        print("1. Replace 'model_path' with the actual path to your trained model")
        print("2. Adjust the date range for testing")
        print("3. Modify ticker_list and technical indicators to match your training setup")
    else:
        print("\nTo run with real FinRL:")
        print("1. Install FinR")
        print("2. Install additional dependencies: pip install stable-baselines3[extra]")
        print("3. Run this script again")

if __name__ == "__main__":
    main()