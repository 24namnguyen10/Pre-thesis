"""
FinRL PPO Demo Implementation
This script demonstrates how to use a PPO-trained model from FinRL for stock trading.
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

class FinRLPPODemo:
    def __init__(self, 
                 start_date='2020-01-01',
                 end_date='2025-05-31',
                 train_split='2022-01-01',
                 initial_amount=1000000,
                 transaction_cost_pct=0.001,
                 tech_indicators=None):
        """
        Initialize the FinRL PPO demo.
        
        Args:
            start_date (str): Start date for data collection
            end_date (str): End date for data collection
            train_split (str): Date to split training and testing data
            initial_amount (int): Initial investment amount
            transaction_cost_pct (float): Transaction cost percentage
            tech_indicators (list): List of technical indicators to use
        """
        self.start_date = start_date
        self.end_date = end_date
        self.train_split = train_split
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
        self.ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        self.df = None
        self.train_df = None
        self.test_df = None
        self.model = None
        self.env_kwargs = {}
        
    def download_data(self):
        """Download stock data using Yahoo Finance."""
        print(f"Downloading data for {self.ticker_list}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        
        if not FINRL_AVAILABLE:
            # Create dummy data for demonstration
            print("Creating dummy data for demonstration...")
            return self._create_dummy_data()
        
        try:
            df = YahooDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                ticker_list=self.ticker_list
            ).fetch_data()
            
            print(f"Downloaded {len(df)} rows of data")
            self.df = df
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating dummy data for demonstration...")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy stock data for demonstration purposes."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Only weekdays
        
        data = []
        for ticker in self.ticker_list:
            # Generate realistic stock price movements
            np.random.seed(hash(ticker) % 1000)
            initial_price = np.random.uniform(100, 500)
            
            for i, date in enumerate(dates):
                # Random walk with slight upward trend
                if i == 0:
                    price = initial_price
                else:
                    price = data[-len(self.ticker_list)]['close'] * (1 + np.random.normal(0.001, 0.02))
                
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'tic': ticker,
                    'open': price * (1 + np.random.normal(0, 0.01)),
                    'high': price * (1 + abs(np.random.normal(0, 0.02))),
                    'low': price * (1 - abs(np.random.normal(0, 0.02))),
                    'close': price,
                    'volume': volume,
                    'adjcp': price
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        self.df = df
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
            self.df = processed_df
            print(f"Added {len(self.tech_indicators)} technical indicators")
            return processed_df
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return self._add_dummy_indicators()
    
    def _add_dummy_indicators(self):
        """Add dummy technical indicators for demonstration."""
        df = self.df.copy()
        
        for indicator in self.tech_indicators:
            df[indicator] = np.random.randn(len(df))
        
        self.df = df
        return df
    
    def prepare_data(self):
        """Prepare training and testing datasets."""
        print("Preparing training and testing datasets...")
        
        if not FINRL_AVAILABLE:
            # Simple split for dummy data
            train_mask = self.df['date'] < pd.to_datetime(self.train_split)
            self.train_df = self.df[train_mask].reset_index(drop=True)
            self.test_df = self.df[~train_mask].reset_index(drop=True)
        else:
            self.train_df = data_split(self.df, self.start_date, self.train_split)
            self.test_df = data_split(self.df, self.train_split, self.end_date)
        
        print(f"Training data: {len(self.train_df)} rows")
        print(f"Testing data: {len(self.test_df)} rows")
        
        return self.train_df, self.test_df
    
    def create_environment(self, df):
        """Create the trading environment."""
        self.env_kwargs = {
            'hmax': 100,
            'initial_amount': self.initial_amount,
            'num_stock_shares': [0] * len(self.ticker_list),
            'buy_cost_pct': [self.transaction_cost_pct] * len(self.ticker_list),
            'sell_cost_pct': [self.transaction_cost_pct] * len(self.ticker_list),
            'state_space': 1 + 2 * len(self.ticker_list) + len(self.tech_indicators) * len(self.ticker_list),
            'stock_dim': len(self.ticker_list),
            'tech_indicator_list': self.tech_indicators,
            'action_space': len(self.ticker_list),
            'reward_scaling': 1e-4
        }
        
        if FINRL_AVAILABLE:
            env = StockTradingEnv(df=df, **self.env_kwargs)
        else:
            # Create a dummy environment for demonstration
            env = DummyTradingEnv(df=df, **self.env_kwargs)
        
        return env
    
    def train_model(self):
        """Train the PPO model."""
        print("Training PPO model...")
        
        # Create training environment
        train_env = self.create_environment(self.train_df)
        
        if FINRL_AVAILABLE:
            # Use DRLAgent for training
            agent = DRLAgent(env=train_env)
            
            # Train PPO model
            self.model = agent.get_model("ppo", model_kwargs={
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0001,
                "batch_size": 128,
            })
            
            # Train for a reasonable number of timesteps
            trained_model = agent.train_model(
                model=self.model,
                tb_log_name='ppo',
                total_timesteps=50000
            )
            
            self.model = trained_model
        else:
            # Create dummy trained model for demonstration
            self.model = DummyPPOModel()
        
        print("Model training completed!")
        return self.model
    
    def test_model(self):
        """Test the trained model."""
        print("Testing trained model...")
        
        # Create testing environment
        test_env = self.create_environment(self.test_df)
        
        if FINRL_AVAILABLE:
            # Test the model
            df_account_value, df_actions = DRLAgent.DRL_prediction(
                model=self.model,
                environment=test_env
            )
        else:
            # Create dummy results for demonstration
            df_account_value, df_actions = self._create_dummy_results()
        
        print("Model testing completed!")
        return df_account_value, df_actions
    
    def _create_dummy_results(self):
        """Create dummy results for demonstration."""
        dates = pd.date_range(start=self.train_split, end=self.end_date, freq='D')
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
    
    def plot_results(self, df_account_value):
        """Plot the results."""
        print("Plotting results...")
        
        plt.figure(figsize=(12, 6))
        
        # Plot account value over time
        plt.subplot(1, 2, 1)
        plt.plot(df_account_value['date'], df_account_value['account_value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Account Value ($)')
        plt.xticks(rotation=45)
        
        # Calculate and plot daily returns
        plt.subplot(1, 2, 2)
        daily_returns = df_account_value['account_value'].pct_change().dropna()
        plt.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        total_return = (df_account_value['account_value'].iloc[-1] / 
                       df_account_value['account_value'].iloc[0] - 1) * 100
        
        print(f"\n=== Performance Summary ===")
        print(f"Initial Value: ${self.initial_amount:,.2f}")
        print(f"Final Value: ${df_account_value['account_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Average Daily Return: {daily_returns.mean():.4f}")
        print(f"Daily Return Std: {daily_returns.std():.4f}")
        print(f"Sharpe Ratio: {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.2f}")
    
    def run_demo(self):
        """Run the complete demo."""
        print("=== FinRL PPO Demo ===")
        print(f"FinRL Available: {FINRL_AVAILABLE}")
        
        # Step 1: Download data
        self.download_data()
        
        # Step 2: Add technical indicators
        self.add_technical_indicators()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train model
        self.train_model()
        
        # Step 5: Test model
        df_account_value, df_actions = self.test_model()
        
        # Step 6: Plot results
        self.plot_results(df_account_value)
        
        return df_account_value, df_actions


class DummyTradingEnv:
    """Dummy trading environment for demonstration when FinRL is not available."""
    def __init__(self, df, **kwargs):
        self.df = df
        self.kwargs = kwargs
    
    def reset(self):
        return np.random.randn(self.kwargs['state_space'])
    
    def step(self, action):
        state = np.random.randn(self.kwargs['state_space'])
        reward = np.random.randn()
        done = False
        info = {}
        return state, reward, done, info


class DummyPPOModel:
    """Dummy PPO model for demonstration when FinRL is not available."""
    def predict(self, observation):
        # Return random actions
        return np.random.randint(-1, 2, size=5), None


def main():
    """Main function to run the demo."""
    # Initialize the demo
    demo = FinRLPPODemo(
        start_date='2020-01-01',
        end_date='2023-12-31',
        train_split='2022-01-01',
        initial_amount=1000000
    )
    
    # Run the complete demo
    df_account_value, df_actions = demo.run_demo()
    
    print("\nDemo completed successfully!")
    
    if FINRL_AVAILABLE:
        print("\nTo use this with your own data:")
        print("1. Modify ticker_list to include your desired stocks")
        print("2. Adjust the date ranges as needed")
        print("3. Customize technical indicators")
        print("4. Tune PPO hyperparameters for better performance")
    else:
        print("\nTo run with real FinRL:")
        print("1. Install FinRL: pip install finrl")
        print("2. Install additional dependencies: pip install stable-baselines3[extra]")
        print("3. Run this script again")


if __name__ == "__main__":
    main()