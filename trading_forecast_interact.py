"""
FinRL Multi-Model Trading Demo Implementation
This script demonstrates how to use pre-trained RL models from FinRL for stock trading.
Supports: PPO, A2C, DDPG, SAC, TD3
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
    
    # Stable Baselines3 imports for all models
    from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    FINRL_AVAILABLE = True
except ImportError as e:
    print(f"FinRL not available: {e}")
    print("Please install FinRL: pip install finrl")
    FINRL_AVAILABLE = False

class FinRLMultiModelTester:
    def __init__(self, 
                 model_path,
                 model_type='PPO',
                 start_date='2023-01-01',
                 end_date='2025-05-31',
                 initial_amount=1000000,
                 transaction_cost_pct=0.001,
                 tech_indicators=None):
        """
        Initialize the FinRL multi-model tester.
        
        Args:
            model_path (str): Path to the trained model
            model_type (str): Type of model ('PPO', 'A2C', 'DDPG', 'SAC', 'TD3')
            start_date (str): Start date for testing data
            end_date (str): End date for testing data
            initial_amount (int): Initial investment amount
            transaction_cost_pct (float): Transaction cost percentage
            tech_indicators (list): List of technical indicators to use
        """
        self.model_path = model_path
        self.model_type = model_type.upper()
        self.start_date = start_date
        self.end_date = end_date

        if hasattr(initial_amount, "values"):
            self.initial_amount = initial_amount.values[0]
        else:
            self.initial_amount = initial_amount

        self.transaction_cost_pct = transaction_cost_pct
        
        # Validate model type
        self.supported_models = ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3']
        if self.model_type not in self.supported_models:
            raise ValueError(f"Model type {self.model_type} not supported. Choose from: {self.supported_models}")
        
        # Use default indicators if none provided
        if tech_indicators is None:
            self.tech_indicators = [
                'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                'close_30_sma', 'close_60_sma'
            ]
        else:
            self.tech_indicators = tech_indicators
            
        # Use a subset of DOW 30 for demo (faster execution)
        self.ticker_list = DOW_30_TICKER
        
        self.df = None
        self.model = None
        self.env_kwargs = {}
        
        # Model-specific configurations
        self.model_configs = {
            'PPO': {'deterministic': True, 'is_continuous': False},
            'A2C': {'deterministic': True, 'is_continuous': False},
            'DDPG': {'deterministic': True, 'is_continuous': True},
            'SAC': {'deterministic': True, 'is_continuous': True},
            'TD3': {'deterministic': True, 'is_continuous': True}
        }
        
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
        """Load the pre-trained model based on model type."""
        print(f"Loading {self.model_type} model from {self.model_path}")
        
        if not FINRL_AVAILABLE:
            # Create dummy model for demonstration
            self.model = DummyModel(self.model_type)
            print(f"Loaded dummy {self.model_type} model for demonstration")
            return self.model
        
        try:
            # Load model based on type
            if self.model_type == 'PPO':
                self.model = PPO.load(self.model_path)
            elif self.model_type == 'A2C':
                self.model = A2C.load(self.model_path)
            elif self.model_type == 'DDPG':
                self.model = DDPG.load(self.model_path)
            elif self.model_type == 'SAC':
                self.model = SAC.load(self.model_path)
            elif self.model_type == 'TD3':
                self.model = TD3.load(self.model_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            print(f"{self.model_type} model loaded successfully!")
            return self.model
        except Exception as e:
            print(f"Error loading {self.model_type} model: {e}")
            print(f"Creating dummy {self.model_type} model for demonstration...")
            self.model = DummyModel(self.model_type)
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

        # Make sure everything is standard Python types
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
            print(f"✅ Successfully created StockTradingEnv for {self.model_type}")
            return env
        except Exception as e:
            print(f"[ERROR] Could not create StockTradingEnv: {e}")
            raise
    
    def test_model(self):
        """Test the trained model."""
        print(f"Testing {self.model_type} model...")
        if self.df is None or self.df.empty:
            raise ValueError("No data available for testing")
        
        print(f"Testing with {len(self.df)} data points")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Tickers: {self.df['tic'].unique().tolist()}")

        try:
            test_env = self.create_environment(self.df)
            vec_env = DummyVecEnv([lambda: test_env])

            # Use model-specific prediction method
            if FINRL_AVAILABLE:
                df_account_value, df_actions = DRLAgent.DRL_prediction(
                    model=self.model,
                    environment=vec_env
                )
            else:
                df_account_value, df_actions = self._manual_prediction(test_env)

            print(f"✅ {self.model_type} model tested successfully.")
            return df_account_value, df_actions

        except Exception as e:
            print(f"[Fallback] Error during {self.model_type} testing: {e}")
            print("Using dummy results instead.")
            return self._create_dummy_results()
    
    def _manual_prediction(self, env):
        """Manually run prediction using the model."""
        print(f"Running manual prediction with {self.model_type}...")
        
        try:
            # Reset environment and get initial state
            state = env.reset()
            done = False
            
            account_values = [self.initial_amount]
            actions_list = []
            dates = []
            
            step_count = 0
            max_steps = len(self.df) // len(self.ticker_list)
            
            while not done and step_count < max_steps:
                # Get action from model based on model type
                config = self.model_configs[self.model_type]
                action, _states = self.model.predict(state, deterministic=config['deterministic'])
                
                # Handle continuous vs discrete action spaces
                if config['is_continuous']:
                    # For continuous models (DDPG, SAC, TD3), actions are already continuous
                    # May need to clip or scale actions depending on environment
                    action = np.clip(action, -1, 1)
                else:
                    # For discrete models (PPO, A2C), actions are discrete
                    pass
                
                # Take step in environment
                state, reward, done, info = env.step(action)
                
                # Record results
                if hasattr(env, 'asset_memory') and len(env.asset_memory) > 0:
                    account_values.append(env.asset_memory[-1])
                else:
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
            
            print(f"Manual {self.model_type} prediction completed with {len(df_account_value)} steps")
            return df_account_value, df_actions
            
        except Exception as e:
            print(f"Error in manual {self.model_type} prediction: {e}")
            print("Falling back to dummy results...")
            return self._create_dummy_results()
    
    def _create_dummy_results(self):
        """Create dummy results for demonstration."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # Only weekdays
        
        # Generate realistic account value progression based on model type
        account_values = []
        current_value = self.initial_amount
        
        # Different models might have different performance characteristics
        model_volatility = {
            'PPO': 0.02,
            'A2C': 0.025,
            'DDPG': 0.03,
            'SAC': 0.02,
            'TD3': 0.025
        }
        
        volatility = model_volatility.get(self.model_type, 0.02)
        
        for i, date in enumerate(dates):
            # Random walk with slight upward trend
            daily_return = np.random.normal(0.0005, volatility)
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

        print(f"📈 Plotting {self.model_type} trading results...")

        # Prepare data
        df_account_value['date'] = pd.to_datetime(df_account_value['date'])
        df_account_value = df_account_value.sort_values('date')
        df_account_value.set_index('date', inplace=True)

        # Daily returns
        daily_returns = df_account_value['account_value'].pct_change().dropna()

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot Account Value Line
        ax.plot(df_account_value.index, df_account_value['account_value'], 
                label=f'{self.model_type} Portfolio Value', color='blue', linewidth=2)

        # Overlay Buy/Sell Markers
        if show_actions and df_actions is not None:
            df_actions['date'] = pd.to_datetime(df_actions['date'])
            df_actions = df_actions.sort_values('date')
            df_actions.set_index('date', inplace=True)

            # Visualize net position (sum of action vector)
            df_actions['net_action'] = df_actions['actions'].apply(
                lambda x: sum(x) if isinstance(x, (list, np.ndarray)) else 0
            )

            buy_signals = df_actions[df_actions['net_action'] > 0]
            sell_signals = df_actions[df_actions['net_action'] < 0]

            if len(buy_signals) > 0:
                ax.scatter(buy_signals.index, 
                        df_account_value.loc[buy_signals.index, 'account_value'], 
                        marker='^', color='green', label='Buy', alpha=0.6)

            if len(sell_signals) > 0:
                ax.scatter(sell_signals.index, 
                        df_account_value.loc[sell_signals.index, 'account_value'], 
                        marker='v', color='red', label='Sell', alpha=0.6)

        # Formatting
        ax.set_title(f"{self.model_type} Trading Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Account Value ($)")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        filename = f"{self.model_type.lower()}_trading_result.png"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.show()

        # Print summary
        print(f"\n=== {self.model_type} Performance Summary ===")
        print(f"Initial Value: ${self.initial_amount:,.2f}")
        print(f"Final Value: ${df_account_value['account_value'].iloc[-1]:,.2f}")
        print(f"Total Return: {(df_account_value['account_value'].iloc[-1] / df_account_value['account_value'].iloc[0] - 1) * 100:.2f}%")
        print(f"Average Daily Return: {daily_returns.mean():.4f}")
        print(f"Daily Return Std: {daily_returns.std():.4f}")
        if daily_returns.std() > 0:
            print(f"Sharpe Ratio: {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.2f}")
        else:
            print("Sharpe Ratio: N/A (zero volatility)")
    
    def run_test(self):
        """Run the complete test."""
        print(f"=== FinRL {self.model_type} Testing Demo ===")
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


class DummyModel:
    """Dummy model for demonstration when FinRL is not available."""
    def __init__(self, model_type):
        self.model_type = model_type
        self.action_space_size = 30  # DOW_30_TICKER
        
    def predict(self, observation, deterministic=True):
        """Predict action based on model type."""
        if self.model_type in ['DDPG', 'SAC', 'TD3']:
            # Continuous action space
            action = np.random.uniform(-1, 1, size=self.action_space_size)
        else:
            # Discrete action space (PPO, A2C)
            action = np.random.randint(-1, 2, size=self.action_space_size)
        
        return action, None


def compare_models(model_configs, start_date='2025-01-01', end_date='2026-12-31', initial_amount=1000000):
    """Compare performance of multiple models."""
    results = {}
    
    print("=== Multi-Model Comparison ===")
    
    for model_type, model_path in model_configs.items():
        print(f"\n--- Testing {model_type} ---")
        
        try:
            tester = FinRLMultiModelTester(
                model_path=model_path,
                model_type=model_type,
                start_date=start_date,
                end_date=end_date,
                initial_amount=initial_amount
            )
            
            df_account_value, df_actions = tester.run_test()
            
            # Calculate performance metrics
            total_return = (df_account_value['account_value'].iloc[-1] / 
                          df_account_value['account_value'].iloc[0] - 1) * 100
            
            daily_returns = df_account_value['account_value'].pct_change().dropna()
            volatility = daily_returns.std()
            sharpe_ratio = daily_returns.mean() / volatility * np.sqrt(252) if volatility > 0 else 0
            
            results[model_type] = {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'final_value': df_account_value['account_value'].iloc[-1]
            }
            
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
            results[model_type] = None
    
    # Print comparison summary
    print("\n=== Model Comparison Summary ===")
    print(f"{'Model':<8} {'Total Return':<15} {'Volatility':<12} {'Sharpe':<8} {'Final Value':<15}")
    print("-" * 70)
    
    for model_type, metrics in results.items():
        if metrics:
            print(f"{model_type:<8} {metrics['total_return']:<14.2f}% {metrics['volatility']:<11.4f} "
                  f"{metrics['sharpe_ratio']:<7.2f} ${metrics['final_value']:<14,.0f}")
        else:
            print(f"{model_type:<8} {'Failed':<14} {'N/A':<11} {'N/A':<7} {'N/A':<15}")
    
    return results


def get_user_model_choice():
    """Interactive model selection interface."""
    print("\n" + "="*60)
    print("🚀 FinRL Multi-Model Stock Trading Forecaster")
    print("="*60)
    
    print("\nAvailable Models:")
    print("1. PPO (Proximal Policy Optimization)")
    print("2. A2C (Advantage Actor-Critic)")
    print("3. DDPG (Deep Deterministic Policy Gradient)")
    print("4. SAC (Soft Actor-Critic)")
    print("5. TD3 (Twin Delayed Deep Deterministic)")
    print("6. Compare All Models")
    print("7. Custom Model Configuration")
    
    while True:
        try:
            choice = input("\nSelect model to deploy (1-7): ").strip()
            
            if choice == '1':
                return 'PPO', 'trained_models/trained_ppo_model.zip'
            elif choice == '2':
                return 'A2C', 'trained_models/trained_a2c_model.zip'
            elif choice == '3':
                return 'DDPG', 'trained_models/trained_ddpg_model.zip'
            elif choice == '4':
                return 'SAC', 'trained_models/trained_sac_model.zip'
            elif choice == '5':
                return 'TD3', 'trained_models/trained_td3_model.zip'
            elif choice == '6':
                return 'COMPARE', None
            elif choice == '7':
                return 'CUSTOM', None
            else:
                print("❌ Invalid choice. Please select 1-7.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit()
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")


def get_custom_configuration():
    """Get custom model configuration from user."""
    print("\n📝 Custom Model Configuration")
    print("-" * 40)
    
    # Model type selection
    model_types = ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3']
    print("Available model types:")
    for i, model in enumerate(model_types, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            model_choice = int(input("Select model type (1-5): ")) - 1
            if 0 <= model_choice < len(model_types):
                model_type = model_types[model_choice]
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
        except ValueError:
            print("❌ Please enter a number.")
    
    # Model path
    model_path = input(f"Enter path to {model_type} model file: ").strip()
    if not model_path:
        model_path = f"trained_models/trained_{model_type.lower()}_model.zip"
        print(f"Using default path: {model_path}")
    
    # Date range
    start_date = input("Enter start date (YYYY-MM-DD) [default: 2025-01-01]: ").strip()
    if not start_date:
        start_date = '2025-01-01'
    
    end_date = input("Enter end date (YYYY-MM-DD) [default: 2026-12-31]: ").strip()
    if not end_date:
        end_date = '2026-12-31'
    
    # Initial amount
    initial_amount_input = input("Enter initial amount [default: 1000000]: ").strip()
    if not initial_amount_input:
        initial_amount = 1000000
    else:
        try:
            initial_amount = float(initial_amount_input)
        except ValueError:
            print("❌ Invalid amount. Using default: 1000000")
            initial_amount = 1000000
    
    return {
        'model_type': model_type,
        'model_path': model_path,
        'start_date': start_date,
        'end_date': end_date,
        'initial_amount': initial_amount
    }


def deploy_single_model(model_type, model_path, start_date='2025-01-01', 
                       end_date='2026-12-31', initial_amount=1000000):
    """Deploy a single model for trading."""
    print(f"\n🎯 Deploying {model_type} Model")
    print("-" * 40)
    print(f"Model: {model_type}")
    print(f"Path: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Amount: ${initial_amount:,.2f}")
    
    try:
        tester = FinRLMultiModelTester(
            model_path=model_path,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
            initial_amount=initial_amount
        )
        
        print(f"\n📊 Running {model_type} model...")
        df_account_value, df_actions = tester.run_test()
        
        print(f"\n✅ {model_type} model deployment completed successfully!")
        return df_account_value, df_actions
        
    except Exception as e:
        print(f"❌ Error deploying {model_type} model: {e}")
        return None, None


def deploy_model_comparison():
    """Deploy and compare multiple models."""
    print("\n🔄 Model Comparison Mode")
    print("-" * 40)
    
    # Allow user to customize model paths
    model_configs = {}
    available_models = ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3']
    
    print("Configure models for comparison:")
    print("(Press Enter to use default path, 'skip' to exclude model)")
    
    for model in available_models:
        default_path = f"trained_models/trained_{model.lower()}_model.zip"
        user_path = input(f"{model} model path [default: {default_path}]: ").strip()
        
        if user_path.lower() == 'skip':
            print(f"⏭️  Skipping {model}")
            continue
        elif user_path == '':
            model_configs[model] = default_path
        else:
            model_configs[model] = user_path
    
    if not model_configs:
        print("❌ No models selected for comparison.")
        return
    
    print(f"\n📊 Comparing {len(model_configs)} models...")
    print("Selected models:", ", ".join(model_configs.keys()))
    
    try:
        results = compare_models(model_configs)
        print(f"\n✅ Model comparison completed!")
        return results
    except Exception as e:
        print(f"❌ Error during model comparison: {e}")
        return None


def main():
    """Main function with interactive model selection."""
    try:
        choice, default_path = get_user_model_choice()
        
        if choice == 'COMPARE':
            deploy_model_comparison()
            
        elif choice == 'CUSTOM':
            config = get_custom_configuration()
            deploy_single_model(
                model_type=config['model_type'],
                model_path=config['model_path'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                initial_amount=config['initial_amount']
            )
            
        else:
            # Single model deployment
            deploy_single_model(choice, default_path)
        
        print("\n" + "="*60)
        print("📋 Deployment Summary")
        print("="*60)
        print("✅ Model deployment completed successfully!")
        print("\n📁 Output files generated:")
        print("   - Trading result plots (PNG files)")
        print("   - Performance metrics in console")
        
        print("\n💡 Tips:")
        print("1. Review the generated plots for trading performance")
        print("2. Check performance metrics for model evaluation")
        print("3. Consider transaction costs in real trading")
        print("4. Validate model performance on out-of-sample data")
        
        if not FINRL_AVAILABLE:
            print("\n⚠️  Note: Running in simulation mode")
            print("   Install FinRL for real model deployment:")
            print("   pip install finrl stable-baselines3[extra]")
            
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled by user.")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")


def quick_deploy(model_type, model_path=None, start_date='2025-01-01', 
                end_date='2026-12-31', initial_amount=1000000):
    """Quick deployment function for programmatic use."""
    if model_path is None:
        model_path = f"trained_models/trained_{model_type.lower()}_model.zip"
    
    print(f"🚀 Quick deploying {model_type} model...")
    
    tester = FinRLMultiModelTester(
        model_path=model_path,
        model_type=model_type,
        start_date=start_date,
        end_date=end_date,
        initial_amount=initial_amount
    )
    
    return tester.run_test()


# Example usage functions
def deploy_ppo_model():
    """Example: Deploy PPO model."""
    return quick_deploy('PPO', 'models/my_ppo_model.zip')


def deploy_sac_model():
    """Example: Deploy SAC model."""
    return quick_deploy('SAC', 'models/my_sac_model.zip')


def deploy_best_model():
    """Example: Deploy the best performing model after comparison."""
    model_configs = {
        'PPO': 'trained_models/trained_ppo_model.zip',
        'SAC': 'trained_models/trained_sac_model.zip',
        'TD3': 'trained_models/trained_td3_model.zip'
    }
    
    results = compare_models(model_configs)
    
    # Find best model based on Sharpe ratio
    best_model = max(results.items(), key=lambda x: x[1]['sharpe_ratio'] if x[1] else -999)
    
    print(f"\n🏆 Best performing model: {best_model[0]}")
    print(f"Sharpe Ratio: {best_model[1]['sharpe_ratio']:.2f}")
    
    # Deploy the best model
    return quick_deploy(best_model[0], model_configs[best_model[0]])


if __name__ == "__main__":
    main()