# rate_forecaster.py
import pandas as pd
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import os


class RateForecaster:
    def __init__(self):
        """Initialize with historical data and trained model"""
        try:
            # Load historical rates
            self.historical_rates = self._load_historical_rates()

            # Store the last available historical date
            self.last_historical_date = self.historical_rates['date'].max()

            # Load or train model
            try:
                self.model = joblib.load('rate_forecaster.joblib')
            except:
                print("No pre-trained model found - training new model...")
                self.model = self._train_model(self.historical_rates)

        except Exception as e:
            print(f"Error initializing RateForecaster: {e}")
            # Fallback empty initialization
            self.historical_rates = pd.DataFrame(columns=['date', 'rate_per_kWh'])
            self.last_historical_date = None
            self.model = None

    def _load_historical_rates(self):
        """Load and validate historical rate data"""
        try:
            # Load from CSV (adjust path as needed)
            csv_path = os.path.join('assets', 'energy_rate.csv')
            rates_df = pd.read_csv(csv_path)

            # Validate columns
            required_cols = ['year', 'month', 'rate_per_kWh']
            if not all(col in rates_df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns. Found: {rates_df.columns.tolist()}")

            # Create datetime column
            rates_df['date'] = pd.to_datetime(rates_df[['year', 'month']].assign(day=1))
            return rates_df[['date', 'rate_per_kWh']].sort_values('date')

        except Exception as e:
            print(f"Error loading historical rates: {e}")
            return pd.DataFrame(columns=['date', 'rate_per_kWh'])

    def _train_model(self, historical_data):
        """Train forecasting model on historical data"""
        try:
            model = ExponentialSmoothing(
                historical_data.set_index('date').asfreq('MS')['rate_per_kWh'],
                trend='add',
                seasonal='add',
                seasonal_periods=12
            ).fit()

            # Save the trained model
            joblib.dump(model, 'rate_forecaster.joblib')
            return model

        except Exception as e:
            print(f"Error training model: {e}")
            return None

    def predict_rate(self, date):
        """
        Predict rate for a given date:
        - Returns exact historical rate if available
        - Forecasts future rate if beyond historical data
        """
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)

        # Check if we have exact historical data
        monthly_date = pd.to_datetime(f"{date.year}-{date.month}-01")
        historical_match = self.historical_rates[
            self.historical_rates['date'] == monthly_date
            ]

        if not historical_match.empty:
            return float(historical_match['rate_per_kWh'].iloc[0])

        # Forecast future rate if we have a trained model
        if self.model and self.last_historical_date:
            n_months = ((date.year - self.last_historical_date.year) * 12 +
                        (date.month - self.last_historical_date.month))

            if n_months > 0:
                forecast = self.model.forecast(n_months)
                return float(forecast.iloc[-1])

        # Fallback to last known rate if no forecast available
        if not self.historical_rates.empty:
            return float(self.historical_rates['rate_per_kWh'].iloc[-1])

        # Ultimate fallback
        return 11.0  # Default rate if all else fails