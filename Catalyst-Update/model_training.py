# model_training.py
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from data_preprocessing import load_and_preprocess_data
from rate_forecaster import RateForecaster


def train_models():
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Prepare features for kWh prediction
    X_kwh = df.drop(['energy_kWh', 'energy_cost_Php', 'rate_per_kWh'], axis=1)
    y_kwh = df['energy_kWh']

    # Prepare features for rate prediction (simpler features)
    X_rate = df[['year', 'month', 'day', 'day_of_week', 'is_weekend']]
    y_rate = df['rate_per_kWh']

    # Split data for both models
    X_kwh_train, X_kwh_test, y_kwh_train, y_kwh_test = train_test_split(
        X_kwh, y_kwh, test_size=0.2, random_state=42
    )

    X_rate_train, X_rate_test, y_rate_train, y_rate_test = train_test_split(
        X_rate, y_rate, test_size=0.2, random_state=42
    )

    # Train kWh prediction model
    kwh_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    kwh_model.fit(X_kwh_train, y_kwh_train)

    # Train rate prediction model
    rate_model = XGBRegressor(
        n_estimators=100,  # Can use simpler model for rates
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    rate_model.fit(X_rate_train, y_rate_train)

    historical_rates = load_historical_rates()
    rate_forecaster = RateForecaster()
    rate_forecaster.train(historical_rates)

    # Evaluate models
    kwh_pred = kwh_model.predict(X_kwh_test)
    rate_pred = rate_model.predict(X_rate_test)

    print("kWh Model Performance:")
    print(f"MAE: {mean_absolute_error(y_kwh_test, kwh_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_kwh_test, kwh_pred)):.2f}")

    print("\nRate Model Performance:")
    print(f"MAE: {mean_absolute_error(y_rate_test, rate_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_rate_test, rate_pred)):.2f}")

    # Save models
    joblib.dump(kwh_model, '../../../PD1_TEST/kwh_predictor.joblib')
    joblib.dump(rate_model, 'rate_predictor.joblib')

    return kwh_model, rate_model


if __name__ == "__main__":
    train_models()
