# test_rate_forecaster.py
from rate_forecaster import RateForecaster
from datetime import datetime

forecaster = RateForecaster()

# Test predictions
dates_to_test = [
    '2023-01-01',  # Historical date (should return actual rate)
    '2023-06-01',  # Historical date
    '2025-06-01',  # Future date (should return forecast)
    '2025-12-01',  # Further future date
]

for date_str in dates_to_test:
    date = datetime.strptime(date_str, "%Y-%m-%d")
    rate = forecaster.predict_rate(date)
    print(f"{date_str}: ₱{rate:.2f}/kWh")