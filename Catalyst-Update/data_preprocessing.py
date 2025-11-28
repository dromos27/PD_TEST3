# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib


def load_and_preprocess_data():
    # Load energy consumption data
    consumption_df = pd.read_csv('assets/sched_database.csv')

    # Convert date and extract features (same as before)
    consumption_df['date'] = pd.to_datetime(consumption_df['date'], format='mixed')
    consumption_df['year'] = consumption_df['date'].dt.year
    consumption_df['month'] = consumption_df['date'].dt.month
    consumption_df['day'] = consumption_df['date'].dt.day
    consumption_df['day_of_week'] = consumption_df['date'].dt.dayofweek
    consumption_df['is_weekend'] = consumption_df['day_of_week'].isin([5, 6]).astype(int)

    # One-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(consumption_df[['room_type']]).toarray()
    encoded_df = pd.DataFrame(encoded_features,
                              columns=encoder.get_feature_names_out(['room_type']))

    # Combine with original data
    final_df = pd.concat([consumption_df.drop(['room_type', 'date'], axis=1), encoded_df], axis=1)

    # Ensure we have rate_per_kWh column (either from CSV or calculated)
    if 'rate_per_kWh' not in final_df.columns:
        # If no rate data, you might need to add default rates or another source
        final_df['rate_per_kWh'] = 11.0  # Default rate if not available

    # Calculate energy cost (for training data)
    final_df['energy_cost_Php'] = final_df['energy_kWh'] * final_df['rate_per_kWh']

    # Save the encoder
    joblib.dump(encoder, 'room_type_encoder.joblib')

    return final_df

# Add to data_preprocessing.py
def load_historical_rates():
    """Load historical energy rates with proper date handling"""
    try:
        rates_df = pd.read_csv('assets/energy_rate.csv')
        # Ensure we have the required columns
        if not all(col in rates_df.columns for col in ['year', 'month', 'rate_per_kWh']):
            raise ValueError("CSV file missing required columns")

        rates_df['date'] = pd.to_datetime(rates_df[['year', 'month']].assign(day=1))
        return rates_df[['date', 'rate_per_kWh']]
    except Exception as e:
        print(f"Error loading historical rates: {e}")
        # Return empty DataFrame with expected columns if file is missing
        return pd.DataFrame(columns=['date', 'rate_per_kWh'])