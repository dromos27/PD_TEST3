# energy_predictor.py
import pandas as pd
import joblib
from datetime import datetime
from rate_forecaster import RateForecaster

class EnergyPredictor:
    def __init__(self):
        self.kwh_model = joblib.load('../../../PD1_TEST/kwh_predictor.joblib')
        self.rate_forecaster = RateForecaster()  # Add rate forecaster
        self.encoder = joblib.load('room_type_encoder.joblib')
        self.room_multipliers = pd.read_csv('assets/room_multiplier.csv')
        self.recommendations = pd.read_csv('assets/recommendations.csv')  # Load recommendations

    def get_recommendations(self, kwh, cost):
        """Return recommendations based on energy and cost thresholds"""
        kwh_recs = self.recommendations[
            (self.recommendations['threshold_type'] == 'energy_kWh') &
            (self.recommendations['threshold_value'] <= kwh)
            ].sort_values('threshold_value', ascending=False)

        cost_recs = self.recommendations[
            (self.recommendations['threshold_type'] == 'energy_cost_Php') &
            (self.recommendations['threshold_value'] <= cost)
            ].sort_values('threshold_value', ascending=False)

        # Combine and deduplicate recommendations
        all_recs = pd.concat([kwh_recs, cost_recs]).drop_duplicates()

        # Return as list of strings
        return all_recs['recommendation'].tolist()

    def predict(self, year, month, day, room_number):
        try:
            input_date = pd.to_datetime(f"{year}-{month}-{day}")

            rate_pred = self.rate_forecaster.predict_rate(input_date)
            # print(f"Forecasted rate for {input_date.date()}: ₱{rate_pred:.2f}")  # Debug

            # Create input data with default room_type (will be overridden by multiplier)
            input_data = pd.DataFrame({
                'room': [room_number],
                'year': [year],
                'month': [month],
                'day': [day],
                'day_of_week': [input_date.weekday()],
                'is_weekend': [1 if input_date.weekday() in [5, 6] else 0],
                'room_type': ['lab']  # Default value, multiplier will adjust
            })

            # One-hot encoding
            encoded = self.encoder.transform(input_data[['room_type']]).toarray()
            encoded_df = pd.DataFrame(encoded,
                                      columns=self.encoder.get_feature_names_out(['room_type']))
            final_input = pd.concat([input_data.drop('room_type', axis=1), encoded_df], axis=1)

            # Get room-specific multiplier
            multiplier = float(self.room_multipliers[
                                   self.room_multipliers['room'] == room_number
                                   ]['multiplier'].values[0])

            # Predict kWh
            kwh_pred = max(0, round(float(self.kwh_model.predict(final_input)[0]), 2))

            # DON'T predict rate again - use the forecasted rate we already have
            # Calculate final cost with multiplier
            cost_pred = max(0, round(kwh_pred * rate_pred * multiplier, 2))

            recommendations = self.get_recommendations(kwh_pred, cost_pred)

            return {
                'energy_kWh': kwh_pred,
                'energy_rate': rate_pred,
                'energy_cost_Php': cost_pred,
                'recommendations': recommendations
            }


        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'energy_kWh': 0,
                'energy_rate': 0,
                'energy_cost_Php': 0
            }

    def predict_monthly(self, room_number, year=None):
        """
        Predict total monthly energy consumption by summing daily predictions.
        """
        if year is None:
            year = datetime.now().year

        monthly_totals = {}

        for month in range(1, 13):
            total_kwh = 0.0
            days_in_month = calendar.monthrange(year, month)[1]

            for day in range(1, days_in_month + 1):
                result = self.predict(year, month, day, room_number)
                total_kwh += result.get('energy_kWh', 0)

            month_name = datetime(year, month, 1).strftime("%B")
            monthly_totals[month_name] = round(total_kwh, 2)

        return monthly_totals

