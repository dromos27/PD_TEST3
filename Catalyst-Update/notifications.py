# notifications.py
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
from typing import Optional, List, Dict


class NotificationSystem:
    def __init__(self,
                 rules_path: str = 'assets/notifications.csv',
                 schedules_path: str = 'assets/room_schedules.csv'):
        """Initialize with notification rules and room schedules"""
        try:
            # Load notification rules
            self.notification_rules = pd.read_csv(rules_path)
            self.active_rules = self.notification_rules[self.notification_rules['is_active'] == 1]
            logging.info(f"Loaded {len(self.active_rules)} active notification rules")

            # Add occupancy rule if not exists
            if 'occupancy' not in self.active_rules['notification_type'].values:
                new_rule = {
                    'notification_type': 'occupancy',
                    'trigger_condition': 'unauthorized',
                    'message': '🚨 Unauthorized occupancy detected in {room}! Person present at {current_time} without scheduled booking.',
                    'priority': 'high',
                    'is_active': 1
                }
                self.notification_rules = pd.concat([self.notification_rules, pd.DataFrame([new_rule])],
                                                    ignore_index=True)
                self.active_rules = self.notification_rules[self.notification_rules['is_active'] == 1]

            # Load room schedules
            self.schedule_df = pd.read_csv(schedules_path)
            self.schedule_df['day'] = self.schedule_df['day'].str.lower()
            logging.info(f"Loaded schedules for {len(self.schedule_df['room'].unique())} rooms")

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            self.active_rules = pd.DataFrame()
            self.schedule_df = pd.DataFrame()

    def _parse_time_range(self, time_range: str) -> tuple:
        """Convert time range string (7:30-12:00) to time objects"""
        try:
            start_str, end_str = time_range.split('-')
            start_time = datetime.strptime(start_str.strip(), '%H:%M').time()
            end_time = datetime.strptime(end_str.strip(), '%H:%M').time()
            return start_time, end_time
        except Exception as e:
            logging.error(f"Error parsing time range {time_range}: {e}")
            return None, None

    def _get_current_schedule(self, room: str) -> List[Dict]:
        """Get today's schedule for a specific room"""
        try:
            today = datetime.now().strftime('%A').lower()
            room_schedules = self.schedule_df[
                (self.schedule_df['room'] == room) &
                (self.schedule_df['day'] == today)
                ]

            schedules = []
            for _, row in room_schedules.iterrows():
                start_time, end_time = self._parse_time_range(row['time_range'])
                if start_time and end_time:
                    schedules.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'value': row['value']
                    })
            return schedules
        except Exception as e:
            logging.error(f"Error getting schedule for room {room}: {e}")
            return []

    def _is_within_scheduled_time(self, room: str) -> bool:
        """Check if current time is within any scheduled time block"""
        current_time = datetime.now().time()
        schedules = self._get_current_schedule(room)

        for schedule in schedules:
            if schedule['start_time'] <= current_time <= schedule['end_time']:
                return True
        return False

    def _check_threshold_rules(self, value: float, rule_type: str, room: str, date: datetime) -> List[Dict]:
        """Check rules that compare against threshold values"""
        notifications = []
        for _, rule in self.active_rules[self.active_rules['notification_type'] == rule_type].iterrows():
            try:
                threshold = float(rule['trigger_condition'].replace('>', ''))
                if value > threshold:
                    notifications.append({
                        'message': rule['message'].format(
                            room=room,
                            value=round(value, 2),
                            date=date.strftime('%b %d')
                        ),
                        'priority': rule['priority']
                    })
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing rule {rule}: {e}")
        return notifications

    def _check_trend_rules(self, consumption_data: pd.DataFrame, room: str) -> List[Dict]:
        """Check for consumption trends (weekly comparisons)"""
        if len(consumption_data) < 14:
            return []

        current_week = consumption_data.iloc[-7:]['energy_kWh'].mean()
        previous_week = consumption_data.iloc[-14:-7]['energy_kWh'].mean()

        if previous_week <= 0:
            return []

        change_pct = ((current_week - previous_week) / previous_week) * 100
        abs_change = abs(round(change_pct, 1))

        if change_pct > 10:
            trend_type = 'increase_weekly'
            priority = 'medium'
        elif change_pct < -10:
            trend_type = 'decrease_weekly'
            priority = 'low'
        else:
            return []

        rule = self.active_rules[
            (self.active_rules['notification_type'] == 'trend') &
            (self.active_rules['trigger_condition'] == trend_type)
            ].iloc[0]

        return [{
            'message': rule['message'].format(room=room, value=abs_change),
            'priority': priority
        }]

    def _check_occupancy_rules(self, room: str) -> List[Dict]:
        """Check for unauthorized occupancy using camera and schedule"""
        try:
            # Get current occupancy status
            response = requests.get('http://localhost:5001/occupancy_status', timeout=2)
            occupancy_data = response.json()

            if occupancy_data.get('person_detected', False):
                current_time = datetime.now().strftime('%H:%M')

                if not self._is_within_scheduled_time(room):
                    rule = self.active_rules[
                        (self.active_rules['notification_type'] == 'occupancy') &
                        (self.active_rules['trigger_condition'] == 'unauthorized')
                        ].iloc[0]

                    return [{
                        'message': rule['message'].format(
                            room=room,
                            current_time=current_time
                        ),
                        'priority': rule['priority']
                    }]
        except Exception as e:
            logging.error(f"Error checking occupancy: {e}")
        return []

    def generate_notifications(self,
                               room: str,
                               consumption_data: pd.DataFrame,
                               cost_data: pd.DataFrame) -> List[Dict]:
        """Generate all notifications including occupancy checks"""
        if self.active_rules.empty:
            return []

        notifications = []
        today = datetime.now().date()

        try:
            # Original notification checks (consumption, cost, trends)
            if not consumption_data.empty and 'energy_kWh' in consumption_data.columns:
                latest_consumption = consumption_data['energy_kWh'].iloc[-1]
                notifications.extend(self._check_threshold_rules(
                    latest_consumption, 'consumption', room, today
                ))

            if not cost_data.empty and 'cost' in cost_data.columns:
                latest_cost = cost_data['cost'].iloc[-1]
                notifications.extend(self._check_threshold_rules(
                    latest_cost, 'cost', room, today
                ))

            if (not consumption_data.empty and
                    'energy_kWh' in consumption_data.columns and
                    'date' in consumption_data.columns):
                consumption_data = consumption_data.sort_values('date')
                notifications.extend(self._check_trend_rules(consumption_data, room))

            # New occupancy check with schedule integration
            notifications.extend(self._check_occupancy_rules(room))

        except Exception as e:
            logging.error(f"Error generating notifications: {e}")

        # Sort by priority (high first)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        return sorted(notifications, key=lambda x: priority_order[x['priority']])