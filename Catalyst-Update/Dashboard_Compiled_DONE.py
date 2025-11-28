import pandas as pd
import plotly.express as px
import requests
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from energy_predictor import EnergyPredictor
from flask_caching import Cache
from datetime import date, datetime, timedelta
from notifications import NotificationSystem
import dash
from dash_extensions import WebSocket
import base64
import io
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dash.exceptions import PreventUpdate
import numpy as np
from rate_forecaster import RateForecaster
from data_preprocessing import load_historical_rates

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


# ESP32 Configuration (update with your ESP32's IP)
ESP32_IP = "192.168.1.103"  # Change to your ESP32's IP
ESP32_BASE_URL = f"http://192.168.1.103/"

# Set up caching (replaces @st.cache_resource)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

#Schedule
def load_schedules():
    """Load schedules with better error handling and header validation"""
    schedules = {}
    try:
        with open('assets/room_schedules.csv', mode='r', encoding='utf-8-sig') as file:  # utf-8-sig handles BOM
            # Read first line to verify headers
            first_line = file.readline().strip()
            if not first_line.startswith("room,day,time_range,value"):
                raise ValueError("CSV headers don't match expected format")

            # Reset file pointer
            file.seek(0)

            reader = csv.DictReader(file)
            for row_num, row in enumerate(reader, 1):
                try:
                    room = row.get('room', '').strip()
                    day = row.get('day', '').strip()
                    time_range = row.get('time_range', '').strip()
                    value = row.get('value', '').strip()

                    if not all([room, day, time_range, value]):
                        print(f"Warning: Skipping row {row_num} - missing data")
                        continue

                    value = float(value)

                    if room not in schedules:
                        schedules[room] = {}
                    if day not in schedules[room]:
                        schedules[room][day] = {}

                    schedules[room][day][time_range] = value

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row {row_num} - {str(e)}")
                    continue

    except FileNotFoundError:
        print("Error: CSV file not found at 'assets/room_schedules.csv'")
    except Exception as e:
        print(f"Error loading schedules: {str(e)}")

    return schedules


def generate_time_slots():
    """Generate time slots from 7:30 to 21:30 in 30-minute increments"""
    time_slots = []
    current_hour = 7
    current_minute = 30
    for _ in range(29):
        time_slots.append(f"{current_hour}:{current_minute:02d}")
        current_minute += 30
        if current_minute >= 60:
            current_hour += 1
            current_minute = 0
    return time_slots


def create_schedule_matrix(time_slots, days, room_schedule):
    """Create a schedule matrix from the room schedule data"""
    schedule_data = np.zeros((len(time_slots), len(days)))
    for day_idx, day in enumerate(days):
        if day in room_schedule:
            for time_range, value in room_schedule[day].items():
                start_time, end_time = time_range.split('-')
                try:
                    start_idx = time_slots.index(start_time)
                    end_idx = time_slots.index(end_time)
                    for i in range(start_idx, end_idx):
                        schedule_data[i][day_idx] = value
                except ValueError:
                    continue
    return schedule_data


def generate_schedule_image(room_number):
    """Generate and return base64 encoded image"""
    if room_number not in schedules:
        raise ValueError(f"No schedule found for room {room_number}")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_slots = generate_time_slots()
    schedule_data = create_schedule_matrix(time_slots, days, schedules[room_number])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.invert_yaxis()

    # Create colored rectangles
    for i in range(len(time_slots)):
        for j in range(len(days)):
            if schedule_data[i][j] == 0.5:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#FF9999', edgecolor='white', lw=1))

    # Configure plot
    ax.set_xticks(np.arange(len(days)) + 0.5)
    ax.set_xticklabels(days, ha='center')
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(time_slots)))
    ax.set_yticklabels(time_slots)
    plt.title(f"ROOM {room_number} Weekly Schedule", pad=20, fontsize=14, y=1.05)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

schedules = load_schedules()  # Global variable
available_rooms = list(schedules.keys())  # For dropdown

# Initialize predictor with caching
@cache.memoize()
def load_predictor():
    return EnergyPredictor()

predictor = load_predictor()

# Initialize notification system with caching
@cache.memoize()
def load_notification_system():
    return NotificationSystem()

notifier = load_notification_system()

# df_rooms = pd.read_csv("assets/sched_database.csv")
# available_rooms = sorted(df_rooms['room'].astype(str).unique()) # sched_database.csv

current_date = datetime.now().date()

# Constants
COST_PER_KWH = 11  # Philippine Peso

# Load and preprocess updated data
df = pd.read_csv("assets/sample_data.csv")
available_rooms_tab1 = sorted(df['Room'].astype(str).unique()) # sample_data.csv

# Normalize column names to lowercase for consistency
df.columns = [col.lower().replace(' ', '_').replace('(kwh)', 'kwh') for col in df.columns]
# Now columns include: 'date', 'time_hour', 'room', 'energy_kwh'

# Combine 'date' and 'time_hour' into datetime
df['timestamp'] = pd.to_datetime(df['date'].str.strip() + ' ' + df['time_hour'].str.strip(),
                                 format='%d/%m/%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['timestamp'])

# Extract date as datetime.date
df['date'] = df['timestamp'].dt.date

# energy_kwh is already a column but confirm it's float
df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce').fillna(0)

# Precompute daily energy & cost per room
daily_room_df = df.groupby(['room', 'date'])['energy_kwh'].sum().reset_index()
daily_room_df['cost'] = daily_room_df['energy_kwh'] * COST_PER_KWH

# Precompute average daily energy consumption per room (over all dates)
# avg_daily_per_room = daily_room_df.groupby('room')['energy_kwh'].mean().to_dict()

# Set default date and room
default_date_tab1 = daily_room_df['date'].max()
default_room_tab1 = available_rooms_tab1[0]


# Dash app
# app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    html.Div([
        dcc.Tabs(
            id='tabs',
            value='tab-1',
            children=[
                dcc.Tab(label='Overview', value='tab-1', className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Predictive Analytics', value='tab-2', className='custom-tab',
                        selected_className='custom-tab--selected', id='predictive-tab'),
                dcc.Tab(label='Camera', value='tab-3', className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Room Schedule', value='tab-4', className='custom-tab',
                        selected_className='custom-tab--selected')
            ],
            className='custom-tabs',
            vertical=True
        ),
    ], className='tabs-container'),

    # THIS div now stretches and centers its content
    html.Div(id='tab-content', style={
        'flex': '1',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        # 'backgroundColor': '#121212'
    })
], style={'display': 'flex', 'height': '100vh', 'overflow': 'hidden'})


# Callback to update content based on selected tab
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-1':
        return html.Div([
        html.H2("Overview", style={'color': 'white', 'marginBottom': '0px'}),

        # Date and Room Selector
        html.Div([
            html.Div([
                html.Label("Select a date:", style={'color': 'white'}),
                dcc.DatePickerSingle(
                    id='tab1-date-picker',
                    min_date_allowed=daily_room_df['date'].min(),
                    max_date_allowed=daily_room_df['date'].max(),
                    date=default_date_tab1,
                    display_format='MM/DD/YYYY',
                    style={'color': 'black'}
                )
            ], style={'flex': '1', 'paddingRight': '10px'}),

            html.Div([
                html.Label("Select Room:", style={'color': 'white', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='tab1-room-dropdown',
                    options=[{'label': str(r), 'value': r} for r in available_rooms_tab1],
                    value=default_room_tab1,
                    clearable=False,
                    style={'color': 'black'}
                )
            ], style={'width': '200px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'padding': '10px 0', 'marginBottom': '20px'}),

        # Main content (Graph and Metrics)
        html.Div([
            # Left - Graph
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(
                            id='minute-level-graph',
                            figure=px.line(
                                title='Select a date to view minute-level consumption'
                            ).update_layout(
                                plot_bgcolor='#1e1e3f',
                                paper_bgcolor='#1e1e3f',
                                font=dict(color='white'),
                                title_font=dict(color='white')
                            ))
                    ])
                ], style={
                    'backgroundColor': '#1e1e3f',
                    'borderRadius': '10px',
                    'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                    'border': '1px solid #00FFFF',
                    'height': '100%'
                })
            ], style={'flex': '1', 'paddingRight': '20px', 'width': '50%'}),

            # Right - Split into two parts
            html.Div([
                # First half (top) - Metrics cards
                html.Div([
                    # First row
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Energy Consumption", style={'color': 'white', 'textAlign': 'center'}),
                                html.H3(id='total-energy', style={
                                    'color': 'cyan', 'fontSize': '2.5rem',
                                    'margin': '0', 'textAlign': 'center'
                                })
                            ], style={
                                'display': 'flex', 'flexDirection': 'column',
                                'justifyContent': 'center', 'height': '100%'
                            })
                        ], style={
                            'backgroundColor': '#1e1e3f', 'borderRadius': '10px',
                            'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                            'border': '1px solid #00FFFF', 'width': '100%', 'height': '150px'
                        }),

                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Total Cost", style={'color': 'white', 'textAlign': 'center'}),
                                html.H3(id='total-cost', style={
                                    'color': 'cyan', 'fontSize': '2.5rem',
                                    'margin': '0', 'textAlign': 'center'
                                })
                            ], style={
                                'display': 'flex', 'flexDirection': 'column',
                                'justifyContent': 'center', 'height': '100%'
                            })
                        ], style={
                            'backgroundColor': '#1e1e3f', 'borderRadius': '10px',
                            'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                            'border': '1px solid #00FFFF', 'width': '100%', 'height': '150px'
                        }),
                    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

                    # Second row
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Average Hourly Consumption", style={'color': 'white', 'textAlign': 'center'}),
                                html.H3(id='avg-hourly', style={
                                    'color': 'cyan', 'fontSize': '2.5rem',
                                    'margin': '0', 'textAlign': 'center'
                                })
                            ], style={
                                'display': 'flex', 'flexDirection': 'column',
                                'justifyContent': 'center', 'height': '100%'
                            })
                        ], style={
                            'backgroundColor': '#1e1e3f', 'borderRadius': '10px',
                            'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                            'border': '1px solid #00FFFF', 'width': '100%', 'height': '150px'
                        }),

                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Live Consumption", style={'color': 'white', 'textAlign': 'center'}),
                                html.H3(id='live-consumption', style={
                                    'color': 'cyan', 'fontSize': '2.5rem',
                                    'margin': '0', 'textAlign': 'center'
                                }, children="0.00 kWh"),
                                dcc.Interval(id='live-update-interval', interval=5 * 1000, n_intervals=0)
                            ], style={
                                'display': 'flex', 'flexDirection': 'column',
                                'justifyContent': 'center', 'alignItems': 'center',
                                'height': '100%'
                            })
                        ], style={
                            'backgroundColor': '#1e1e3f', 'borderRadius': '10px',
                            'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                            'border': '1px solid #00FFFF', 'width': '100%', 'height': '150px'
                        }),
                    ], style={'display': 'flex', 'gap': '20px'})
                ], style={'marginBottom': '20px'}),

                # Second half (bottom) - Live consumption and notifications
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔔 Notifications", style={'color': 'white'}),
                            html.Div(id='notification-container', children=[]),
                            dcc.Interval(id='notification-update-interval', interval=60 * 1000, n_intervals=0)
                        ])
                    ], style={
                        'backgroundColor': '#1e1e3f',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'border': '1px solid #00FFFF',
                        'height': '143px'
                    })
                ], style={'flex': '1'})
            ], style={'flex': '1', 'width': '50%', 'display': 'flex', 'flexDirection': 'column'})
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'stretch',
            'alignItems': 'flex-start',
            'width': '1300px',
            'padding': '0 40px'
        }),
        dcc.Store(id='live-consumption-store', data="0.00 kWh"),
     # Add this with your other dcc.Store components
        dcc.Store(id='relay-state-store', data={'lights_on': False}),
    ])

    elif tab == 'tab-2':
        current_date = datetime.now().date()
        return html.Div([
            html.H2("Predictive Analytics", style={'color': 'white', 'marginBottom': '20px'}),
            html.Div([
                # Left Column: Parameters and Recommendations
                html.Div([
                    # Prediction Parameters Card
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Prediction Parameters",
                                    style={'color': 'white', 'marginBottom': '20px'}),

                            # Row for date and room selectors
                            html.Div([
                                # Date selector (left)
                                html.Div([
                                    html.Label("Date",
                                               style={'color': 'white', 'display': 'block', 'marginBottom': '5px'}),
                                    dcc.DatePickerSingle(
                                        id='date-picker',
                                        min_date_allowed=date(2023, 1, 1),
                                        max_date_allowed=date(2025, 12, 31),
                                        initial_visible_month=current_date,
                                        date=current_date,
                                        style={
                                            'backgroundColor': '#1e1e3f',
                                            'border': '1px solid #00FFFF',
                                            'color': 'white',
                                            'width': '100%'
                                        }
                                    )
                                ], style={'flex': '1', 'marginRight': '10px'}),

                                # Room selector (right)
                                html.Div([
                                    html.Label("Room",
                                               style={'color': 'white', 'display': 'block', 'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='room-dropdown',
                                        options=[{'label': str(r), 'value': r} for r in available_rooms],
                                        value=available_rooms[0],
                                        clearable=False,
                                        style={
                                            'backgroundColor': 'white',
                                            'color': 'black',
                                            'border': '1px solid #00FFFF',
                                        }
                                    )
                                ], style={'flex': '1', 'marginLeft': '10px'})
                            ], style={
                                'display': 'flex',
                                'justifyContent': 'center',
                                'marginBottom': '30px',
                                'gap': '20px'
                            }),

                            # Predict button (centered below)
                            dbc.Button(
                                "Predict Energy Usage",
                                id='predict-button',
                                color="primary",
                                style={
                                    'backgroundColor': '#00FFFF',
                                    'color': '#252A4B',
                                    'border': 'none',
                                    'padding': '5px 20px',
                                    'fontWeight': 'bold',
                                    'boxShadow': '0 0 10px rgba(0,255,255,0.5)',
                                    'width': '100%',
                                    'maxWidth': '300px',
                                    'margin': '0 auto'
                                }
                            )
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'height': '100%',
                            'padding': '20px'
                        })
                    ], style={
                        'backgroundColor': '#252A4B',
                        'border': '1px solid #00FFFF',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'marginBottom': '20px',
                        'height': '320px'
                    }),

                    # Recommendations Card
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("💡 Recommendations", style={'color': 'white'}),
                            html.Ul(id='recommendations-list', children=[
                                html.Li("Enter parameters and click 'Predict' to get recommendations",
                                        style={'color': 'lightgray'})
                            ], style={'paddingLeft': '20px'})
                        ])
                    ], style={
                        'backgroundColor': '#1e1e3f',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'border': '1px solid #00FFFF',
                        'height': '150px'  # Adjusted height to match parameters card
                    })
                ], style={'flex': '1', 'paddingRight': '20px', 'width': '50%'}),
                # Right Column: KPI Cards
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predicted Energy Consumption",
                                    style={'color': 'white', 'textAlign': 'center', 'marginBottom': '15px'}),
                            html.H3(id='kwh-result', children="0 kWh",
                                    style={'color': 'cyan', 'textAlign': 'center', 'fontSize': '2.5rem',
                                           'margin': '0'})
                        ])
                    ], style={
                        'backgroundColor': '#1e1e3f',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'border': '1px solid #00FFFF',
                        'height': '150px',  # Adjusted height
                        'marginBottom': '20px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    }),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predicted Electricity Rate",
                                    style={'color': 'white', 'textAlign': 'center', 'marginBottom': '15px'}),

                            # Add this to force updates (if needed)
                            dcc.Store(id='rate-store', data={'rate': 0}),

                            html.H3(
                                id='rate-result',
                                children="₱0.00/kWh",
                                style={
                                    'color': 'cyan',
                                    'textAlign': 'center',
                                    'fontSize': '2.5rem',
                                    'margin': '0',
                                    'fontWeight': 'bold'  # Make it stand out
                                }
                            )
                        ])
                    ], style={
                        'backgroundColor': '#1e1e3f',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'border': '1px solid #00FFFF',
                        'height': '150px',
                        'marginBottom': '20px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    }),

                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Predicted Energy Cost",
                                    style={'color': 'white', 'textAlign': 'center', 'marginBottom': '15px'}),
                            html.H3(id='cost-result', children="₱0.00",
                                    style={'color': 'cyan', 'textAlign': 'center', 'fontSize': '2.5rem',
                                           'margin': '0'})
                        ])
                    ], style={
                        'backgroundColor': '#1e1e3f',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 15px rgba(0,255,255,0.2)',
                        'border': '1px solid #00FFFF',
                        'height': '150px',  # Adjusted height
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    })
                ], style={'flex': '1', 'width': '50%'})
            ], style={
                'display': 'flex',
                'width': '1650px',
                'justifyContent': 'stretch',
                'alignItems': 'flex-start',
                'padding': '0 40px'
            })
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'stretch',
            'alignItems': 'flex-start',
            'width': '1300px',
            'padding': '0 40px',
        })

    elif tab == 'tab-3':
        return html.Div([
            html.H3("Live Camera Feed", style={'color': 'white', 'textAlign': 'center'}),
            html.Div([
                WebSocket(id="ws", url="ws://localhost:5001/ws"),
                html.Img(id="live-feed", style={
                    'width': '800px',
                    'height': '600px',
                    'border': '3px solid #00FFFF',
                    'display': 'block',
                    'margin': '0 auto'
                }),
                dcc.Interval(id="init-trigger", interval=200, n_intervals=0, max_intervals=1)
            ], style={
                'textAlign': 'center'
            })
        ], style={
            'width': '1300px',
        })

    elif tab == 'tab-4':  # Room Schedule & Device Control
        return html.Div([
            html.Div([
                html.Div([
                    html.H2("Room Schedule & Device Control", className='tab4-header'),
                    html.Div([
                        # Left Column - Room Selection and Controls
                        html.Div([
                            # Room Selection
                            html.Div([
                                html.Label("Select Room:", className='control-label'),
                                dcc.Dropdown(
                                    id='schedule-room-dropdown',
                                    options=[{'label': str(r), 'value': r} for r in available_rooms],
                                    value=available_rooms[0],
                                    clearable=False,
                                    className='room-dropdown'
                                )
                            ], className='dropdown-container'),
                            # Device Control Section
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Device Control", style={'color': 'white', 'marginBottom': '20px'}),
                                    # IR Control Buttons
                                    html.Div([
                                        dbc.Button("Power On", id='power-on-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                        dbc.Button("Power Off", id='power-off-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                        dbc.Button("Speed", id='speed-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                        dbc.Button("Timer", id='timer-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                        dbc.Button("Swing", id='swing-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                    ], className='button-container', style={'marginBottom': '20px'}),
                                    html.Div([  # Fan/Lights Toggle
                                        dbc.Button("Lights On", id='lights-on-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none', 'marginRight': '10px'}),
                                        dbc.Button("Lights Off", id='lights-off-btn', className='control-btn',
                                                   style={'backgroundColor': '#00FFFF', 'color': '#252A4B',
                                                          'border': 'none'}),
                                    ], style={'textAlign': 'center'}),
                                    # Status indicator
                                    html.Div(id='device-control-status',
                                             style={'color': 'cyan', 'marginTop': '20px', 'textAlign': 'center'})
                                ])
                            ], style={
                                'backgroundColor': '#1e1e3f',
                                'border': '1px solid #00FFFF',
                                'marginBottom': '20px'
                            }),
                            # Device Status
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Device Status", className='status-header'),
                                    html.Div(id='device-status', children="Ready to connect...", className='status-content')
                                ])
                            ], className='status-card')
                        ], className='left-column'),
                        # Right Column - Schedule Display
                        html.Div([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Room Schedule", className='card-header'),
                                    html.Div(
                                        id='room-schedule-4container',
                                        style={
                                            'height': '370px',  # Adjust this value as needed
                                            'overflowY': 'auto',
                                            'padding': '10px'
                                        },
                                        children=[
                                            html.Img(
                                                id='room-schedule-image',
                                                style={
                                                    'width': '100%',
                                                    'height': 'auto',
                                                    'display': 'block'
                                                }
                                            )
                                        ]
                                    )
                                ])
                            ], className='schedule-4card')
                        ], className='right-4column')
                    ], className='tab-4content')
                ], className='tab-4container'),

                # Hidden div to store schedule data
                dcc.Store(id='room-schedule-data', data={}),

                # Interval for updating device status
                dcc.Interval(
                    id='device-status-interval',
                    interval=5 * 1000,  # 5 seconds
                    n_intervals=0
            )
        ])
    ])
    return none


@app.callback(
    [Output('kwh-result', 'children'),
     Output('rate-result', 'children'),
     Output('cost-result', 'children'),
     Output('recommendations-list', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('date-picker', 'date'),
     State('room-dropdown', 'value')])
def update_prediction(n_clicks, selected_date, room):
    if n_clicks and n_clicks > 0 and selected_date:
        try:
            selected_date = datetime.strptime(selected_date[:10], "%Y-%m-%d")
            year = selected_date.year
            month = selected_date.month
            day = selected_date.day
            room_number = int(room)

            # Initialize predictor (make sure this is defined somewhere in your app)
            predictor = EnergyPredictor()  # You might want to initialize this outside the callback for better performance

            # Get prediction
            result = predictor.predict(year, month, day, room_number)

            # TEMPORARY DEBUG PRINT
            # print(
            #     f"DEBUG - Returned values: kWh={result['energy_kWh']}, Rate={result['energy_rate']}, Cost={result['energy_cost_Php']}")

            # Create list items for recommendations (keep your nice styling)
            rec_items = [html.Li(rec, style={'color': 'lightgreen', 'marginBottom': '8px'})
                         for rec in result.get('recommendations', [])]

            if not rec_items:  # Default recommendations if none matched
                rec_items = [
                    html.Li("No specific recommendations. Energy usage is within normal ranges.",
                            style={'color': 'lightgreen'})
                ]

            # Format the outputs with your existing number formatting
            return (
                f"{result['energy_kWh']:.2f} kWh",
                f"₱{result['energy_rate']:.2f}/kWh",
                f"₱{result['energy_cost_Php']:.2f}",
                rec_items
            )
        except Exception as e:
            print(f"Callback error: {e}")
            error_item = html.Li("Error loading recommendations", style={'color': 'red'})
            return "Error", "Error", "Error", [error_item]

    # Default return values
    return "0 kWh", "₱0.00/kWh", "₱0.00", [
        html.Li("Enter parameters and click 'Predict' to get recommendations",
                style={'color': 'lightgray'})
    ]


@app.callback(
    Output('notification-container', 'children'),
    [Input('notification-update-interval', 'n_intervals'),
     Input('tab1-room-dropdown', 'value')]  # Changed from room-dropdown to tab1-room-dropdown
)
def update_notifications(n_intervals, room):
    try:
        if not room:
            return []  # Correct - returning empty list

        # Get recent consumption data (last 14 days)
        recent_data = daily_room_df[
            (daily_room_df['room'].astype(str) == str(room)) &
            (daily_room_df['date'] >= (datetime.now().date() - timedelta(days=14)))
            ].sort_values('date')

        if recent_data.empty:
            return [
                html.Div("No recent data available for notifications",
                         style={'color': 'lightgray'})
            ]

        notifications = notifier.generate_notifications(room, recent_data, recent_data)

        if not notifications:
            return [
                html.Div("No notifications at this time",
                         style={'color': 'lightgray'})
            ]

        notification_elements = []
        for notification in notifications:
            bg_color = '#3b3f5c' if notification['priority'] == 'high' else '#2d324b'
            text_color = 'orange' if notification['priority'] == 'high' else 'lightblue'

            notification_elements.append(
                html.Div(
                    notification['message'],
                    style={
                        'color': text_color,
                        'backgroundColor': bg_color,
                        'padding': '10px',
                        'borderRadius': '5px',
                        'boxShadow': '0 0 10px rgba(0,255,255,0.3)',
                        'marginBottom': '10px'
                    }
                )
            )

        return notification_elements

    except Exception as e:
        print(f"Error generating notifications: {e}")
        return [
            html.Div("Error loading notifications", 
                     style={'color': 'red'})
        ]


@app.callback(
    Output('minute-level-graph', 'figure'),
    Output('total-energy', 'children'),
    Output('total-cost', 'children'),
    Output('avg-hourly', 'children'),
    Input('tab1-date-picker', 'date'),
    Input('tab1-room-dropdown', 'value'),
    Input('live-update-interval', 'n_intervals')
)
def update_graph_metrics(selected_date, selected_room, n_intervals):
    try:
        # Read and preprocess data
        df = pd.read_csv("assets/sample_data.csv")
        df.columns = [col.lower().replace(' ', '_').replace('(kwh)', 'kwh') for col in df.columns]

        df['timestamp'] = pd.to_datetime(
            df['date'].str.strip() + ' ' + df['time_hour'].str.strip(),
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        df = df.dropna(subset=['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce').fillna(0)

        # Create explicit copy for daily calculations
        daily_df = df.copy()
        daily_room_df = daily_df.groupby(['room', 'date'])['energy_kwh'].sum().reset_index()
        daily_room_df['cost'] = daily_room_df['energy_kwh'] * COST_PER_KWH

    except Exception as e:
        print(f"Error processing data: {e}")
        return px.line(), "0.00 kWh", "₱0.00", "0.00 kWh"

    # Initialize defaults
    fig = px.line(title='Select a date to view consumption').update_layout(
        plot_bgcolor='#1e1e3f',
        paper_bgcolor='#1e1e3f',
        font=dict(color='white'),
        title_font=dict(color='white')
    )
    day_total = "0.00 kWh"
    cost = "₱0.00"
    avg_text = "0.00 kWh"

    if selected_date and selected_room:
        selected_date_obj = datetime.strptime(selected_date[:10], "%Y-%m-%d").date()

        # Create explicit copy for filtered data
        filtered_df = df[
            (df['date'] == selected_date_obj) &
            (df['room'].astype(str) == str(selected_room))
            ].copy()

        if not filtered_df.empty:
            # Add hour column safely
            filtered_df.loc[:, 'hour'] = filtered_df['timestamp'].dt.hour

            # Update graph
            fig = px.line(
                filtered_df,
                x='timestamp',
                y='energy_kwh',  # Note: lowercase
                title=f"Energy Consumption on {selected_date_obj.strftime('%b %d, %Y')}",
                labels={"energy_kwh": "Energy (kWh)",
                        "timestamp": "Time"}
            ).update_layout(
                plot_bgcolor='#1e1e3f',
                paper_bgcolor='#1e1e3f',
                font=dict(color='white'),
                title_font=dict(color='white')
            )

            # Get totals
            day_data = daily_room_df[
                (daily_room_df['date'] == selected_date_obj) &
                (daily_room_df['room'].astype(str) == str(selected_room))
                ]

            if not day_data.empty:
                day_total = f"{day_data.iloc[0]['energy_kwh']:,.2f} kWh"
                cost = f"₱{day_data.iloc[0]['cost']:,.2f}"

            # Calculate hourly average
            hourly_avg = filtered_df.groupby('hour')['energy_kwh'].mean().mean()
            avg_text = f"{hourly_avg:,.5f} kWh"

    return fig, day_total, cost, avg_text

# JavaScript callback to handle WebSocket frames
app.clientside_callback(
    """
    function(n) {
        const ws = new WebSocket('ws://localhost:5001/ws');
        ws.onmessage = function(msg) {
            const blob = new Blob([msg.data], {type: 'image/jpeg'});
            const url = URL.createObjectURL(blob);
            document.getElementById('live-feed').src = url;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("live-feed", "src"),  # Dummy output
    Input("init-trigger", "n_intervals")
)


@app.callback(
    Output('live-consumption', 'children'),
    Output('live-consumption-store', 'data'),  # store the value
    Input('live-update-interval', 'n_intervals'),
    State('tab1-room-dropdown', 'value'),
    State('live-consumption-store', 'data'),  # get previous value
    prevent_initial_call=True
)
def update_live_consumption(n_intervals, selected_room, prev_value):
    try:
        live_df = pd.read_csv("assets/sample_data.csv")
        live_df.columns = [col.lower().replace(' ', '_').replace('(kwh)', 'kwh') for col in live_df.columns]
        live_df['timestamp'] = pd.to_datetime(
            live_df['date'].str.strip() + ' ' + live_df['time_hour'].str.strip(),
            format='%d/%m/%Y %H:%M:%S', errors='coerce'
        )
        live_df = live_df.dropna(subset=['timestamp'])

        room_data = live_df[live_df['room'].astype(str) == str(selected_room)]
        if room_data.empty:
            return prev_value, prev_value  # return previous if no new data

        latest = room_data.sort_values('timestamp', ascending=False).iloc[0]
        new_value = f"{latest['energy_kwh']:.5f} kWh"

        # Only update if new timestamp
        if prev_value != new_value:
            return new_value, new_value
        else:
            return prev_value, prev_value

    except Exception as e:
        print(f"Error reading live data: {e}")
        return prev_value, prev_value


@app.callback(
    Output('room-schedule-display', 'children'),
    Input('schedule-room-dropdown', 'value')
)
def update_schedule(selected_room):
    try:
        # Load the schedule data
        df = pd.read_csv('assets/room_schedule.csv')

        # Filter for selected room
        room_df = df[df['room'] == selected_room]

        # Define the order of days
        days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

        # Get all unique time slots and sort them properly
        time_slots = room_df['time_slot'].unique()

        # Sort time slots by converting to minutes since midnight
        def time_to_minutes(time_str):
            if ':' in time_str:
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            return 0

        time_order = sorted(time_slots, key=time_to_minutes)

        # Create an empty DataFrame with the correct structure
        schedule_df = pd.DataFrame(index=time_order, columns=days_order)

        # Fill the DataFrame with occupancy data
        for _, row in room_df.iterrows():
            if row['day'] in days_order and row['time_slot'] in time_order:
                schedule_df.loc[row['time_slot'], row['day']] = row['occupancy']

        # Fill NaN values with 0 (available)
        schedule_df = schedule_df.fillna(0)

        # Create the table header
        table_header = [html.Thead(html.Tr([html.Th("Time")] + [html.Th(day) for day in days_order]))]

        # Create table rows
        table_rows = []
        for time_slot in time_order:
            row_cells = [html.Td(time_slot)]
            for day in days_order:
                occupancy = schedule_df.loc[time_slot, day]
                cell_class = "occupied" if occupancy == 1 else ""
                row_cells.append(html.Td(className=cell_class))
            table_rows.append(html.Tr(row_cells))

        table_body = html.Tbody(table_rows)

        return dbc.Table(table_header + [table_body], bordered=True, className='schedule-table')

    except Exception as e:
        return html.Div(f"Error loading schedule: {str(e)}", style={'color': 'red'})


# Callbacks for device control
# Change the callback decorator to:
@app.callback(
    [Output('device-status', 'children'),
     Output('device-control-status', 'children')],  # Removed the toggle value output
    [Input('power-on-btn', 'n_clicks'),
     Input('power-off-btn', 'n_clicks'),
     Input('speed-btn', 'n_clicks'),
     Input('timer-btn', 'n_clicks'),
     Input('swing-btn', 'n_clicks'),
     Input('lights-on-btn', 'n_clicks'),  # Added
     Input('lights-off-btn', 'n_clicks'),  # Added
     Input('device-status-interval', 'n_intervals')],
    prevent_initial_call=True
)
def handle_device_control(power_on, power_off, speed, timer, swing, lights_on, lights_off, n_intervals):
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    status_message = ""
    device_status = dash.no_update

    try:
        # Handle button presses
        if triggered_id in ['power-on-btn', 'power-off-btn', 'speed-btn', 'timer-btn', 'swing-btn', 'lights-on-btn', 'lights-off-btn']:
            if triggered_id == 'power-on-btn':
                response = requests.get(f"{ESP32_BASE_URL}/power")
                status_message = "Power ON command sent"
            elif triggered_id == 'power-off-btn':
                response = requests.get(f"{ESP32_BASE_URL}/poweroff")
                status_message = "Power OFF command sent"
            elif triggered_id == 'speed-btn':
                response = requests.get(f"{ESP32_BASE_URL}/speed")
                status_message = "Speed command sent"
            elif triggered_id == 'timer-btn':
                response = requests.get(f"{ESP32_BASE_URL}/timer")
                status_message = "Timer command sent"
            elif triggered_id == 'swing-btn':
                response = requests.get(f"{ESP32_BASE_URL}/swing")
                status_message = "Swing command sent"
            elif triggered_id == 'lights-on-btn':
                response = requests.get(f"{ESP32_BASE_URL}/toggle")  # Will turn on if off
                status_message = "Lights turned ON"
            elif triggered_id == 'lights-off-btn':
                response = requests.get(f"{ESP32_BASE_URL}/toggle")  # Will turn off if on
                status_message = "Lights turned OFF"

            # After any button press, get updated status
            status_response = requests.get(f"{ESP32_BASE_URL}/info", timeout=5)
            if status_response.status_code == 200:
                return status_response.text, status_message

        # Handle periodic status updates
        elif triggered_id == 'device-status-interval':
            status_response = requests.get(f"{ESP32_BASE_URL}/info", timeout=5)
            if status_response.status_code == 200:
                return status_response.text, dash.no_update

        return device_status, status_message

    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"

#Schedule
@app.callback(
    Output('room-schedule-image', 'src'),
    Input('schedule-room-dropdown', 'value')
)
def update_schedule(selected_room):
    try:
        image_data = generate_schedule_image(selected_room)
        return f"data:image/png;base64,{image_data}"
    except Exception as e:
        print(f"Error generating schedule: {e}")
        raise PreventUpdate

if __name__ == '__main__':
    app.run(debug=True)

