import dash
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from energy_predictor import EnergyPredictor
from flask_caching import Cache
from datetime import date, datetime, timedelta
from notifications import NotificationSystem
from dash import ctx  # to check which input triggered


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Set up caching (replaces @st.cache_resource)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})

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

df_rooms = pd.read_csv("assets/sched_database.csv")
available_rooms = sorted(df_rooms['room'].astype(str).unique()) # sched_database.csv

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
                                html.H4("Daily Average Consumption", style={'color': 'white', 'textAlign': 'center'}),
                                html.H3(id='avg-daily', style={
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
            'width': '1680px',
            'padding': '0 40px'
        }),
        dcc.Store(id='live-consumption-store', data="0.00 kWh")
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
                            html.H3(id='rate-result', children="₱0.00/kWh",
                                    style={'color': 'cyan', 'textAlign': 'center', 'fontSize': '2.5rem'})
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
            'width': '1680px',
        })

    elif tab == 'tab-3':
        return html.Div([
            html.H3("Live Camera Feed", style={'color': 'white', 'textAlign': 'center'}),
            html.Div([
                html.Img(
                    src="http://127.0.0.1:5001/video_feed",  # or use your LAN IP
                    style={
                        'width': '800px',
                        'height': '600px',
                        'border': '5px solid #00FFFF',
                        'borderRadius': '10px',
                        'boxShadow': '0 0 20px rgba(0,255,255,0.5)'
                    }
                )
            ], style={
                'textAlign': 'center'
            })
        ], style={
            'width': '1680px',
        })

    elif tab == 'tab-4':  # Changed from `if` to `elif` to properly chain conditions
        return html.H3('Room Schedule Content', style={'color': 'white'})
    else:
        return html.H3('No content available', style={'color': 'white'})


@app.callback(
    [Output('kwh-result', 'children'),
     Output('rate-result', 'children'),
     Output('cost-result', 'children'),
     Output('recommendations-list', 'children')],  # Add this output
    [Input('predict-button', 'n_clicks')],
    [State('date-picker', 'date'),
     State('room-dropdown', 'value')]
)
def update_prediction(n_clicks, selected_date, room):
    if n_clicks and n_clicks > 0 and selected_date:
        try:
            selected_date = datetime.strptime(selected_date[:10], "%Y-%m-%d")
            year = selected_date.year
            month = selected_date.month
            day = selected_date.day
            room_number = int(room)

            result = predictor.predict(year, month, day, room_number)

            # Create list items for recommendations
            rec_items = [html.Li(rec, style={'color': 'lightgreen', 'marginBottom': '8px'})
                         for rec in result.get('recommendations', [])]

            if not rec_items:  # Default recommendations if none matched
                rec_items = [
                    html.Li("No specific recommendations. Energy usage is within normal ranges.",
                            style={'color': 'lightgreen'})
                ]

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
    Output('avg-daily', 'children'),
    Input('tab1-date-picker', 'date'),
    Input('tab1-room-dropdown', 'value'),
    Input('live-update-interval', 'n_intervals')  # 👈 added
)
def update_graph_metrics(selected_date, selected_room, n_intervals):
    # Re-read and preprocess CSV for live updates
    try:
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

        # Precompute daily energy & cost
        daily_room_df = df.groupby(['room', 'date'])['energy_kwh'].sum().reset_index()
        daily_room_df['cost'] = daily_room_df['energy_kwh'] * COST_PER_KWH
    except Exception as e:
        print(f"Error processing sample_data.csv: {e}")
        return px.line(), "0.00 kWh", "₱0.00", "0.00 kWh"

    # Initialize default values
    fig = px.line(title='Select a date to view minute-level consumption').update_layout(
        plot_bgcolor='#1e1e3f',
        paper_bgcolor='#1e1e3f',
        font=dict(color='white'),
        title_font=dict(color='white')
    )
    day_total = "0.00 kWh"
    cost = "₱0.00"
    room_avg = "0.00 kWh"

    if selected_date and selected_room:
        selected_date_obj = datetime.strptime(selected_date[:10], "%Y-%m-%d").date()

        # Filter data for selected date and room
        filtered_df = df[(df['date'] == selected_date_obj) & (df['room'].astype(str) == str(selected_room))]

        if not filtered_df.empty:
            # Update graph
            fig = px.line(
                filtered_df,
                x='timestamp',
                y='energy_kwh',
                title=f"Energy Consumption on {selected_date_obj.strftime('%b %d, %Y')} - Room {selected_room}",
                labels={"timestamp": "Time", "energy_kwh": "Energy Consumption (kWh)"}
            ).update_layout(
                plot_bgcolor='#1e1e3f',
                paper_bgcolor='#1e1e3f',
                font=dict(color='white'),
                title_font=dict(color='white')
            )

            # Update metrics
            day_data = daily_room_df[(daily_room_df['date'] == selected_date_obj) &
                                     (daily_room_df['room'].astype(str) == str(selected_room))]
            if not day_data.empty:
                day_total = f"{day_data.iloc[0]['energy_kwh']:,.2f} kWh"
                cost = f"₱{day_data.iloc[0]['cost']:,.2f}"

            room_avg = daily_room_df[daily_room_df['room'].astype(str) == str(selected_room)]['energy_kwh'].mean()
            room_avg = f"{0 if pd.isna(room_avg) else room_avg:,.2f} kWh"

    return fig, day_total, cost, room_avg


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
        new_value = f"{latest['energy_kwh']:.2f} kWh"

        # Only update if new timestamp
        if prev_value != new_value:
            return new_value, new_value
        else:
            return prev_value, prev_value

    except Exception as e:
        print(f"Error reading live data: {e}")
        return prev_value, prev_value

if __name__ == '__main__':
    app.run(debug=True)


