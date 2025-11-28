import serial
import os
import datetime
import time

# Serial and file settings
arduino_port = "COM7"
baud = 115200
fileName = "assets/sample_data.csv"
headers = "Date,Time_Hour,Room,Room_Type,Voltage (V),Current (A),Energy (kWh)"

# Define rooms and their types
rooms = {
    "101": "lab", "102": "lab", "103": "lab", "104": "lab", "105": "lab", "106": "lab", "108": "lab", "109": "lab", "111": "lab",
    "201": "lab", "202": "lab", "203": "lab", "204": "lab", "205": "lab", "206": "lab", "207": "lab", "209": "lab",
    "210": "lab",
    "302": "lab", "304": "lab", "305": "lab", "306": "lab", "307": "lab", "309": "lab",
    "211": "lecture", "213": "lecture", "215": "lecture", "222": "lecture", "223": "lecture", "225": "lecture",
    "227": "lecture", "228": "lecture",
    "313": "lecture", "314": "lecture", "316": "lecture", "318": "lecture", "319": "lecture", "320": "lecture",
    "322": "lecture", "324": "lecture", "326": "lecture",
    "401": "lecture", "402": "lecture", "403": "lecture", "404": "lecture", "406": "lecture", "407": "lecture",
    "408": "lecture", "409": "lecture", "410": "lecture",
    "412": "lecture", "413": "lecture", "414": "lecture", "415": "lecture", "417": "lecture", "418": "lecture",
    "419": "lecture",
    "702": "lecture", "703": "lecture", "704": "lecture", "705": "lecture"
}

# Check file state
file_exists = os.path.exists(fileName)
is_empty = not file_exists or os.stat(fileName).st_size == 0

# Start serial communication
ser = serial.Serial(arduino_port, baud, timeout=1)
print("Connected to " + arduino_port)

# Initialize logging timestamp
next_log_time = (int(time.time()) // 60 + 1) * 60

# Energy values
latest_voltage = 0.0
latest_current = 0.0
latest_power_kW = 0.0
total_energy = 0.0
energy_buffer = []

# Open file and prepare logging
with open(fileName, 'a') as file:
    if is_empty:
        print("Creating new log file...")
        file.write(headers + '\n')
    else:
        print("Appending to existing file...")
        # Try reading last energy value
        with open(fileName, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                try:
                    total_energy = max(0.0, float(last_line[-1]))
                except (IndexError, ValueError):
                    total_energy = 0.0

    try:
        while True:
            current_time = time.time()

            # Read from serial
            getData = ser.readline().decode('utf-8', errors='ignore').strip()
            if getData:
                parts = [p.strip() for p in getData.split(',')]
                if len(parts) == 4:
                    try:
                        voltage = max(0.0, float(parts[0]))
                        current = max(0.0, float(parts[1]))
                        #power = max(0.0, float(parts[2]))
                        energy_reading = max(0.0, float(parts[3]))

                        # Store latest values
                        latest_voltage = voltage
                        latest_current = current
                        #latest_power_kW = power / 1000.0  # for reference

                        energy_buffer.append(energy_reading)

                    except ValueError:
                        print(f"Skipping malformed data: {getData}")
                else:
                    print(f"Ignoring unexpected format: {getData}")

            # Log every full minute
            if current_time >= next_log_time:
                energy_per_minute = sum(energy_buffer)
                energy_buffer.clear()  # reset for next minute

                now = datetime.datetime.now()
                date_str = now.strftime("%d/%m/%Y")
                time_str = now.strftime("%H:%M:%S")

                for room, room_type in rooms.items():
                    line = f"{date_str},{time_str},{room},{room_type},{latest_voltage},{latest_current},{energy_per_minute:.8f}"
                    file.write(line + '\n')
                    print(f"Logged @ {time_str}: Room {room}")

                file.flush()
                next_log_time = (int(current_time) // 60 + 1) * 60

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        ser.close()
        print("Serial connection closed.")
