import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# Number of rows
num_rows = 100

# Define time series start
start_time = datetime(2025, 3, 17, 10, 0)

# Possible values for cell IDs
cell_ids = [101, 102, 103, 104, 105]

# Possible actions
actions = ["Increase Bandwidth", "Reduce Latency", "Optimize Power", "No Action"]

# Data storage
data = []

for i in range(num_rows):
    timestamp = start_time + timedelta(minutes=5 * i)  # Generate timestamps every 5 mins
    cell_id = random.choice(cell_ids)
    bandwidth = random.choice([20, 40, 60, 80, 100])  # MHz
    latency = random.randint(5, 20)  # ms
    throughput = random.randint(40, 80)  # Mbps
    packet_loss = round(random.uniform(0, 5), 2)  # Percentage
    jitter = round(random.uniform(1, 10), 2)  # ms
    action = random.choice(actions)  # Select an action randomly

    # Compute energy score based on given formula
    energy_score = (
        0.4 * bandwidth +
        0.3 * throughput -
        0.2 * latency -
        0.05 * packet_loss -
        0.05 * jitter
    )

    # Normalize energy_score between 0 and 1
    energy_score = (energy_score - 10) / (100 - 10)  # Assume min=10, max=100 for normalization
    energy_score = round(min(max(energy_score, 0), 1), 2)  # Ensure it's between 0 and 1

    # Assign energy efficiency label
    energy_label = "Efficient" if energy_score > 0.7 else "Inefficient"

    # Randomly assign anomaly labels (5% anomalies)
    label = 1 if random.random() < 0.05 else 0

    # Future bandwidth (lookahead 1 step)
    future_bandwidth = bandwidth if i < num_rows - 1 else None  # Last row has no future bandwidth

    # Append to data list
    data.append([
        latency, bandwidth, throughput, packet_loss, jitter,
        energy_score, energy_label, label, future_bandwidth, action,
        timestamp, cell_id
    ])

# Create DataFrame with proper column names
df = pd.DataFrame(data, columns=[
    "latency", "bandwidth", "throughput", "packet_loss", "jitter",
    "energy_score", "energy_label", "label", "future_bandwidth", "action",
    "timestamp", "cell_id"
])

# Save to CSV
df.to_csv("network_data.csv", index=False)

print("CSV file 'network_data.csv' has been created successfully!")
