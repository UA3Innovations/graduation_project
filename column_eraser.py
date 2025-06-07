import pandas as pd

# Load the 2-day CSV
df = pd.read_csv("passenger_flow_results_2gun.csv")

# Columns to erase
columns_to_clear = ['boarding', 'alighting', 'current_load', 'new_load', 'occupancy_rate']

# Erase contents by setting them to None (blank)
df[columns_to_clear] = None

# Save the blanked-out version
df.to_csv("passenger_flow_results_2gun_prophet.csv", index=False)

print("Selected columns have been cleared and saved to passenger_flow_results_2gun_prophet.csv")
