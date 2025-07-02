import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import subprocess
import time

# Function to run hackrf_sweep and save output
def run_hackrf_sweep(freq_range, output_file):
    cmd = f"hackrf_sweep -f {freq_range} -w 600000 -l 40 -g 20 -r {output_file}"
    subprocess.run(cmd, shell=True)
    return output_file

# Process CSV data
def process_csv(csv_file):
    df = pd.read_csv(csv_file, names=['date', 'time', 'hz_low', 'hz_high', 'hz_bin_width', 'num_samples'] + [f'db_{i}' for i in range(100)])
    signal_strength = df[[f'db_{i}' for i in range(100)]].mean(axis=1)
    return signal_strength

# Load trained model (assumes pre-trained model)
def load_model():
    model = RandomForestClassifier()
    # Load pre-trained model (replace with actual model loading)
    return model

# Main detection loop
def detect_drones():
    model = load_model()
    while True:
        # Scan 2.4 GHz
        run_hackrf_sweep("2400:2483", "output_2g.csv")
        signal_2g = process_csv("output_2g.csv")
        
        # Scan 5.8 GHz
        run_hackrf_sweep("5725:5875", "output_5g.csv")
        signal_5g = process_csv("output_5g.csv")
        
        # Combine features
        features = np.concatenate([signal_2g, signal_5g])
        prediction = model.predict([features])[0]
        
        if prediction == 1:
            print("Drone detected!")
        else:
            print("No drone detected.")
        
        time.sleep(10)  # Scan every 10 seconds

if __name__ == "__main__":
    detect_drones()