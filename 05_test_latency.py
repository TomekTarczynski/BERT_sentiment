import requests
import time
import argparse

parser = argparse.ArgumentParser(description="Set training parameters.")
parser.add_argument("--text", type=str, default="", help="The text to check the sentiment")
parser.add_argument("--n_times", type=int, default=128, help="Number of times to run the API")
args = parser.parse_args()
text = args.text
n_times = args.n_times

# API URL
url = "http://127.0.0.1:8000/predict"

# Input data
payload = {"text": f"{text}"}

# Measure latency
start_time = time.time()
for i in range(n_times):
    response = requests.post(url, json=payload)
end_time = time.time()

# Calculate elapsed time
latency = 1000 * (end_time - start_time)/n_times

print(f"Response: {response.json()}")
print(f"Latency: {latency:.1f} miliseconds")
