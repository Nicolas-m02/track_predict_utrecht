import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# cleanup any existing containers
try:
    subprocess.run(["docker", "rm", "-f", "prediction_container_main"], check=True)
    subprocess.run(["docker", "rm", "-f", "tracking_container_main"], check=True)
    
    subprocess.run(["docker", "rm", "-f", "gui_container_main"], check=True)

except subprocess.CalledProcessError as e:
    print(f"Error removing existing containers: {e}")
print("Removed existing containers.")

