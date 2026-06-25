#%%
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['settings']['enable_gui']:
    script_pred = 'prediction_module_with_gui.py'
    if config['settings']['interactive']:
        script_track = 'point_tracking_module.py'
        gui_loc = 'gui_interactive'
        print("Starting the tracker in interactive mode...")
    else:
        script_track = 'improved_tracker_with_gui_support.py'
        gui_loc = 'gui'
        print("Starting the tracker with GUI support...")


else:
    script_pred = 'prediction_module_nogui.py'
    script_track = 'improved_tracking_module.py'
    print("Starting the modules without GUI...")

# cleanup any existing containers
try:
    subprocess.run(["docker", "rm", "-f", "prediction_container"], check=True)
    subprocess.run(["docker", "rm", "-f", "tracking_container"], check=True)
    if config['settings']['enable_gui']:
        subprocess.run(["docker", "rm", "-f", "gui_container"], check=True)

    print("Removed existing containers if any.")
except subprocess.CalledProcessError as e:
    print(f"Error removing existing containers: {e}")


print(f'Running {script_pred} and {script_track} in Docker containers...')

cmd1 = f"""
docker run --rm --name {config["paths"]["pred_name"]} --gpus=all --shm-size=32G \
-v {config["paths"]["code_path"]}:/utrecht_exp \
--network network_testing \
-w /utrecht_exp \
gitlab.lrz.de:5005/lmuk-radonc-phys-res/nmuehlschlegel/utrecht_experiments:01 \
bash -c 'cd code/socket_experiments && \
python -m pip install pyyaml && \
python {script_pred}'
"""

cmd2 = f"""
docker run --rm --name {config["paths"]["track_name"]} --gpus=all --shm-size=32G \
-v {config["paths"]["code_path"]}:/utrecht_exp \
--network network_testing \
-w /utrecht_exp \
gitlab.lrz.de:5005/lmuk-radonc-phys-res/nmuehlschlegel/utrecht_experiments:seg_02 \
bash -c "apt update && \
apt install -y libzmq5 libprotobuf32 && \
pip install pymri-0.1.0-cp311-cp311-linux_x86_64.whl && \
cd segmentation && \
python {script_track}"
"""

if config['settings']['enable_gui']:
    print("Starting the GUI")
    cmd3 = f"""
    docker run --rm --name {config["paths"]["gui_name"]} --gpus=all --shm-size=32G \
    -v {config["paths"]["code_path"]}:/utrecht_exp \
    --network network_testing \
    -p {config['ports']['port_gui_ext']}:{config['ports']['port_gui_ext']} \
    -w /utrecht_exp \
    gitlab.lrz.de:5005/lmuk-radonc-phys-res/nmuehlschlegel/utrecht_experiments:gui_02 \
    bash -c "pip install uvicorn['standard'] opencv-python-headless SimpleITK fastapi && \
    cd /utrecht_exp/{gui_loc} && \
    uvicorn app2:app --reload --host 0.0.0.0 --port {config['ports']['port_gui_ext']}"
    """


    


    

p1 = subprocess.Popen(
    ["bash", "-c", cmd1],
    stdout=open("prediction.log", "w"),
    stderr=subprocess.STDOUT,
)

if config['settings']['enable_gui']:
    print("Starting the prediction module with GUI...")
    time.sleep(10)
    print(f"Pinging adress at {config['ports']['port_gui_ext']}")
    p4 = subprocess.Popen(
        ["bash", "-c", f"curl http://localhost:{config['ports']['port_gui_ext']}/"],
        stdout=open("curl.log", "w"),
        stderr=subprocess.STDOUT,
    )
    print("Starting GUI ...")
    p3 = subprocess.Popen(
        ["bash", "-c", cmd3],
        stdout=open("gui.log", "w"),
        stderr=subprocess.STDOUT,
    )
else:
    print("Starting the prediction module...")

time.sleep(15)


print("Starting tracker ...")
p2 = subprocess.Popen(
    ["bash", "-c", cmd2],
    stdout=open("tracking.log", "w"),
    stderr=subprocess.STDOUT,
)

print("Startup complete")


