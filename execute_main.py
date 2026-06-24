#%%
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import time

print("Starting the prediction module with GUI...")

cmd1 = """
docker run --rm --gpus=all --shm-size=32G \
-v /project_data/mridian/mridian_tracking/nmuehlschlegel/utrecht_exp:/utrecht_exp \
-w /utrecht_exp \
gitlab.lrz.de:5005/lmuk-radonc-phys-res/nmuehlschlegel/utrecht_experiments:01 \
bash -c 'cd code/socket_experiments && which python && which pip && \
python -m pip install pyyaml && \
python /utrecht_exp/code/socket_experiments/prediction_module_nogui.py'
"""

cmd2 = """
docker run --rm --gpus=all --shm-size=32G \
-v /project_data/mridian/mridian_tracking/nmuehlschlegel/utrecht_exp:/utrecht_exp \
-w /utrecht_exp \
gitlab.lrz.de:5005/lmuk-radonc-phys-res/nmuehlschlegel/utrecht_experiments:seg_02 \
bash -c "apt update && \
apt install -y libzmq5 libprotobuf32 && \
pip install pymri-0.1.0-cp311-cp311-linux_x86_64.whl && \
python /utrecht_exp/segmentation/improved_tracking_module.py"
"""

p1 = subprocess.Popen(
    ["bash", "-c", cmd1],
    stdout=open("prediction.log", "w"),
    stderr=subprocess.STDOUT,
)

p2 = subprocess.Popen(
    ["bash", "-c", cmd2],
    stdout=open("tracking.log", "w"),
    stderr=subprocess.STDOUT,
)


