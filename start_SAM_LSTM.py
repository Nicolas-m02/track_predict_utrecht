#### still racing condition in, both have to run asynchronously


import subprocess
import time
import socket
import os


HOST = "0.0.0.0"
READY_PORT = 9002


env = os.environ.copy()
env["PYTHONPATH"] = "/utrecht_exp/segmentation:/utrecht_exp"

PYTHON = "/opt/conda/bin/python"

def wait_for_ready(host, port):
    while True:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            return
        except OSError:
            time.sleep(0.1)


p1 = subprocess.Popen([PYTHON, "/utrecht_exp/code/socket_experiments/prediction_module_nogui.py"], env=env)

wait_for_ready(HOST, READY_PORT)

p2 = subprocess.Popen([PYTHON, "/utrecht_exp/segmentation/improved_tracking_module.py"], env=env)

p1.wait()
p2.wait()

#p3 = subprocess.Popen([
#    "/opt/conda/bin/python",
#    "/utrecht_exp/code/socket_experiments/.py"
#])